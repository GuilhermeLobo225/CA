"""SmartHandover - Synthetic-text quality filter.

Two-stage filter applied to ``text.jsonl`` (output of ``generate_text``):

  Stage A - heuristics
      length bounds, deduplication, banned-token check, emoji rejection,
      detection of meta-leaks ("As an AI"...), label/cause/persona leaks
      ("she sounds frustrated...").

  Stage B - LLM judge
      A second LLM call asks: "Does this utterance plausibly express
      <label> at intensity <intensity>/5 in a customer-support context?
      Score 1-5 and return JSON". Records below ``MIN_JUDGE_SCORE`` are
      rejected.

The script is **resumable**: rejection / acceptance verdicts are cached
in ``text_judged.jsonl`` so re-runs only judge new samples.

Outputs
-------
* ``data/synthetic/text_filtered.jsonl`` (kept records, full schema)
* ``data/synthetic/text_rejected.jsonl`` (with reason + score)
* ``data/synthetic/text_judged.jsonl`` (cache - id -> verdict)
* console summary with counts per rejection reason

Usage
-----
    python -m src.data.synthetic.filter_text                # default
    python -m src.data.synthetic.filter_text --no-judge     # heuristics only
    python -m src.data.synthetic.filter_text --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from src.data.synthetic import config as cfg
from src.data.synthetic._openai_client import (
    append_jsonl,
    get_pool,
    load_dotenv,
    loaded_ids_from_jsonl,
    with_pool_backoff,
)
from src.data.synthetic.text_normalize import normalise_record

JUDGED_CACHE = os.path.join(cfg.OUT_DIR, "text_judged.jsonl")

# ---------------------------------------------------------------------------
# Stage A - heuristic filter
# ---------------------------------------------------------------------------


_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]+"
)

_BANNED_PHRASES = [
    "as an ai", "as a language model", "i cannot", "i'm sorry, but",
    "openai", "i was generated", "[insert", "<insert",
    "the customer", "the speaker", "the persona",
    # narrator-mode leaks where the model describes the utterance
    " sounds frustrated", " is frustrated about", " is angry about",
    "the user is", "the customer is",
]

_LABEL_LEAK_RE = re.compile(
    r"\b(label|persona|style|cause|intensity)\b[: =]",
    re.IGNORECASE,
)


def heuristic_check(record: Dict) -> Tuple[bool, str]:
    """Return (ok, reason). ``ok=True`` if record passes all heuristics."""
    text = (record.get("text") or "").strip()
    if not text:
        return False, "empty"

    # Length
    words = text.split()
    if len(words) < cfg.MIN_WORDS:
        return False, f"too_short({len(words)}w)"
    if len(words) > cfg.MAX_WORDS:
        return False, f"too_long({len(words)}w)"

    # Emoji
    if _EMOJI_RE.search(text):
        return False, "emoji"

    # Surrounding quotes (model-escape)
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        return False, "wrapped_in_quotes"

    # Common LLM meta leaks
    lower = text.lower()
    for phrase in _BANNED_PHRASES:
        if phrase in lower:
            return False, f"banned_phrase('{phrase}')"

    # Diversity-axis leak ("Style: sarcastic")
    if _LABEL_LEAK_RE.search(text):
        return False, "axis_leak"

    return True, "ok"


# ---------------------------------------------------------------------------
# Stage B - LLM judge
# ---------------------------------------------------------------------------


JUDGE_SYSTEM_PROMPT = (
    "You are an emotion-annotation expert. Given an utterance allegedly "
    "spoken by a customer on a support call, judge whether it plausibly "
    "expresses the labelled emotion at the labelled intensity.\n\n"
    "Return STRICT JSON with these fields:\n"
    "  score          : integer 1-5  (1=label clearly wrong, 5=label clearly right)\n"
    "  intensity_obs  : integer 1-5  (your read of the actual intensity)\n"
    "  natural        : boolean      (does it sound like spontaneous speech?)\n"
    "  reason         : short string (<=140 chars)\n"
    "Output ONLY the JSON object, no preamble."
)


def _judge_user_prompt(record: Dict) -> str:
    return (
        f"Utterance:\n  \"{record['text']}\"\n\n"
        f"Claimed label    : {record['label']}\n"
        f"Claimed intensity: {record['intensity']}/5\n"
        f"Claimed style    : {record['style']}\n"
        f"Claimed cause    : {record['cause']}\n"
    )


def _safe_json_parse(s: str) -> Optional[Dict]:
    """Parse JSON, salvaging the case where the model wraps it in code fences."""
    s = s.strip()
    if s.startswith("```"):
        # Remove the first fence (with optional language tag)
        s = re.sub(r"^```(json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Try to extract the first {...} block
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def judge_one(record: Dict) -> Dict:
    """Send one judge request. Returns a verdict dict.

    Verdict keys: id, score, intensity_obs, natural, reason, ok, fail_reason

    Routed via the dedicated judge pool (Ollama by default, see
    cfg.JUDGE_PROVIDER) so the model that grades is independent of the
    one that generated the text.
    """
    pool = get_pool("judge")

    def _call(client):
        return client.chat.completions.create(
            model=cfg.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": _judge_user_prompt(record)},
            ],
            temperature=0.1,
            max_tokens=200,
            response_format={"type": "json_object"},
        )

    try:
        resp = with_pool_backoff(pool, _call)
    except Exception as e:
        return {"id": record["id"], "ok": False,
                "fail_reason": f"judge_api_error:{type(e).__name__}"}

    parsed = _safe_json_parse(resp.choices[0].message.content or "")
    if not parsed:
        return {"id": record["id"], "ok": False,
                "fail_reason": "judge_unparseable"}

    score = int(parsed.get("score", 0))
    intensity_obs = int(parsed.get("intensity_obs", 0))
    natural = bool(parsed.get("natural", True))
    reason = str(parsed.get("reason", ""))[:200]

    intensity_diff = abs(intensity_obs - record["intensity"])
    ok = (
        score >= cfg.MIN_JUDGE_SCORE
        and intensity_diff <= cfg.MIN_JUDGE_INTENSITY_AGREE + 1
        # +1 because the judge's intensity reading has its own noise
    )
    fail_reason = ""
    if not ok:
        if score < cfg.MIN_JUDGE_SCORE:
            fail_reason = f"low_judge_score({score})"
        else:
            fail_reason = f"intensity_mismatch({intensity_obs}vs{record['intensity']})"

    return {
        "id":            record["id"],
        "score":         score,
        "intensity_obs": intensity_obs,
        "natural":       natural,
        "reason":        reason,
        "ok":            ok,
        "fail_reason":   fail_reason,
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def _read_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _write_jsonl(path: str, records: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Filter synthetic-text JSONL via heuristics + LLM judge."
    )
    p.add_argument("--input",  type=str, default=cfg.TEXT_OUT)
    p.add_argument("--output", type=str, default=cfg.TEXT_FILTERED)
    p.add_argument("--rejected", type=str, default=cfg.TEXT_REJECTED)
    p.add_argument("--workers", type=int, default=cfg.JUDGE_CONCURRENCY)
    p.add_argument("--no-judge", action="store_true",
                   help="Skip the LLM judge stage (heuristics only).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    load_dotenv()
    cfg.ensure_dirs()

    print("=" * 72)
    print("  SmartHandover - Synthetic Text Filter")
    print("=" * 72)
    print(f"  input        : {args.input}")
    print(f"  output       : {args.output}")
    print(f"  rejected     : {args.rejected}")
    print(f"  judge        : {'OFF (heuristics only)' if args.no_judge else cfg.JUDGE_MODEL}")
    print(f"  workers      : {args.workers}")

    records = _read_jsonl(args.input)
    print(f"  records read : {len(records)}")
    if not records:
        print("  Nothing to do.")
        return

    # --- Stage 0: punctuation normalisation -------------------------------
    # GPT-4o emits curly quotes / em-dashes that don't match the MELD
    # tokenisation. Flatten everything to ASCII before any heuristic or
    # judge sees the text. The kept records are written with the
    # normalised form so the rest of the pipeline (RoBERTa fine-tune,
    # VADER, GoEmotions, TTS) receives consistent input.
    n_modified = 0
    for i, rec in enumerate(records):
        original = rec.get("text", "")
        new = normalise_record(rec)
        if new["text"] != original:
            n_modified += 1
        records[i] = new
    print(f"  normalised   : {n_modified} record(s) had smart punctuation "
          "flattened")

    # --- Stage A: heuristics ----------------------------------------------
    heuristic_pass: List[Dict] = []
    rejected: List[Dict] = []
    reason_counts: Dict[str, int] = {}

    for rec in tqdm(records, desc="heuristics", unit="rec"):
        ok, reason = heuristic_check(rec)
        if ok:
            heuristic_pass.append(rec)
        else:
            rejected.append({**rec, "fail_stage": "heuristic",
                             "fail_reason": reason})
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print(f"\n  heuristic pass: {len(heuristic_pass)} / {len(records)}  "
          f"(rejected {len(rejected)})")

    # --- Stage B: judge (optional) ----------------------------------------
    judged_verdicts: Dict[str, Dict] = {}
    if not args.no_judge and heuristic_pass:
        # Resume cache
        cached = _read_jsonl(JUDGED_CACHE)
        for v in cached:
            if "id" in v:
                judged_verdicts[v["id"]] = v
        if judged_verdicts:
            print(f"  cache hit    : {len(judged_verdicts)} previously judged")

        to_judge = [r for r in heuristic_pass if r["id"] not in judged_verdicts]
        print(f"  to judge     : {len(to_judge)}")

        if to_judge:
            try:
                judge_pool = get_pool("judge")
            except RuntimeError as e:
                print(f"\n[ERROR] {e}", file=sys.stderr)
                sys.exit(2)
            print()
            print(judge_pool.summary())
            print(f"  judge model  : {cfg.JUDGE_MODEL}")
            print(f"  provider     : {cfg.JUDGE_PROVIDER}")
            print()

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(judge_one, r): r for r in to_judge}
                with tqdm(total=len(futures), desc="judging", unit="rec") as bar:
                    for fut in as_completed(futures):
                        verdict = fut.result()
                        judged_verdicts[verdict["id"]] = verdict
                        # Cache as we go
                        append_jsonl(JUDGED_CACHE, verdict)
                        bar.set_postfix(
                            ok=sum(1 for v in judged_verdicts.values()
                                   if v.get("ok"))
                        )
                        bar.update(1)

    # --- Final selection ---------------------------------------------------
    kept: List[Dict] = []
    for rec in heuristic_pass:
        if args.no_judge:
            kept.append(rec)
            continue
        v = judged_verdicts.get(rec["id"])
        if v and v.get("ok"):
            # Augment record with judge fields for traceability
            kept.append({**rec,
                         "judge_score": v.get("score"),
                         "judge_intensity_obs": v.get("intensity_obs"),
                         "judge_natural": v.get("natural")})
        else:
            reason = (v or {}).get("fail_reason", "judge_no_verdict")
            rejected.append({**rec, "fail_stage": "judge",
                             "fail_reason": reason})
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    # --- Write outputs -----------------------------------------------------
    _write_jsonl(args.output,  kept)
    _write_jsonl(args.rejected, rejected)

    # --- Summary -----------------------------------------------------------
    print()
    print("=" * 72)
    print(f"  kept      : {len(kept)} ({100 * len(kept) / max(len(records), 1):.1f}%)")
    print(f"  rejected  : {len(rejected)}")
    print(f"  output    : {args.output}")
    print(f"  rejected -> {args.rejected}")
    print("\n  Rejection reasons (top 10):")
    for reason, count in sorted(reason_counts.items(),
                                key=lambda kv: -kv[1])[:10]:
        print(f"    {count:>5d}  {reason}")

    # --- Per-class stats ---------------------------------------------------
    by_class: Dict[str, int] = {}
    for r in kept:
        by_class[r["label"]] = by_class.get(r["label"], 0) + 1
    print("\n  Kept per class:")
    for label in cfg.TARGET_LABELS:
        print(f"    {label:<14s} {by_class.get(label, 0):>5d}")


if __name__ == "__main__":
    main()
