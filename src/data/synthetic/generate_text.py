"""SmartHandover - Synthetic text generator.

Generates ~20 000 customer-support utterances over the 5 SmartHandover
classes, each conditioned on a unique point of the 5-axis diversity
space (intensity x cause x style x persona x turn_position).

Design choices
--------------
* **Single utterance per request** so we can inject a different
  diversity point in every call. Batching multiple utterances per call
  collapses style diversity within a batch.
* **Resumable**. Records are appended to a .jsonl as they are produced;
  on restart we read existing IDs and skip them. Safe to ``Ctrl+C`` and
  re-run.
* **Concurrent**. ThreadPoolExecutor with a configurable worker count
  (default 8) hits ~6-12 RPS comfortably under OpenAI Tier-1 limits.
* **Reproducible**. Each (label, sample_index_within_label) deterministically
  maps to one diversity point via a class-specific seed. Two runs with
  the same SEED produce the exact same diversity assignments.

Output format (.jsonl)
----------------------
Each line is a JSON object::

    {
      "id":            "frust_00042",
      "label":         "frustration",
      "label_id":      1,
      "intensity":     4,
      "cause":         "billing_error",
      "style":         "exasperated",
      "persona":       "elderly_formal",
      "turn_position": "escalation",
      "text":          "I have been calling for three days about a charge ...",
      "model":         "gpt-4o-mini",
      "prompt_hash":   "sha1:...",
      "ts":            1714400000.123
    }

Usage
-----
    python -m src.data.synthetic.generate_text                      # full 20k run
    python -m src.data.synthetic.generate_text --limit 50           # smoke test
    python -m src.data.synthetic.generate_text --label frustration  # one class only
    python -m src.data.synthetic.generate_text --workers 4          # gentler RPS
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm

from src.data.synthetic import config as cfg
from src.data.synthetic._openai_client import (
    append_jsonl,
    describe_pools,
    get_pool,
    load_dotenv,
    loaded_ids_from_jsonl,
    with_pool_backoff,
)
from src.data.synthetic.diversity import enumerate_points

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are a data-generation tool that writes single utterances spoken by "
    "a customer to a customer-support agent during a phone call. Your job "
    "is to produce one realistic utterance that exactly matches the "
    "specification provided by the user.\n\n"
    "Output rules:\n"
    " - Output ONLY the utterance text, no quotes, no preamble, no labels.\n"
    " - 4 to 60 words. Prefer 8-30 words for most cases.\n"
    " - Sound like spontaneous spoken English, not a script. Contractions, "
    "filler words ('uh', 'I mean', 'like'), false starts, and short "
    "sentences are encouraged.\n"
    " - Do NOT mention the persona/style/cause labels explicitly. The "
    "emotion should be conveyed by what is said and how, not stated.\n"
    " - Do NOT use emojis. ASCII punctuation only.\n"
    " - Do NOT include the agent's response."
)


def _user_prompt(point: Dict) -> str:
    intensity_word = {1: "very mild", 2: "mild", 3: "moderate",
                      4: "strong", 5: "extreme"}[point["intensity"]]
    return (
        f"Generate ONE customer-support utterance with these properties:\n"
        f"  Emotion: {point['label']}\n"
        f"  Intensity: {intensity_word} ({point['intensity']}/5)\n"
        f"  Cause / topic: {point['cause']}\n"
        f"  Style: {point['style']}\n"
        f"  Speaker persona: {point['persona']}\n"
        f"  Turn position in call: {point['turn_position']}"
    )


def _prompt_hash(point: Dict) -> str:
    """Stable hash of the prompt parameters - lets us audit collisions."""
    h = hashlib.sha1(
        json.dumps(point, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"sha1:{h[:16]}"


# ---------------------------------------------------------------------------
# Sample-id generation
# ---------------------------------------------------------------------------


_LABEL_SHORT = {
    "anger":        "anger",
    "frustration":  "frust",
    "sadness":      "sad",
    "neutral":      "neut",
    "satisfaction": "satis",
}


def _sample_id(label: str, idx: int) -> str:
    return f"{_LABEL_SHORT[label]}_{idx:05d}"


# ---------------------------------------------------------------------------
# Per-sample generation
# ---------------------------------------------------------------------------


def _generate_one(point: Dict, sample_id: str) -> Optional[Dict]:
    """Call the LLM for ``point`` and return a record dict (or None on failure)."""
    pool = get_pool("text")

    def _call(client):
        return client.chat.completions.create(
            model=cfg.TEXT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _user_prompt(point)},
            ],
            temperature=0.95,
            top_p=0.95,
            max_tokens=180,
        )

    try:
        resp = with_pool_backoff(pool, _call)
    except Exception as e:
        # Surface but do not crash the whole run
        print(f"\n  [WARN] {sample_id} ({point['label']}): "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return None

    text = (resp.choices[0].message.content or "").strip()
    # Strip stray quotes that some models add even when told not to
    text = text.strip('"').strip("'").strip()
    if not text:
        return None

    return {
        "id":            sample_id,
        "label":         point["label"],
        "label_id":      cfg.TARGET_LABEL2ID[point["label"]],
        "intensity":     point["intensity"],
        "cause":         point["cause"],
        "style":         point["style"],
        "persona":       point["persona"],
        "turn_position": point["turn_position"],
        "text":          text,
        "model":         cfg.TEXT_MODEL,
        "prompt_hash":   _prompt_hash(point),
        "ts":            time.time(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic customer-support utterances."
    )
    p.add_argument("--limit", type=int, default=None,
                   help="Cap total samples (useful for smoke tests).")
    p.add_argument("--label", type=str, default=None,
                   choices=cfg.TARGET_LABELS,
                   help="Generate a single class only.")
    p.add_argument("--workers", type=int, default=cfg.TEXT_CONCURRENCY,
                   help=f"Concurrent worker threads (default {cfg.TEXT_CONCURRENCY}).")
    p.add_argument("--out", type=str, default=cfg.TEXT_OUT,
                   help="Output .jsonl path.")
    p.add_argument("--seed", type=int, default=cfg.SEED,
                   help="Diversity-sampler seed (default from config).")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore existing output and start fresh.")
    p.add_argument("--preview", type=int, default=0, metavar="N",
                   help="Show N example prompts per class WITHOUT calling "
                        "any API. Useful to audit the diversity space "
                        "before spending credits.")
    return p.parse_args(argv)


def _preview(args: argparse.Namespace, n_per_class: int) -> None:
    """Render the prompts that *would* be sent for ``n_per_class`` samples
    per class, without doing any API call. Pure local computation."""
    distribution = cfg.scaled_distribution()
    print("\n  Per-class generation plan:")
    print(f"  {'class':<14s} {'will_send':>10s} {'preview_shown':>13s}")
    print("  " + "-" * 41)
    for label in cfg.TARGET_LABELS:
        n = distribution.get(label, 0)
        print(f"  {label:<14s} {n:>10d} {min(n_per_class, n):>13d}")

    for label in cfg.TARGET_LABELS:
        n = distribution.get(label, 0)
        if n == 0:
            continue
        class_seed = args.seed + cfg.TARGET_LABEL2ID[label] * 1_000_003
        points = enumerate_points(label, min(n_per_class, n), seed=class_seed)
        print()
        print("=" * 72)
        print(f"  PREVIEW - label = {label}  ({n} calls planned)")
        print("=" * 72)
        for i, pt in enumerate(points):
            sid = _sample_id(label, i)
            up = _user_prompt(pt)
            print(f"\n  -- {sid} (combination point) --")
            for k in ("intensity", "cause", "style", "persona",
                      "turn_position"):
                print(f"    {k:<14s}: {pt[k]}")
            print("    user_prompt:")
            for line in up.split("\n"):
                print(f"      | {line}")


def _build_work_items(args: argparse.Namespace) -> List[Dict]:
    """Build the list of (id, point) pairs to send to the API."""
    distribution = cfg.scaled_distribution()

    if args.label:
        distribution = {args.label: distribution[args.label]}

    items: List[Dict] = []
    for label, n in distribution.items():
        # Per-class seed so different classes don't collide on points
        class_seed = args.seed + cfg.TARGET_LABEL2ID[label] * 1_000_003
        points = enumerate_points(label, n, seed=class_seed)
        for i, pt in enumerate(points):
            items.append({
                "id":    _sample_id(label, i),
                "point": pt,
            })

    if args.limit is not None:
        items = items[:args.limit]
    return items


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    load_dotenv()
    cfg.ensure_dirs()

    if args.preview > 0:
        print("=" * 72)
        print("  SmartHandover - Synthetic Text Generator (PREVIEW MODE)")
        print("=" * 72)
        print(f"  No API calls will be made.")
        print(f"  seed: {args.seed}")
        _preview(args, args.preview)
        return

    print("=" * 72)
    print("  SmartHandover - Synthetic Text Generator")
    print("=" * 72)
    print(f"  model        : {cfg.TEXT_MODEL}")
    print(f"  workers      : {args.workers}")
    print(f"  output       : {args.out}")
    print(f"  seed         : {args.seed}")
    print(f"  resume       : {not args.no_resume}")

    items = _build_work_items(args)
    print(f"  total items  : {len(items)}")

    # --- Resume ------------------------------------------------------------
    if args.no_resume:
        # Truncate file
        open(args.out, "w").close()
        already = set()
    else:
        already = loaded_ids_from_jsonl(args.out, key="id")
    if already:
        print(f"  resuming - {len(already)} samples already present")
        items = [it for it in items if it["id"] not in already]
        print(f"  remaining    : {len(items)}")

    if not items:
        print("\n  Nothing to do.")
        return

    # --- Eagerly fail if API key missing ---------------------------------
    try:
        text_pool = get_pool("text")
    except RuntimeError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    print()
    print(text_pool.summary())
    print()

    # --- Run ---------------------------------------------------------------
    t0 = time.time()
    n_ok, n_fail = 0, 0
    per_label_ok = {label: 0 for label in cfg.TARGET_LABELS}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_generate_one, it["point"], it["id"]): it
            for it in items
        }
        with tqdm(total=len(futures), desc="generating", unit="utt") as bar:
            for fut in as_completed(futures):
                rec = fut.result()
                if rec is None:
                    n_fail += 1
                else:
                    append_jsonl(args.out, rec)
                    n_ok += 1
                    per_label_ok[rec["label"]] += 1
                bar.set_postfix(
                    ok=n_ok, fail=n_fail,
                    frust=per_label_ok["frustration"],
                    anger=per_label_ok["anger"],
                )
                bar.update(1)

    elapsed = time.time() - t0
    print()
    print(f"  ok={n_ok}  fail={n_fail}  elapsed={elapsed:.0f}s "
          f"({n_ok / max(elapsed, 1):.1f} utt/s)")
    print("\n  Per-label generated this run:")
    for label in cfg.TARGET_LABELS:
        print(f"    {label:<14s} {per_label_ok[label]:>5d}")

    print()
    print(text_pool.summary())

    if n_fail:
        print("\n  Some samples failed - re-run the script to retry "
              "(it will resume).")


if __name__ == "__main__":
    main()
