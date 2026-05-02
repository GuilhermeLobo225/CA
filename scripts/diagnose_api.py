#!/usr/bin/env python3
"""SmartHandover - API connectivity diagnostic.

Checks the configured backends in order:
  1) IAEDU pool (configs/iaedu_accounts.json) - if present
  2) OpenAI standard fallback (OPENAI_API_KEYS) - otherwise
  3) OpenAI TTS (OPENAI_TTS_API_KEY) - if set

Run from project root:

    python scripts/diagnose_api.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.synthetic._openai_client import (  # noqa: E402
    IAEDU_ACCOUNTS_FILE,
    IAEduClient,
    IAEduConnectionError,
    IAEduError,
    IAEduRateLimit,
    _load_iaedu_accounts,
    load_dotenv,
)


def mask(s: str) -> str:
    if len(s) <= 8:
        return s[:2] + "***"
    return s[:6] + "..." + s[-4:]


def step(n: int, title: str) -> None:
    print()
    print("=" * 72)
    print(f"  STEP {n}: {title}")
    print("=" * 72)


def diagnose_iaedu(accounts) -> int:
    step(2, f"Test IAEDU agent with each of the {len(accounts)} account(s)")
    n_ok = 0
    for acc in accounts:
        client = IAEduClient(acc, timeout=20.0)
        print(f"\n  {acc.name}  (key {mask(acc.api_key)} | "
              f"channel {mask(acc.channel_id)})")
        print(f"    endpoint: {acc.endpoint}")
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": "Reply with the single word: ok",
                }],
                max_tokens=5,
                temperature=0.0,
                timeout=20.0,
            )
            dt = time.time() - t0
            text = (resp.choices[0].message.content or "").strip()
            print(f"    OK in {dt*1000:.0f}ms -> {text!r}")
            n_ok += 1
        except IAEduRateLimit as e:
            print(f"    RATE LIMITED (account works, just throttled): {e}")
            n_ok += 1
        except IAEduConnectionError as e:
            print(f"    CONNECTION ERROR: {e}")
        except IAEduError as e:
            print(f"    IAEDU ERROR: {e}")
        except Exception as e:
            print(f"    {type(e).__name__}: {str(e)[:140]}")
    return n_ok


def diagnose_openai_text(keys, base_url, model) -> int:
    step(2, f"Test OpenAI fallback - {len(keys)} key(s)")
    try:
        from openai import OpenAI
        import openai as openai_pkg
    except ImportError:
        print("  [FAIL] openai SDK not installed.")
        return 0

    n_ok = 0
    for i, key in enumerate(keys, 1):
        client = OpenAI(api_key=key, base_url=base_url or None, timeout=20.0)
        try:
            t0 = time.time()
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply 'ok'"}],
                max_tokens=5,
                temperature=0.0,
            )
            dt = time.time() - t0
            text = (r.choices[0].message.content or "").strip()
            print(f"  key#{i} ({mask(key)}): OK in {dt*1000:.0f}ms -> {text!r}")
            n_ok += 1
        except openai_pkg.RateLimitError:
            print(f"  key#{i} ({mask(key)}): RATE LIMITED (key works)")
            n_ok += 1
        except openai_pkg.AuthenticationError:
            print(f"  key#{i} ({mask(key)}): AUTH FAILED - bad key")
        except openai_pkg.NotFoundError:
            print(f"  key#{i} ({mask(key)}): MODEL NOT FOUND - "
                  f"'{model}' not on this proxy")
        except Exception as e:
            print(f"  key#{i} ({mask(key)}): {type(e).__name__}: "
                  f"{str(e)[:140]}")
    return n_ok


def diagnose_judge() -> bool:
    """Test the judge pool (Ollama by default). Returns True if a smoke
    chat call succeeds."""
    from src.data.synthetic import config as cfg
    from src.data.synthetic._openai_client import (
        get_pool, with_pool_backoff,
    )

    step(3, f"Test judge pool ({cfg.JUDGE_PROVIDER} / {cfg.JUDGE_MODEL})")
    if cfg.JUDGE_PROVIDER == "ollama":
        print(f"  endpoint: {cfg.OLLAMA_BASE_URL}")
        print(f"  Hint: make sure 'ollama serve' is running and you have "
              f"pulled the model:")
        print(f"    ollama pull {cfg.JUDGE_MODEL}")
    try:
        pool = get_pool("judge")
    except RuntimeError as e:
        print(f"  [FAIL] {e}")
        return False

    def _call(client):
        return client.chat.completions.create(
            model=cfg.JUDGE_MODEL,
            messages=[{
                "role": "user",
                "content": 'Reply with the JSON object: {"ok": true}',
            }],
            max_tokens=20,
            temperature=0.0,
        )

    try:
        t0 = time.time()
        resp = with_pool_backoff(pool, _call)
        dt = time.time() - t0
        text = (resp.choices[0].message.content or "").strip()
        print(f"  OK in {dt*1000:.0f}ms -> {text!r}")
        return True
    except Exception as e:
        print(f"  {type(e).__name__}: {str(e)[:200]}")
        if cfg.JUDGE_PROVIDER == "ollama":
            print()
            print("  Most common causes for Ollama judge failures:")
            print("    1) 'ollama serve' is not running (start it in a "
                  "terminal: ollama serve)")
            print(f"    2) Model not pulled yet: ollama pull {cfg.JUDGE_MODEL}")
            print(f"    3) OLLAMA_BASE_URL is wrong "
                  f"(default http://localhost:11434/v1)")
        return False


def diagnose_openai_tts() -> bool:
    tts_key = os.environ.get("OPENAI_TTS_API_KEY", "").strip()
    if not tts_key or tts_key.startswith("sk-direct-openai-key"):
        return False  # not configured
    step(3, "Test OpenAI TTS (gpt-4o-mini-tts) with a 3-character clip")
    try:
        from openai import OpenAI
        import openai as openai_pkg
    except ImportError:
        print("  [FAIL] openai SDK not installed.")
        return False

    base_url = os.environ.get("OPENAI_TTS_BASE_URL") or None
    client = OpenAI(api_key=tts_key, base_url=base_url, timeout=30.0)
    model = os.environ.get("SYNTH_TTS_MODEL", "gpt-4o-mini-tts")
    try:
        t0 = time.time()
        r = client.audio.speech.create(
            model=model,
            voice="alloy",
            input="ok",
            instructions="Speak the word neutrally.",
            response_format="wav",
        )
        # Try to read at least the header
        audio = r.read() if hasattr(r, "read") else getattr(r, "content", b"")
        dt = time.time() - t0
        print(f"  TTS key {mask(tts_key)}: OK in {dt*1000:.0f}ms "
              f"({len(audio)} bytes)")
        return True
    except openai_pkg.AuthenticationError:
        print(f"  TTS key {mask(tts_key)}: AUTH FAILED")
    except openai_pkg.NotFoundError:
        print(f"  TTS key {mask(tts_key)}: MODEL NOT FOUND - "
              f"'{model}' not available on this account")
    except Exception as e:
        print(f"  TTS key {mask(tts_key)}: {type(e).__name__}: "
              f"{str(e)[:140]}")
    return False


def main() -> None:
    print("SmartHandover - API Diagnostic\n")

    # --- 1: env loading ---
    step(1, "Load .env + IAEDU accounts file")
    if not os.path.isfile(".env"):
        print("  [WARN] No .env file. Copy from .env.example and fill in.")
    else:
        load_dotenv()
        print("  .env loaded OK")

    accounts = _load_iaedu_accounts()
    iaedu_path = IAEDU_ACCOUNTS_FILE
    if accounts:
        print(f"  IAEDU accounts file: {iaedu_path}")
        print(f"  IAEDU accounts found (with real values): {len(accounts)}")
        for acc in accounts:
            print(f"    - {acc.name}: key={mask(acc.api_key)}, "
                  f"channel={mask(acc.channel_id)}")
    elif os.path.isfile(iaedu_path):
        print(f"  [WARN] {iaedu_path} exists but has no usable accounts.")
        print(f"         Make sure api_key and channel_id are filled in "
              "(not placeholders).")
    else:
        print(f"  No IAEDU accounts file at {iaedu_path}.")

    text_model = os.environ.get("SYNTH_TEXT_MODEL", "gpt-4o-mini")
    print(f"  text model: {text_model}")

    # --- 2: text backend ---
    if accounts:
        n_ok = diagnose_iaedu(accounts)
        backend = f"IAEDU ({len(accounts)} account(s))"
        n_total = len(accounts)
    else:
        keys_csv = os.environ.get("OPENAI_API_KEYS") or os.environ.get(
            "OPENAI_API_KEY", "")
        keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
        if not keys:
            print()
            print("[FAIL] No text backend configured. Either:")
            print(f"  - fill {iaedu_path} with IAEDU accounts, or")
            print("  - set OPENAI_API_KEYS in .env")
            sys.exit(2)
        base_url = os.environ.get("OPENAI_BASE_URL") or None
        n_ok = diagnose_openai_text(keys, base_url, text_model)
        backend = f"OpenAI fallback ({len(keys)} key(s))"
        n_total = len(keys)

    # --- 3: judge pool ---
    judge_ok = diagnose_judge()

    # --- 4: TTS (optional) ---
    tts_ok = diagnose_openai_tts()

    # --- summary ---
    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Text backend  : {backend}")
    print(f"  Text working  : {n_ok}/{n_total}")
    print(f"  Judge backend : {'OK' if judge_ok else 'FAIL'}")
    print(f"  TTS configured: {'yes (works)' if tts_ok else 'no/skipped'}")
    print()
    if n_ok > 0:
        print("  Text generation is good to go. Next step:")
        print("    python -m src.data.synthetic.generate_text --limit 20")
    else:
        print("  No working text backend - fix the errors above before "
              "running generate_text.")
        sys.exit(3)


if __name__ == "__main__":
    main()
