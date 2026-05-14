#!/usr/bin/env python3
"""SmartHandover - Download the trained .pt checkpoints.

The two large model weights (RoBERTa + wav2vec2 fine-tuned) cannot be
stored directly in git (GitHub blocks files > 100 MB). They live in a
GitHub Release attached to the project repo.

This script downloads them to ``checkpoints/`` so the demo and the
end-to-end pipeline can run.

Usage
-----
    python scripts/download_checkpoints.py
    python scripts/download_checkpoints.py --only roberta
    python scripts/download_checkpoints.py --only wav2vec2
    python scripts/download_checkpoints.py --force          # re-download

If GitHub Release-based hosting is later replaced by another URL
(OneDrive, etc.), only the ``CHECKPOINTS`` dict at the top needs to
be updated.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Where the .pt files live. Update these URLs after publishing the release.
# ---------------------------------------------------------------------------

# After publishing the GitHub Release, copy each asset's "Download" URL
# (right-click -> Copy link address) and paste it below.
# The placeholder URLs are formatted to make the search-and-replace easy
# - just swap GuilhermeLobo225/CA and v1.0-models if needed.

CHECKPOINTS: Dict[str, Dict] = {
    # =========================================================
    # REQUIRED for demo (pipeline_v4 + src/demo/app.py)
    # =========================================================
    "roberta_combined.pt": {
        "url":      "https://github.com/GuilhermeLobo225/CA/releases/download/"
                    "v1.0-models/roberta_combined.pt",
        "size_mb":  487,
        "required": True,
        "purpose":  "[DEMO] RoBERTa fine-tuned on MELD + 19 280 synthetic.",
    },
    "wav2vec2_finetuned.pt": {
        "url":      "https://github.com/GuilhermeLobo225/CA/releases/download/"
                    "v1.0-models/wav2vec2_finetuned.pt",
        "size_mb":  1233,
        "required": True,
        "purpose":  "[DEMO] wav2vec2-large fine-tuned on MELD + CREMA-D + "
                    "synthetic phone-band (with-synth variant).",
    },
    # =========================================================
    # OPTIONAL - only needed to reproduce the ablation study
    # =========================================================
    "roberta_text_only.pt": {
        "url":      "https://github.com/GuilhermeLobo225/CA/releases/download/"
                    "v1.0-models/roberta_text_only.pt",
        "size_mb":  487,
        "required": False,
        "purpose":  "[ABLATION] v1 baseline RoBERTa (MELD only, no synthetic).",
    },
    "roberta_meld_only.pt": {
        "url":      "https://github.com/GuilhermeLobo225/CA/releases/download/"
                    "v1.0-models/roberta_meld_only.pt",
        "size_mb":  487,
        "required": False,
        "purpose":  "[ABLATION] RoBERTa trained on MELD only (Day F8).",
    },
    "roberta_synth_only.pt": {
        "url":      "https://github.com/GuilhermeLobo225/CA/releases/download/"
                    "v1.0-models/roberta_synth_only.pt",
        "size_mb":  487,
        "required": False,
        "purpose":  "[ABLATION] RoBERTa trained on synthetic only "
                    "(cross-corpus benchmark).",
    },
    "roberta_combined_cw.pt": {
        "url":      "https://github.com/GuilhermeLobo225/CA/releases/download/"
                    "v1.0-models/roberta_combined_cw.pt",
        "size_mb":  487,
        "required": False,
        "purpose":  "[ABLATION] RoBERTa combined with class weights "
                    "(discarded variant).",
    },
    "wav2vec2_finetuned_no_synth.pt": {
        "url":      "https://github.com/GuilhermeLobo225/CA/releases/download/"
                    "v1.0-models/wav2vec2_finetuned_no_synth.pt",
        "size_mb":  1233,
        "required": False,
        "purpose":  "[ABLATION] wav2vec2 trained on MELD + CREMA-D only "
                    "(no synthetic).",
    },
}

CHECKPOINT_DIR = "checkpoints"


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _download(url: str, out_path: str) -> bool:
    """Stream-download with a progress bar. Returns True on success."""
    import urllib.request
    import urllib.error

    tmp = out_path + ".part"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "smarthandover-downloader/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            written = 0
            chunk_size = 1024 * 1024  # 1 MB
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    written += len(chunk)
                    if total > 0:
                        pct = written * 100 / total
                        bar_len = 30
                        filled = int(bar_len * written / total)
                        bar = "#" * filled + "-" * (bar_len - filled)
                        sys.stdout.write(
                            f"\r    [{bar}] {pct:5.1f}%  "
                            f"{_human(written)} / {_human(total)}"
                        )
                        sys.stdout.flush()
                    else:
                        sys.stdout.write(f"\r    {_human(written)} downloaded")
                        sys.stdout.flush()
            sys.stdout.write("\n")
        os.replace(tmp, out_path)
        return True
    except urllib.error.HTTPError as e:
        sys.stdout.write("\n")
        print(f"  [ERROR] HTTP {e.code}: {e.reason}", file=sys.stderr)
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    except urllib.error.URLError as e:
        sys.stdout.write("\n")
        print(f"  [ERROR] {e.reason}", file=sys.stderr)
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    except KeyboardInterrupt:
        sys.stdout.write("\n  [INTERRUPTED] partial file kept at: "
                         f"{tmp}\n")
        return False
    except Exception as e:
        sys.stdout.write("\n")
        print(f"  [ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=list(CHECKPOINTS),
                   help="Download only one file (default: required ones).")
    p.add_argument("--ablation", action="store_true",
                   help="Also download the ablation .pt files "
                        "(~3 GB extra, only needed to reproduce the ablation "
                        "study).")
    p.add_argument("--all", action="store_true",
                   help="Alias for --ablation (download every checkpoint).")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if file already exists.")
    p.add_argument("--ckpt-dir", default=CHECKPOINT_DIR,
                   help=f"Destination dir (default: {CHECKPOINT_DIR}).")
    args = p.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.only:
        targets: List[str] = [args.only]
    elif args.ablation or args.all:
        targets = list(CHECKPOINTS.keys())
    else:
        # Default: only the required ones (for the demo)
        targets = [name for name, info in CHECKPOINTS.items()
                   if info.get("required")]

    print("=" * 72)
    print("  SmartHandover - Checkpoint downloader")
    print("=" * 72)
    print(f"  destination : {args.ckpt_dir}")
    print(f"  files       : {len(targets)}")
    print()

    n_ok, n_skip, n_fail = 0, 0, 0
    for name in targets:
        info = CHECKPOINTS[name]
        out_path = os.path.join(args.ckpt_dir, name)
        size_mb = info["size_mb"]
        print(f"  >>> {name}  ({size_mb} MB)")
        print(f"      {info['purpose']}")

        if os.path.exists(out_path) and not args.force:
            existing = os.path.getsize(out_path)
            # If the existing file is roughly the right size, skip
            if abs(existing - size_mb * 1024 * 1024) < 5 * 1024 * 1024:
                print(f"      [SKIP] already at {out_path} "
                      f"({_human(existing)}). Use --force to redownload.")
                n_skip += 1
                print()
                continue
            else:
                print(f"      [WARN] existing file size mismatch "
                      f"({_human(existing)} vs expected ~{size_mb} MB). "
                      "Re-downloading.")

        url = info["url"]
        if "PASTE_URL" in url or "GuilhermeLobo225" not in url:
            print(f"      [SKIP] no URL configured for this file. "
                  "Update CHECKPOINTS dict in this script.")
            n_skip += 1
            print()
            continue

        print(f"      url: {url}")
        if _download(url, out_path):
            print(f"      [OK] saved to {out_path}")
            n_ok += 1
        else:
            n_fail += 1
        print()

    print("=" * 72)
    print(f"  ok={n_ok}  skipped={n_skip}  failed={n_fail}")
    if n_fail:
        print()
        print("  Some downloads failed. Most common causes:")
        print("    1) The release hasn't been published yet - ask the repo owner.")
        print("    2) The URLs in CHECKPOINTS need to be updated.")
        print("    3) Network issue - try again with --force.")
        sys.exit(1)


if __name__ == "__main__":
    main()
