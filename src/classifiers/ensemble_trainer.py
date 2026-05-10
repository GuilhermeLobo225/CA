"""
SmartHandover — Meta-Classifier Trainer (Day 6)

Builds 19-dimensional feature vectors per utterance by concatenating the
probability / score outputs of the 4 component models:

    RoBERTa       : 5 floats  (anger, frustration, sadness, neutral, satisfaction)
    GoEmotions    : 6 floats  (anger, disgust, fear, joy, neutral, sadness)
    VADER         : 4 floats  (pos, neg, neu, compound)
    SpeechBrain   : 4 floats  (ang, hap, sad, neu)
                    ------
                    19 total

Produces the feature CSVs

    data/processed/ensemble_features_train.csv
    data/processed/ensemble_features_val.csv
    data/processed/ensemble_features_test.csv

Then trains/evaluates three meta-classifier candidates:

    * LogisticRegression(class_weight='balanced')
    * XGBClassifier
    * MLPClassifier

Selects the one with the highest weighted F1 on the validation split and
pickles it to ``checkpoints/meta_classifier.pkl``.

Run directly:

    python -m src.classifiers.ensemble_trainer
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

# scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Project imports -----------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.evaluation.metrics import compute_metrics, print_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_LABELS = ["anger", "frustration", "sadness", "neutral", "satisfaction"]
TARGET_LABEL2ID = {label: i for i, label in enumerate(TARGET_LABELS)}

# Ordered feature columns for the 19-dim vector
ROBERTA_COLS = [
    "prob_anger", "prob_frust", "prob_sadne", "prob_neutr", "prob_satis",
]  # 5
GOEMO_COLS = [
    "goemo_anger", "goemo_disgust", "goemo_fear",
    "goemo_joy", "goemo_neutral", "goemo_sadness",
]  # 6
VADER_COLS = [
    "vader_pos", "vader_neg", "vader_neu", "vader_compound",
]  # 4
SB_COLS = [
    "sb_ang", "sb_hap", "sb_sad", "sb_neu",
]  # 4

FEATURE_COLUMNS = ROBERTA_COLS + GOEMO_COLS + VADER_COLS + SB_COLS
assert len(FEATURE_COLUMNS) == 19, "Feature vector must be 19-dim"

DATA_DIR = os.path.join("data", "processed")
CKPT_DIR = "checkpoints"

# ---------------------------------------------------------------------------
# Helpers: align per-split predictions across the four CSVs
# ---------------------------------------------------------------------------


def _split_from_audio_id(audio_id: str) -> str:
    """Map the SpeechBrain audio_id prefix to a canonical split name.

    audio_id examples: 'train_00000', 'validation_00012', 'test_00123'.
    """
    prefix = audio_id.split("_", 1)[0].lower()
    if prefix.startswith("train"):
        return "train"
    if prefix.startswith("val"):
        return "val"
    if prefix.startswith("test"):
        return "test"
    raise ValueError(f"Unexpected audio_id prefix: {audio_id}")


def _load_aligned_raw_csvs() -> pd.DataFrame:
    """Load vader / goemo / speechbrain CSVs and join them row-wise.

    All three files were produced by iterating the MELD splits in the same
    order (train → val → test) so they can be aligned by row index.

    Returns a DataFrame with:
        split, audio_id, text, true_label, <all vader/goemo/sb score cols>.
    """
    vader = pd.read_csv(os.path.join(DATA_DIR, "vader_predictions.csv"))
    goemo = pd.read_csv(os.path.join(DATA_DIR, "goemo_predictions.csv"))
    sb = pd.read_csv(os.path.join(DATA_DIR, "speechbrain_predictions.csv"))

    n = min(len(vader), len(goemo), len(sb))
    if not (len(vader) == len(goemo) == len(sb)):
        print(f"  [WARN] Length mismatch: vader={len(vader)}, "
              f"goemo={len(goemo)}, sb={len(sb)} — truncating to {n}")

    vader = vader.iloc[:n].reset_index(drop=True)
    goemo = goemo.iloc[:n].reset_index(drop=True)
    sb = sb.iloc[:n].reset_index(drop=True)

    # Sanity check text consistency (first few rows)
    for i in (0, n // 2, n - 1):
        if vader.at[i, "text"] != sb.at[i, "text"]:
            print(f"  [WARN] Row {i} text mismatch between vader and "
                  f"speechbrain — alignment may be off.")
            break

    df = pd.DataFrame({
        "audio_id": sb["audio_id"].astype(str),
        "text": sb["text"].astype(str),
        "true_label": sb["true_label"].astype(str),
    })
    df["split"] = df["audio_id"].map(_split_from_audio_id)

    for c in VADER_COLS:
        df[c] = vader[c].astype(float)
    for c in GOEMO_COLS:
        df[c] = goemo[c].astype(float)
    for c in SB_COLS:
        df[c] = sb[c].astype(float)

    return df


def _roberta_csv_path(split: str, suffix: str = "") -> str:
    """Return the canonical path of a RoBERTa-predictions CSV.

    ``suffix`` is inserted before the extension (e.g. ``"_v2"`` ->
    ``roberta_train_predictions_v2.csv``).
    """
    base_map = {
        "train": "roberta_train_predictions",
        "val":   "roberta_val_predictions",
        "test":  "roberta_predictions",
    }
    return os.path.join(DATA_DIR, f"{base_map[split]}{suffix}.csv")


def _load_roberta_predictions(
    split: str,
    suffix: str = "",
    roberta_checkpoint: str = None,
    force_regenerate: bool = False,
) -> pd.DataFrame:
    """Load the RoBERTa prediction CSV for a given split.

    Falls back to *generating* the predictions with the fine-tuned model if
    the CSV does not exist or ``force_regenerate=True``.
    """
    csv_path = _roberta_csv_path(split, suffix=suffix)

    if force_regenerate or not os.path.exists(csv_path):
        ckpt = roberta_checkpoint or os.path.join(CKPT_DIR, "roberta_text_only.pt")
        if not os.path.exists(csv_path):
            print(f"  [INFO] RoBERTa {split} predictions not found at {csv_path}")
        else:
            print(f"  [INFO] --force-regenerate -> overwriting {csv_path}")
        print(f"  [INFO] Generating with {ckpt} ...")
        _generate_roberta_predictions_for_split(split, csv_path,
                                                checkpoint_path=ckpt)

    return pd.read_csv(csv_path)


def _generate_roberta_predictions_for_split(
    split: str,
    out_path: str,
    checkpoint_path: str = None,
) -> None:
    """Run a fine-tuned RoBERTa checkpoint on a MELD split and dump CSV."""
    import torch
    from src.training.train_text import TextOnlyClassifier, evaluate_on_test

    ckpt_path = checkpoint_path or os.path.join(CKPT_DIR, "roberta_text_only.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Cannot generate {split} RoBERTa predictions: missing "
            f"checkpoint at {ckpt_path}."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TextOnlyClassifier()
    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=True))

    # evaluate_on_test uses the HF split name "validation" for val.
    hf_split = {"train": "train", "val": "validation", "test": "test"}[split]
    df = evaluate_on_test(model, device=device, split=hf_split)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  [INFO] Saved {len(df)} RoBERTa predictions -> {out_path}")


# ---------------------------------------------------------------------------
# Feature assembly
# ---------------------------------------------------------------------------


def build_ensemble_features(
    roberta_suffix: str = "",
    roberta_checkpoint: str = None,
    force_regenerate: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Build the 19-dim feature DataFrames for train / val / test.

    Args:
        roberta_suffix: inserted before the .csv extension when caching
            the RoBERTa predictions (e.g. ``"_v2"``).
        roberta_checkpoint: explicit path to the .pt; falls back to the
            default ``checkpoints/roberta_text_only.pt``.
        force_regenerate: if True, ignore any cached RoBERTa CSV and
            re-run the model on all three splits.

    Returns:
        dict mapping split name -> DataFrame with columns:
            audio_id, text, true_label, true_label_id, <19 feature cols>
    """
    raw = _load_aligned_raw_csvs()

    out: Dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        raw_split = raw[raw["split"] == split].reset_index(drop=True)

        rob = _load_roberta_predictions(
            split,
            suffix=roberta_suffix,
            roberta_checkpoint=roberta_checkpoint,
            force_regenerate=force_regenerate,
        ).reset_index(drop=True)
        if len(rob) != len(raw_split):
            raise RuntimeError(
                f"Row count mismatch on split={split}: "
                f"roberta={len(rob)} vs aligned-rest={len(raw_split)}"
            )

        df = raw_split.copy()
        for c in ROBERTA_COLS:
            if c not in rob.columns:
                raise KeyError(f"Missing column {c} in RoBERTa {split} CSV")
            df[c] = rob[c].astype(float).values

        df["true_label_id"] = df["true_label"].map(TARGET_LABEL2ID).astype(int)

        # Re-order columns: meta first, then the 19 features in canonical order.
        meta_cols = ["audio_id", "text", "true_label", "true_label_id"]
        df = df[meta_cols + FEATURE_COLUMNS]

        out[split] = df

    return out


def save_feature_csvs(feature_dfs: Dict[str, pd.DataFrame],
                       suffix: str = "") -> None:
    """Persist the 3 ensemble_features_*.csv files under data/processed.

    ``suffix`` (e.g. ``"_v2"``) is appended before the extension so
    multiple versions can coexist:
        ensemble_features_train.csv     <- baseline
        ensemble_features_train_v2.csv  <- with new RoBERTa
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    for split, df in feature_dfs.items():
        path = os.path.join(DATA_DIR,
                            f"ensemble_features_{split}{suffix}.csv")
        df.to_csv(path, index=False)
        print(f"  Saved {path}  (shape={df.shape})")


# ---------------------------------------------------------------------------
# Meta-classifier training / selection
# ---------------------------------------------------------------------------


def _as_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df["true_label_id"].to_numpy(dtype=np.int64)
    return X, y


def get_candidate_classifiers() -> List[Tuple[str, object]]:
    """Return the list of meta-classifier candidates (name, estimator)."""
    candidates: List[Tuple[str, object]] = [
        (
            "LogisticRegression",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                n_jobs=-1,
            ),
        ),
        (
            "MLPClassifier",
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                early_stopping=True,
                random_state=42,
            ),
        ),
    ]

    # XGBoost is optional — include only if installed.
    try:
        from xgboost import XGBClassifier
        candidates.insert(
            1,
            (
                "XGBClassifier",
                XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    objective="multi:softprob",
                    num_class=len(TARGET_LABELS),
                    eval_metric="mlogloss",
                    n_jobs=-1,
                    random_state=42,
                    verbosity=0,
                ),
            ),
        )
    except ImportError:
        print("  [INFO] xgboost not installed — skipping XGBClassifier")

    return candidates


def train_and_select(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
) -> Tuple[str, object, Dict[str, Dict[str, float]]]:
    """Train each candidate and return the one with highest val weighted F1."""
    summary: Dict[str, Dict[str, float]] = {}
    best_name, best_model, best_val_f1 = None, None, -1.0

    for name, estimator in get_candidate_classifiers():
        print(f"\n  >>> Training {name} ...")
        estimator.fit(X_train, y_train)

        val_preds = estimator.predict(X_val)
        test_preds = estimator.predict(X_test)

        val_metrics = compute_metrics(y_val, val_preds, target_names=TARGET_LABELS)
        test_metrics = compute_metrics(y_test, test_preds, target_names=TARGET_LABELS)

        summary[name] = {
            "val_weighted_f1": val_metrics["weighted_f1"],
            "val_macro_f1":    val_metrics["macro_f1"],
            "val_frust_recall": val_metrics["frustration_recall"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_macro_f1":    test_metrics["macro_f1"],
            "test_frust_recall": test_metrics["frustration_recall"],
        }
        print(f"    val  W-F1={val_metrics['weighted_f1']:.4f} | "
              f"M-F1={val_metrics['macro_f1']:.4f} | "
              f"Frust-R={val_metrics['frustration_recall']:.4f}")
        print(f"    test W-F1={test_metrics['weighted_f1']:.4f} | "
              f"M-F1={test_metrics['macro_f1']:.4f} | "
              f"Frust-R={test_metrics['frustration_recall']:.4f}")

        if val_metrics["weighted_f1"] > best_val_f1:
            best_val_f1 = val_metrics["weighted_f1"]
            best_name = name
            best_model = estimator

    return best_name, best_model, summary


def save_best_classifier(name: str, model: object,
                         ckpt_path: str = None) -> str:
    if ckpt_path is None:
        ckpt_path = os.path.join(CKPT_DIR, "meta_classifier.pkl")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    joblib.dump(
        {"name": name, "model": model, "feature_columns": FEATURE_COLUMNS,
         "target_labels": TARGET_LABELS},
        ckpt_path,
    )
    print(f"  Saved best meta-classifier ({name}) -> {ckpt_path}")
    return ckpt_path


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Build 19-dim ensemble features and train the meta-classifier."
    )
    p.add_argument("--roberta-checkpoint", default=None,
                   help="Path to RoBERTa .pt. Default: "
                        "checkpoints/roberta_text_only.pt")
    p.add_argument("--output-suffix", default="",
                   help="Suffix appended to all outputs "
                        "(e.g. '_v2'). Use this to keep the original "
                        "meta_classifier.pkl + features intact.")
    p.add_argument("--force-regenerate", action="store_true",
                   help="Re-run RoBERTa over the MELD splits even if a "
                        "cached predictions CSV already exists.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    suffix = args.output_suffix

    print("=" * 60)
    print("  SmartHandover — Meta-Classifier Training")
    print("=" * 60)
    print(f"  RoBERTa checkpoint : "
          f"{args.roberta_checkpoint or '(default) roberta_text_only.pt'}")
    print(f"  Output suffix      : {suffix or '(none)'}")
    print(f"  Force regenerate   : {args.force_regenerate}")

    print("\n[Step 1/3] Building 19-dim feature vectors ...")
    feature_dfs = build_ensemble_features(
        roberta_suffix=suffix,
        roberta_checkpoint=args.roberta_checkpoint,
        force_regenerate=args.force_regenerate,
    )
    save_feature_csvs(feature_dfs, suffix=suffix)

    X_train, y_train = _as_xy(feature_dfs["train"])
    X_val,   y_val   = _as_xy(feature_dfs["val"])
    X_test,  y_test  = _as_xy(feature_dfs["test"])

    print(f"\n  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape},   y={y_val.shape}")
    print(f"  Test:  X={X_test.shape},  y={y_test.shape}")

    print("\n[Step 2/3] Training meta-classifier candidates ...")
    best_name, best_model, summary = train_and_select(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    print("\n" + "=" * 60)
    print("  Meta-Classifier Comparison (sorted by val W-F1)")
    print("=" * 60)
    print(f"  {'Model':<22s} {'valWF1':>8s} {'valMF1':>8s} "
          f"{'valFR':>8s} {'tstWF1':>8s} {'tstMF1':>8s} {'tstFR':>8s}")
    print("  " + "-" * 66)
    for nm, vals in sorted(summary.items(),
                           key=lambda kv: -kv[1]["val_weighted_f1"]):
        marker = "  <-- BEST" if nm == best_name else ""
        print(
            f"  {nm:<22s} "
            f"{vals['val_weighted_f1']:>8.4f} "
            f"{vals['val_macro_f1']:>8.4f} "
            f"{vals['val_frust_recall']:>8.4f} "
            f"{vals['test_weighted_f1']:>8.4f} "
            f"{vals['test_macro_f1']:>8.4f} "
            f"{vals['test_frust_recall']:>8.4f}{marker}"
        )

    print("\n[Step 3/3] Saving best model ...")
    ckpt_path = os.path.join(CKPT_DIR, f"meta_classifier{suffix}.pkl")
    save_best_classifier(best_name, best_model, ckpt_path=ckpt_path)

    # Save a small JSON manifest for convenience.
    manifest = {
        "best_meta_classifier": best_name,
        "checkpoint": ckpt_path,
        "roberta_checkpoint": (args.roberta_checkpoint
                               or os.path.join(CKPT_DIR,
                                               "roberta_text_only.pt")),
        "output_suffix": suffix,
        "feature_columns": FEATURE_COLUMNS,
        "target_labels": TARGET_LABELS,
        "summary": summary,
    }
    manifest_path = os.path.join(DATA_DIR,
                                  f"meta_classifier{suffix}_summary.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest saved -> {manifest_path}")

    # Final metrics on test for the chosen model
    print("\n" + "=" * 60)
    print(f"  Final Test-Set Metrics — {best_name}")
    print("=" * 60)
    test_preds = best_model.predict(X_test)
    test_metrics = compute_metrics(y_test, test_preds, target_names=TARGET_LABELS)
    print_metrics(test_metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
