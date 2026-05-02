"""
SmartHandover - Gradio Demo (Day 11)

Live demonstration of the full audio -> emotion -> handover pipeline.

Features
--------
* Upload a .wav file OR record from the microphone.
* Per-model breakdown: VADER, GoEmotions, RoBERTa (fine-tuned), SpeechBrain.
* Final ensemble decision + handover trigger (using the threshold tuned in
  ``configs/handover_threshold.json``).
* Bar plot of the meta-classifier emotion probabilities.
* Latency breakdown so the audience can see the < 2 s real-time target.

Run
---
    python -m src.demo.app

The pipeline takes ~30 s to load on first launch (Whisper + RoBERTa + the
two HuggingFace models). Subsequent inference per utterance is < 2 s on
GPU, ~3-4 s on CPU.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Tuple

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.classifiers.pipeline import SmartHandoverPipeline  # noqa: E402

# Lazy-init: the heavy pipeline is built once, on first call.
_pipeline: SmartHandoverPipeline | None = None


def _get_pipeline() -> SmartHandoverPipeline:
    global _pipeline
    if _pipeline is None:
        print("[demo] Loading SmartHandover pipeline (first call) ...")
        _pipeline = SmartHandoverPipeline()
    return _pipeline


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _normalise_audio(audio_payload) -> Tuple[np.ndarray, int]:
    """Accept Gradio's audio payload and return (waveform_float32_mono, sr).

    Gradio passes either a tuple (sr, np.array) for inline microphone /
    upload, or a path string when ``type='filepath'`` is used.
    """
    if audio_payload is None:
        raise ValueError("No audio provided.")

    if isinstance(audio_payload, str):
        # filepath
        import soundfile as sf
        wav, sr = sf.read(audio_payload, dtype="float32", always_2d=False)
    else:
        sr, wav = audio_payload  # gr.Audio default returns (sr, ndarray)
        wav = np.asarray(wav)

    # Gradio sometimes returns int16 - convert to float32 [-1, 1]
    if wav.dtype != np.float32:
        if wav.dtype.kind == "i":
            wav = wav.astype(np.float32) / np.iinfo(wav.dtype).max
        else:
            wav = wav.astype(np.float32)

    # Mono-fy
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    # Resample to 16 kHz if needed
    if sr != 16000:
        try:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        except ImportError:
            # Fallback: very basic linear-interp resample
            new_len = int(round(len(wav) * 16000 / sr))
            wav = np.interp(
                np.linspace(0, len(wav) - 1, new_len, dtype=np.float32),
                np.arange(len(wav), dtype=np.float32),
                wav,
            ).astype(np.float32)
        sr = 16000

    return wav, sr


def _format_details(out: Dict[str, Any]) -> str:
    """Markdown summary of every base model's output."""
    d = out["details"]
    lines = ["### Per-Model Output", ""]

    # VADER
    v = d["vader"]
    lines.append("**VADER (lexicon)**")
    lines.append(f"- pos={v['pos']:.2f}  neg={v['neg']:.2f}  "
                 f"neu={v['neu']:.2f}  compound={v['compound']:+.3f}")
    lines.append("")

    # GoEmotions
    g = d["goemo"]
    g_sorted = sorted(g.items(), key=lambda kv: -kv[1])[:3]
    lines.append("**GoEmotions (top-3)**")
    for label, score in g_sorted:
        lines.append(f"- {label}: {score:.2%}")
    lines.append("")

    # RoBERTa (5 target classes)
    r = d["roberta"]
    lines.append("**RoBERTa - fine-tuned on MELD**")
    name_map = {
        "prob_anger": "anger", "prob_frust": "frustration",
        "prob_sadne": "sadness", "prob_neutr": "neutral",
        "prob_satis": "satisfaction",
    }
    for col, val in r.items():
        lines.append(f"- {name_map.get(col, col)}: {val:.2%}")
    lines.append("")

    # SpeechBrain (audio)
    s = d["speechbrain"]
    lines.append("**Audio (wav2vec2-IEMOCAP)**")
    full = {"ang": "angry", "hap": "happy", "sad": "sad", "neu": "neutral"}
    for k, v in s.items():
        lines.append(f"- {full.get(k, k)}: {v:.2%}")
    return "\n".join(lines)


def _format_timings(out: Dict[str, Any]) -> str:
    t = out["timings"]
    lines = ["### Latency"]
    for stage in ("asr", "vader", "goemo", "roberta", "speechbrain", "meta"):
        lines.append(f"- **{stage}**: {t[stage] * 1000:.0f} ms")
    lines.append(f"- **total**: {t['total'] * 1000:.0f} ms "
                 f"({'OK <2s' if t['total'] < 2 else 'SLOW (>2s)'})")
    return "\n".join(lines)


def _emotion_bar_plot(meta_probs: Dict[str, float]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    labels = list(meta_probs.keys())
    values = [meta_probs[l] for l in labels]

    colors = {
        "anger": "#d62728", "frustration": "#ff7f0e",
        "sadness": "#1f77b4", "neutral": "#7f7f7f",
        "satisfaction": "#2ca02c",
    }
    bar_colors = [colors.get(l, "#888") for l in labels]
    ax.barh(labels, values, color=bar_colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.set_title("Final ensemble probabilities")
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.2%}", va="center", fontsize=9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Main inference callback
# ----------------------------------------------------------------------


def analyse(audio_payload):
    pipe = _get_pipeline()
    if audio_payload is None:
        return ("No audio supplied. Please upload a .wav or record one.",
                "", "", None)

    try:
        wav, sr = _normalise_audio(audio_payload)
    except Exception as e:
        return (f"**Audio error:** {e}", "", "", None)

    out = pipe.predict_from_audio(wav, sr=sr)

    transcription = out["text"] or "(empty transcription)"
    pred = out["predicted_emotion"]
    conf = out["confidence"]
    handover = "YES - escalate to human agent" if out["should_handover"] else "no"

    summary = (
        f"### Result\n\n"
        f"- **Transcription:** {transcription}\n"
        f"- **Detected emotion:** `{pred}`  ({conf:.1%})\n"
        f"- **Handover score:** {out['handover_score']:.3f}  "
        f"(threshold = {out['threshold']:.3f})\n"
        f"- **Handover decision:** **{handover}**"
    )

    details = _format_details(out)
    timings = _format_timings(out)
    plot = _emotion_bar_plot(out["meta_probs"])
    return summary, details, timings, plot


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------


_CSS = ".gr-button {font-weight: 600;}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="SmartHandover - Demo") as ui:
        gr.Markdown(
            "# SmartHandover - Demo\n"
            "Multimodal frustration detection for customer-support calls. "
            "Upload an audio clip (or record one) and see the full pipeline "
            "from ASR through to the handover decision."
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_in = gr.Audio(
                    sources=["upload", "microphone"],
                    type="numpy",
                    label="Input audio (.wav / microphone)",
                )
                run_btn = gr.Button("Analyse", variant="primary")

                gr.Markdown(
                    "### Tip\n"
                    "For the best results, speak clearly for 2-10 seconds. "
                    "The classifier was trained on English MELD data, so "
                    "non-English clips may behave unexpectedly."
                )

            with gr.Column(scale=1):
                summary_out = gr.Markdown(label="Result")
                bar_out = gr.Plot(label="Ensemble probabilities")

        with gr.Accordion("Per-model breakdown", open=False):
            details_out = gr.Markdown()

        with gr.Accordion("Latency breakdown", open=False):
            timings_out = gr.Markdown()

        run_btn.click(
            analyse,
            inputs=[audio_in],
            outputs=[summary_out, details_out, timings_out, bar_out],
        )

    return ui


def main() -> None:
    ui = build_ui()
    print("[demo] Launching Gradio. Use Ctrl+C to stop.")
    ui.launch(server_name="127.0.0.1", show_error=True, css=_CSS)


if __name__ == "__main__":
    main()
