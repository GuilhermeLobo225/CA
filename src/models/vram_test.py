"""
Smart Handover - VRAM Safety Check
Constraint: 16 GB VRAM maximum

Loads roberta-base (text) and wav2vec2-base (audio) simultaneously on GPU,
runs a forward pass with FP16 autocast on realistic dummy inputs, and reports
peak VRAM usage. This must stay well under 16 GB.
"""

import torch
from transformers import RobertaModel, Wav2Vec2Model


def fmt_mb(bytes_val: int) -> str:
    return f"{bytes_val / 1024**2:.1f} MB"


def fmt_gb(bytes_val: int) -> str:
    return f"{bytes_val / 1024**3:.2f} GB"


def main():
    if not torch.cuda.is_available():
        print("CUDA not available. This test requires a GPU.")
        return

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    print("=" * 60)
    print("VRAM SAFETY CHECK — Smart Handover")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Total VRAM: {fmt_gb(torch.cuda.get_device_properties(device).total_memory)}")
    print("=" * 60)

    # --- Load models ---
    print("\n[1/4] Loading roberta-base...")
    text_model = RobertaModel.from_pretrained("roberta-base").to(device).eval()
    after_text = torch.cuda.memory_allocated(device)
    print(f"      VRAM after text model: {fmt_mb(after_text)}")

    print("[2/4] Loading facebook/wav2vec2-base...")
    audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()
    after_both = torch.cuda.memory_allocated(device)
    print(f"      VRAM after both models: {fmt_mb(after_both)}")

    # --- Dummy inputs ---
    batch_size = 8
    seq_len = 128
    sample_rate = 16_000
    audio_seconds = 10

    # Text: token IDs (batch_size x seq_len)
    input_ids = torch.randint(0, 50265, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)

    # Audio: raw waveform (batch_size x samples)
    audio_input = torch.randn(batch_size, sample_rate * audio_seconds, device=device)

    # --- Forward pass with FP16 autocast ---
    print(f"\n[3/4] Forward pass (FP16 autocast) — text batch ({batch_size}x{seq_len})...")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        text_out = text_model(input_ids=input_ids, attention_mask=attention_mask)
    after_text_fwd = torch.cuda.memory_allocated(device)
    print(f"      VRAM after text forward: {fmt_mb(after_text_fwd)}")

    print(f"[4/4] Forward pass (FP16 autocast) — audio batch ({batch_size}x{audio_seconds}s@{sample_rate}Hz)...")
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        audio_out = audio_model(input_values=audio_input)
    after_audio_fwd = torch.cuda.memory_allocated(device)
    print(f"      VRAM after audio forward: {fmt_mb(after_audio_fwd)}")

    # --- Summary ---
    peak = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Text  output shape : {text_out.last_hidden_state.shape}")
    print(f"  Audio output shape : {audio_out.last_hidden_state.shape}")
    print(f"  Peak allocated     : {fmt_mb(peak)}  ({fmt_gb(peak)})")
    print(f"  Peak reserved      : {fmt_mb(reserved)}  ({fmt_gb(reserved)})")
    print(f"  VRAM limit         : 16.00 GB")
    print(f"  Headroom           : {fmt_gb(16 * 1024**3 - peak)}")

    if peak < 16 * 1024**3:
        print("\n  [PASS] Peak VRAM is within the 16 GB budget.")
    else:
        print("\n  [FAIL] Peak VRAM exceeds 16 GB! Reduce batch size or sequence length.")

    # Cleanup
    del text_model, audio_model, text_out, audio_out, input_ids, attention_mask, audio_input
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
