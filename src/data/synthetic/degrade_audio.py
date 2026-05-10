"""SmartHandover - Audio degradation for synthetic call-centre realism.

The TTS output (24 kHz studio quality, no background noise, idealised
prosody) sits in a different distribution from real call-centre audio.
A wav2vec2 fine-tuned directly on this clean synthetic data is at risk
of learning **TTS artefacts** instead of emotional cues - it will then
perform poorly on real-world phone calls.

This module degrades the clean audio along three axes that match what
PSTN / VoIP calls actually look like:

  1. **Codec degradation** (mu-law + bandpass 300-3400 Hz):
     mu-law is the G.711 codec used worldwide on telephone lines; the
     bandpass approximates the 4 kHz baseband.

  2. **Additive noise** (synthetic call-centre hiss):
     Pink noise (1/f) plus a faint 60 Hz hum mimic the typical
     low-frequency rumble + steady hiss heard on landlines and old
     conference systems. SNR randomly sampled from [snr_db_lo, snr_db_hi]
     so the wav2vec2 sees a range of conditions.

  3. **Gain jitter** (+/- gain_db_jitter dB):
     Random per-sample gain so the model cannot rely on absolute
     loudness as a cue.

Why synthetic noise instead of MUSAN
------------------------------------
The original plan called for the MUSAN ``noise`` subset (~6 GB, OpenSLR).
For a coursework project, a 6 GB download is overkill and the academic
defence is the same: synthetic pink+hum noise at known SNR is a
**well-defined** degradation; MUSAN brings realism but also unmodelled
content (shouts, music) that is harder to control. We therefore use
synthetic noise by default; ``add_recorded_noise()`` is exposed so a
MUSAN drop-in can be plugged later if needed.

Public API
----------
* ``degrade_to_phone(wav, sr_in)``   - codec stage
* ``add_synthetic_noise(wav, snr_db, seed)`` - additive noise stage
* ``add_recorded_noise(wav, noise_dir, snr_db, seed)`` - MUSAN-style
* ``apply_gain_jitter(wav, db, seed)``
* ``degrade_pipeline(wav, sr_in, ...)`` - the standard composition

All functions return float32 numpy arrays in the [-1, 1] range.
"""

from __future__ import annotations

import os
import random
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Codec stage (mu-law + bandpass)
# ---------------------------------------------------------------------------


def _mu_law_encode_decode(wav: np.ndarray, mu: int = 255) -> np.ndarray:
    """Round-trip through G.711 mu-law (8-bit). Drops dynamic range as a
    real codec would."""
    wav = np.clip(wav, -1.0, 1.0)
    mu_f = float(mu)
    encoded = np.sign(wav) * np.log1p(mu_f * np.abs(wav)) / np.log1p(mu_f)
    quantised = np.round((encoded + 1) * mu_f / 2).astype(np.int16)
    quantised = np.clip(quantised, 0, int(mu_f))
    decoded_norm = 2.0 * quantised.astype(np.float32) / mu_f - 1.0
    return (np.sign(decoded_norm) *
            (np.power(1 + mu_f, np.abs(decoded_norm)) - 1) / mu_f)


def _butter_bandpass(low: float, high: float, sr: int, order: int = 4):
    """Cache-free SOS bandpass filter."""
    from scipy.signal import butter
    nyq = 0.5 * sr
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def degrade_to_phone(wav: np.ndarray, sr_in: int = 16000,
                     telephone_band_hz: Tuple[int, int] = (300, 3400)
                     ) -> Tuple[np.ndarray, int]:
    """Codec stage: 16 kHz studio -> 8 kHz telephone-band -> 16 kHz.

    Returns (degraded_wav_16k, 16000).
    """
    from scipy.signal import sosfilt, resample_poly

    wav = wav.astype(np.float32, copy=False)

    # Resample 16k -> 8k (anti-alias built into resample_poly)
    if sr_in != 8000:
        if sr_in == 16000:
            wav_8k = resample_poly(wav, 1, 2)
        else:
            # Generic
            from math import gcd
            g = gcd(sr_in, 8000)
            wav_8k = resample_poly(wav, 8000 // g, sr_in // g)
    else:
        wav_8k = wav

    # mu-law round-trip on 8 kHz signal
    wav_8k = _mu_law_encode_decode(wav_8k)

    # Telephone bandpass on 8 kHz
    lo, hi = telephone_band_hz
    if hi >= 4000:
        hi = 3800   # cap below Nyquist of 4 kHz
    sos = _butter_bandpass(lo, hi, 8000)
    wav_8k = sosfilt(sos, wav_8k).astype(np.float32)

    # Resample back to 16 kHz so wav2vec2 can consume it directly.
    wav_16k = resample_poly(wav_8k, 2, 1).astype(np.float32)
    return wav_16k, 16000


# ---------------------------------------------------------------------------
# Synthetic noise stage
# ---------------------------------------------------------------------------


def _pink_noise(n: int, rng: random.Random) -> np.ndarray:
    """Pink noise (1/f spectrum) via FFT shaping.

    Generates Gaussian white noise, transforms it to the frequency
    domain, scales each bin by 1/sqrt(f) to obtain a 1/f power
    spectrum, and transforms back. Deterministic given the rng.
    """
    np_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
    white = np_rng.standard_normal(n).astype(np.float32)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Avoid division by zero at the DC bin
    freqs[0] = 1.0
    spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=n).astype(np.float32)
    return pink / (np.std(pink) + 1e-9)


def _hum_60hz(n: int, sr: int) -> np.ndarray:
    """Faint 60 Hz mains hum + 120 Hz harmonic."""
    t = np.arange(n, dtype=np.float32) / sr
    hum = (np.sin(2 * np.pi * 60.0 * t) +
           0.4 * np.sin(2 * np.pi * 120.0 * t))
    return hum / (np.max(np.abs(hum)) + 1e-9)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2) + 1e-12))


def add_synthetic_noise(wav: np.ndarray, sr: int = 16000,
                        snr_db: float = 20.0,
                        hum_strength: float = 0.15,
                        seed: int = 0) -> np.ndarray:
    """Mix pink noise + faint 60 Hz hum at the requested SNR.

    SNR computed against the speech RMS. ``hum_strength`` is the
    *relative* amplitude of the hum vs the pink noise (0 = pure pink).
    """
    rng = random.Random(seed)
    n = len(wav)
    pink = _pink_noise(n, rng)
    hum = _hum_60hz(n, sr) * hum_strength

    noise = pink + hum
    noise = noise / (_rms(noise) + 1e-9)

    speech_rms = _rms(wav)
    target_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    noise = noise * target_noise_rms

    mixed = wav.astype(np.float32) + noise
    # Clip, but leave a sliver of headroom to avoid hard clipping artefacts
    return np.clip(mixed, -0.99, 0.99).astype(np.float32)


# ---------------------------------------------------------------------------
# (Optional) MUSAN-style recorded noise mix - drop-in if user wants realism
# ---------------------------------------------------------------------------


def add_recorded_noise(wav: np.ndarray, noise_dir: str, sr: int = 16000,
                       snr_db: float = 20.0, seed: int = 0) -> np.ndarray:
    """Pick a random .wav from ``noise_dir`` and mix at the given SNR.

    The noise file is looped or cropped to match the speech length.
    ``noise_dir`` is expected to contain MUSAN-style background noise
    files (e.g. ``musan/noise/free-sound/*.wav``).

    If ``noise_dir`` is missing or empty, falls back silently to
    ``add_synthetic_noise``.
    """
    if not os.path.isdir(noise_dir):
        return add_synthetic_noise(wav, sr=sr, snr_db=snr_db, seed=seed)
    files = [os.path.join(noise_dir, f)
             for f in os.listdir(noise_dir)
             if f.lower().endswith(".wav")]
    if not files:
        return add_synthetic_noise(wav, sr=sr, snr_db=snr_db, seed=seed)

    import soundfile as sf
    rng = random.Random(seed)
    src_path = rng.choice(files)
    noise, src_sr = sf.read(src_path, dtype="float32")
    if noise.ndim == 2:
        noise = noise.mean(axis=1)
    if src_sr != sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(src_sr, sr)
        noise = resample_poly(noise, sr // g, src_sr // g)

    n = len(wav)
    if len(noise) >= n:
        start = rng.randint(0, len(noise) - n)
        noise = noise[start:start + n]
    else:
        # Loop with random offset to avoid repeating the same start
        reps = (n // len(noise)) + 1
        noise = np.tile(noise, reps)[:n]

    speech_rms = _rms(wav)
    n_rms = _rms(noise)
    if n_rms < 1e-9:
        return wav
    target_noise_rms = speech_rms / (10.0 ** (snr_db / 20.0))
    noise = noise * (target_noise_rms / n_rms)

    mixed = wav.astype(np.float32) + noise.astype(np.float32)
    return np.clip(mixed, -0.99, 0.99).astype(np.float32)


# ---------------------------------------------------------------------------
# Gain jitter
# ---------------------------------------------------------------------------


def apply_gain_jitter(wav: np.ndarray, db: float = 6.0,
                      seed: int = 0) -> np.ndarray:
    """Apply +/- ``db`` random gain. Useful so the model doesn't memorise
    the absolute loudness of TTS output."""
    rng = random.Random(seed)
    delta_db = rng.uniform(-db, db)
    factor = 10.0 ** (delta_db / 20.0)
    return np.clip(wav * factor, -0.99, 0.99).astype(np.float32)


# ---------------------------------------------------------------------------
# Standard pipeline
# ---------------------------------------------------------------------------


def degrade_pipeline(wav: np.ndarray, sr_in: int = 16000,
                     snr_db_range: Tuple[float, float] = (15.0, 25.0),
                     gain_db_jitter: float = 6.0,
                     noise_dir: Optional[str] = None,
                     seed: Optional[int] = None
                     ) -> Tuple[np.ndarray, int, dict]:
    """Run the full degradation pipeline on a single utterance.

    Returns (degraded_wav, sr_out=16000, meta_dict).
    The meta dict carries the (random) parameters chosen for this clip
    so the manifest can record them.
    """
    rng = random.Random(seed if seed is not None else random.randint(0, 2**31 - 1))

    # 1) codec
    wav_phone, sr = degrade_to_phone(wav, sr_in=sr_in)

    # 2) noise
    snr = rng.uniform(*snr_db_range)
    if noise_dir:
        wav_phone = add_recorded_noise(wav_phone, noise_dir=noise_dir,
                                        sr=sr, snr_db=snr,
                                        seed=rng.randint(0, 2**31 - 1))
    else:
        wav_phone = add_synthetic_noise(wav_phone, sr=sr, snr_db=snr,
                                         seed=rng.randint(0, 2**31 - 1))

    # 3) gain jitter
    wav_phone = apply_gain_jitter(wav_phone, db=gain_db_jitter,
                                  seed=rng.randint(0, 2**31 - 1))

    return wav_phone, sr, {
        "snr_db":       round(snr, 2),
        "gain_db":      "varied",
        "noise_source": "recorded" if noise_dir else "synthetic",
        "codec":        "mu_law_8k",
        "bandpass_hz":  "300-3400",
    }
