# SmartHandover - Multimodal Emotion Detection for Customer-Support Calls

> Master's project for the **Computação Afetiva** unit, MIA, Universidade do Minho (2025/2026).

SmartHandover analyses a customer-support call in (near) real time and decides
whether the conversation should be **handed over to a human agent**. The decision
is driven by an ensemble of text and audio emotion classifiers running over the
transcribed utterance.

## Team

| Number  | Name                              |
|---------|-----------------------------------|
| PG60225 | Guilherme Lobo Pinto              |
| PG60289 | Pedro Alexandre Silva Gomes       |
| PG60393 | Simão Novais Vieira da Silva      |

## Architecture

```
                 audio (16 kHz mono)
                          |
            +-------------+--------------+
            |                            |
            v                            v
      Whisper (ASR)            wav2vec2 - IEMOCAP
        text out               P(ang/hap/sad/neu)
            |                            |
   +--------+--------+                   |
   |        |        |                   |
   v        v        v                   |
 VADER    GoEmo   RoBERTa                |
  (4)    (6 cls) (5 cls,                 |
                  fine-tuned)            |
   |        |        |                   |
   +--------+--------+---------+---------+
                              |
                       19-dim feature
                              |
                  +-----------+-----------+
                  |                       |
                  v                       v
          Score-fusion MLP         Late fusion (avg)
          (Day 6 default)          Decision fusion
                  |
                  v
          5-class emotion + handover trigger
          (P(anger)+P(frustration) > t)
```

| Component                                       | Role                       | Trained by us? |
|--------------------------------------------------|----------------------------|----------------|
| `openai/whisper-small`                          | ASR (audio -> text)        | No (pretrained)|
| `vaderSentiment`                                | Lexicon sentiment          | No             |
| `j-hartmann/emotion-english-distilroberta-base` | GoEmotions zero-shot text  | No             |
| `roberta-base` (fine-tuned)                     | 5-class text classifier    | **Yes** (MELD) |
| `superb/wav2vec2-large-superb-er`               | IEMOCAP audio classifier   | No             |
| Meta-classifier (MLP)                           | Score fusion of 19 features| **Yes** (MELD) |

The `superb/wav2vec2-large-superb-er` checkpoint replaces the original
SpeechBrain release because of Windows symlink and `k2` incompatibilities.

## Emotion model

We use a **discrete categorical** model (Ekman-style) reduced to the five
classes that matter for handover:

  `anger`, `frustration`, `sadness`, `neutral`, `satisfaction`.

The MELD source labels are remapped as documented in
[src/data/load_meld.py](src/data/load_meld.py); `surprise` is dropped (ambiguous
valence) and `fear` is folded into `frustration` as a proxy. The handover trigger
operates on the **dimensional valence proxy** `P(anger)+P(frustration)` against
a tuned threshold from `configs/handover_threshold.json`.

## Data

| Use                        | Source                                                     | Size                   |
|----------------------------|------------------------------------------------------------|------------------------|
| Training + evaluation      | `ajyy/MELD_audio` (HuggingFace) - text + 16 kHz audio      | ~12 070 utterances     |
| Pretrained text classifier | GoEmotions (Reddit) via DistilRoBERTa                      | (model only, no data)  |
| Pretrained audio classifier| IEMOCAP via `superb/wav2vec2-large-superb-er`              | (model only, no data)  |

Class distribution after the 5-class remap:

| Split      | n     | anger | frustration | sadness | neutral | satisfaction |
|------------|------:|------:|------------:|--------:|--------:|-------------:|
| train      | 8 783 | 1 380 |         268 |     683 |   4 709 |        1 743 |
| validation |   958 |   175 |          40 |     111 |     469 |          163 |
| test       | 2 329 |   413 |          50 |     208 |   1 256 |          402 |

> See [docs/plano_3_semanas.md](docs/plano_3_semanas.md) for the day-by-day
> implementation plan, and the **Limitations** section below for the strong
> caveats around using MELD (Friends sitcom dialogue) as a proxy for real
> contact-centre calls.

## Repository layout

```
configs/             - YAML config + tuned handover threshold
checkpoints/         - roberta_text_only.pt, meta_classifier{.,_balanced}.pkl
data/
  raw/               - (kept empty: dataset streamed from HF)
  processed/         - per-model predictions, ensemble features, plots
docs/                - phase reports + the 3-week plan
notebooks/           - day1..day5 notebooks (Week 1 deliverables)
src/
  data/load_meld.py
  models/text_encoder.py            - RoBERTa wrapper
  classifiers/
    vader_classifier.py
    goemo_classifier.py
    speechbrain_classifier.py       - HF wav2vec2 wrapper
    whisper_asr.py
    ensemble.py                     - Week-1 weighted-average baseline
    ensemble_trainer.py             - Day 6 score-fusion meta-classifier
    fusion_strategies.py            - score / late / decision comparison
    pipeline.py                     - end-to-end audio -> handover
  training/
    train_text.py                   - RoBERTa fine-tuning
    train_meta_balanced.py          - SMOTE + isotonic calibration variant
  evaluation/
    metrics.py
    ablation.py                     - leave-one-out study
    error_analysis.py               - top-20 errors + threshold sweep
  decision/
    handover.py                     - sliding-window handover rules
    simulate_handover.py            - conversation-level evaluation
  demo/app.py                       - Gradio live demo
run_dayN_*.py                       - thin entry points used during the sprint
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

The first run downloads ~3.6 GB of Hugging Face checkpoints (Whisper, RoBERTa,
DistilRoBERTa, wav2vec2) plus the ~2 GB MELD dataset. Subsequent runs are
cache-hits.

## What is and isn't in this repo

The git repo contains the **code, configs, docs, results CSVs and small
meta-classifier weights (.pkl)**. Two categories of large files are
*outside* the repo and have to be obtained separately:

| Item | Size | What to do |
|---|---|---|
| `.env`, `configs/iaedu_accounts.json` | small | **Never committed** (API keys). Each developer creates their own from `.example` files. Only needed to *re-generate* synthetic data — already done. |
| `checkpoints/*.pt` (RoBERTa ~500 MB, wav2vec2 ~1.2 GB) | ~5 GB total | Share via OneDrive/Drive among teammates. Only needed for the live **demo** (`src/demo/app.py`). |
| `data/raw/` (MELD streamed, CREMA-D download) | ~6 GB | MELD: auto-downloaded from HuggingFace on first call. CREMA-D: `git clone --depth 1 https://github.com/CheyneyComputerScience/CREMA-D.git data/raw/CREMA-D`. |
| `data/cache/` (MELD audio materialised on disk) | 1.3 GB | Auto-generated by `train_audio.load_meld_audio_records()`. Skip unless re-training wav2vec2. |
| `data/synthetic/audio*/*.wav` | ~5-10 GB | Already paid (€28). Share via OneDrive only if needed (not required for handover simulation or report). |

## What each teammate needs

**Reading the report material** (no code needed):
- Just clone the repo and read `docs/projeto_estado_atual.md` (briefing).
- All result tables are in there + reproducible from CSVs in `data/processed/`.

**Running the conversation-level handover simulation** (already-computed CSVs):
- Clone repo → `pip install -r requirements.txt` → run:
  ```bash
  python scripts/run_dayF17_simulate_handover_v4.py
  ```
- Outputs are reproducible from `ensemble_features_test_v4.csv` +
  `meta_classifier_v4_calibrated.pkl` (both in the repo).
- **No API keys, no large .pt files needed.**

**Running the live Gradio demo** (need .pt checkpoints):
- Get `roberta_combined.pt` and `wav2vec2_finetuned.pt` from teammate's
  shared OneDrive/Drive folder, place in `checkpoints/`.
- Run: `python -m src.demo.app`

## Reproducing the results

The pipeline was built day-by-day; each day's deliverable is reproducible in
isolation, but they must be run in order because each step writes CSV
predictions consumed by the next.

```bash
# Week 1 - per-model baselines
python run_day1_vader.py            # -> data/processed/vader_predictions.csv
python run_day2_goemo.py            # -> goemo_predictions.csv
python run_day3_audio.py            # -> speechbrain_predictions.csv (+ Whisper smoke test)
python run_day4_train.py            # -> checkpoints/roberta_text_only.pt
python run_day5_ensemble.py         # -> Week-1 weighted-average ensemble report

# Week 2 - meta-classifier, ablation, fusion comparison, handover
python -m src.classifiers.ensemble_trainer
python -m src.evaluation.ablation
python -m src.classifiers.fusion_strategies
python -m src.evaluation.error_analysis
python -m src.decision.simulate_handover

# Optional: SMOTE-balanced + calibrated meta-classifier
python -m src.training.train_meta_balanced

# Phase 4 - synthetic data + multi-corpus + ensemble v4 (final state)
python -m src.data.synthetic.generate_text                 # 21 217 utterances via IAEDU
python -m src.data.synthetic.filter_text                   # heuristics + Ollama judge -> 19 280 kept
python -m src.data.synthetic.generate_audio --max-per-class 2000   # 9 235 wavs via OpenAI TTS
python scripts/run_dayF11_degrade_audio.py                 # phone-band degradation
python scripts/run_dayF8_retrain_roberta.py                # RoBERTa multi-corpus
python -m src.data.load_cremad                             # validate CREMA-D index
python scripts/run_dayF12_cremad_baseline.py               # frozen wav2vec2 cross-corpus
python scripts/run_dayF13_finetune_wav2vec2.py             # wav2vec2 multi-corpus (no-synth + with-synth)
python scripts/run_dayF14_cross_corpus_matrix.py           # central report figure
python scripts/run_dayF9_meta_v2.py                        # meta v2 (frozen audio + combined text)
python scripts/run_dayF15_ensemble_v3.py                   # meta v3 (fine-tuned audio collapsed)
python scripts/run_dayF16_ensemble_v4.py                   # meta v4 final (5-class audio + SMOTE + cal + threshold)
python scripts/run_dayF17_simulate_handover_v4.py          # conversation-level handover (v4)

# Live demo (uses v4 components: roberta_combined + wav2vec2_finetuned + meta_v4_calibrated)
python -m src.demo.app
```

## Headline results

### Per-utterance (MELD test set)

| Configuration                    | Weighted F1 | Macro F1 | Frustration Recall |
|----------------------------------|------------:|---------:|-------------------:|
| VADER only                       |       39.7% |    24.7% |              16.0% |
| GoEmotions only                  |       55.9% |    42.2% |              38.0% |
| Audio (wav2vec2-IEMOCAP frozen)  |       44.8% |    30.3% |               0.0% |
| **RoBERTa baseline (MELD only)** |   **64.4%** |    46.7% |              14.0% |
| Score fusion v1 (MLP)            |       64.7% |    47.3% |              14.0% |
| Late fusion v1 (weighted avg)    |       66.0% |    48.2% |              14.0% |
| Score fusion **v2** (combined RoBERTa) |   **65.8%** |    49.4% |              22.0% |
| **Ensemble v4 (final, 20-dim + cal)** | **65.3%** |   48.9% |              16.0% |

### Handover (binary anger+frustration)

| Version | Threshold rule | Precision | Recall | F1 | Frust Recall |
|---|---|---:|---:|---:|---:|
| v1/v2/v3 | t=0.30 (single) | 53.8% | 46.9% | 50.1% | 24.0% |
| **v4 (final)** | **t=0.20, w_anger=0.6, w_frust=0.4** | 50.4% | **64.6%** | **56.6%** | **46.0%** |

### Conversation-level handover (sliding window, v4 final)

| Metric                          | v1     | **v4** |
|---------------------------------|-------:|-------:|
| Handover recall (per call)      | 88.8%  | **91.2%** |
| Handover precision (per call)   | 75.5%  | 74.5%  |
| Mean catch latency (utterances) | -0.5   | **-0.76** (catches *before* the labelled negative turn) |
| Caught conversations            | 151/170 | **155/170** |

### Cross-corpus generalisation (Phase 9)

**Text (Weighted F1)**
| Train ↓ \ Test → | MELD test | Synth test |
|---|---:|---:|
| MELD only | 60.6% | 46.6% |
| Synth only | 39.3% | 83.7% |
| **MELD + Synth** | **65.4%** | **82.4%** |

**Audio (Weighted F1)**
| Train ↓ \ Test → | MELD test | CREMA-D test | Synth test |
|---|---:|---:|---:|
| Frozen IEMOCAP | n/a | 53.6% | n/a |
| MELD + CREMA-D | 9.3% | 36.5% | 14.8% |
| **MELD + CREMA-D + Synth** | **45.3%** | **53.7%** | 28.4% |

## Limitations

- **MELD is sitcom dialogue (Friends), not contact-centre data.** Emotions are
  acted, the topic distribution has nothing to do with customer support, and
  the audio is mixed with laugh tracks and music. Headline numbers should not
  be read as deployment-ready accuracy.
- **There is no "frustration" class in MELD.** We fold MELD's `fear` into
  `frustration`, but a manual inspection of the training texts shows they are
  mostly *startle/fear* utterances ("please don't hurt me"), not customer
  frustration. This is the dominant cause of the low frustration recall.
- **Class imbalance is severe** (268 frustration vs 4 709 neutral on train).
  We tested SMOTE oversampling + isotonic calibration
  (`src/training/train_meta_balanced.py`) and confirmed that the cap on
  frustration recall (~14%) is a *data* problem, not a *model* problem - no
  algorithmic remedy moved the metric.
- **Whisper transcription quality on MELD is uneven** because of background
  music and laugh tracks; we report metrics with both ASR text and the MELD
  ground-truth transcripts to isolate the effect.
- **Audio model is zero-shot.** `wav2vec2-IEMOCAP` was not fine-tuned on MELD.
  IEMOCAP has no frustration class, so the audio path contributes 0 mass to
  the frustration target by design.

## Ethics, privacy, bias

- **Voice is biometric.** Any production deployment of this system would
  require informed consent at the start of every call and a clear retention
  policy. Models and stored features should be deletable on user request to
  comply with GDPR.
- **Demographic bias.** MELD draws from an English-language US sitcom; we have
  not validated the model on accented English, code-switching, or non-English
  speech. The audio classifier (IEMOCAP) is similarly narrow.
- **Chilling effect.** A monitoring system that escalates "frustrated" callers
  may also nudge agents to suppress legitimate complaints in front of the
  microphone. The handover threshold should be transparent to users.
- **Failure modes.** False positives (escalating non-frustrated callers)
  inconvenience users; false negatives (missed frustration) are the failure
  mode the system is meant to fix. Threshold tuning in `error_analysis.py`
  optimises for recall under a precision floor for that reason.

## License

Educational use only - this is academic coursework, not an open-source release.
