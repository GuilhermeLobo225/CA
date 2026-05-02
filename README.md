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

# Week 3 - live demo
python -m src.demo.app
```

## Headline results (MELD test set)

| Configuration                    | Weighted F1 | Macro F1 | Frustration Recall |
|----------------------------------|------------:|---------:|-------------------:|
| VADER only                       |       39.7% |    24.7% |              16.0% |
| GoEmotions only                  |       55.9% |    42.2% |              38.0% |
| Audio (wav2vec2-IEMOCAP) only    |       44.8% |    30.3% |               0.0% |
| **RoBERTa fine-tuned only**      |   **64.4%** |    46.7% |              14.0% |
| Score fusion (MLP, Day 6)        |       64.7% |    47.3% |              14.0% |
| **Late fusion (weighted avg)**   |   **66.0%** |**48.2%** |              14.0% |
| Decision fusion (majority vote)  |       63.1% |    45.3% |              12.0% |

Conversation-level handover (sliding window, threshold = 0.30):

| Metric                       | Value |
|------------------------------|------:|
| Handover recall (per call)   | 88.8% |
| Handover precision (per call)| 75.5% |
| Mean catch latency           | -0.5 utterances (typically caught **before** the labelled negative turn) |

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
