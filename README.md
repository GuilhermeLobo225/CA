# SmartHandover — Multimodal Frustration Detection for Customer-Support Calls

> Master's project for the **Computação Afetiva** unit, MIA, Universidade do
> Minho (2025/2026).

## Team

| Number  | Name                              |
|---------|-----------------------------------|
| PG60225 | Guilherme Lobo Pinto              |
| PG60289 | Pedro Alexandre Silva Gomes       |
| PG60393 | Simão Novais Vieira da Silva      |

## What this is

A multimodal emotion classifier whose final decision is whether to **hand
over the call to a human agent**. Each utterance is scored by four
pretrained components (one audio, three text) whose probabilities are
fed to a meta-classifier; a thresholded weighted combination of
`P(anger)` and `P(frustration)` is the handover trigger. Evaluation is on
[MELD](https://huggingface.co/datasets/ajyy/MELD_audio), augmented with a
generated synthetic corpus to attack MELD's severe frustration shortage.
**MELD is sitcom dialogue from *Friends*** — these numbers should be
read as a benchmark on that distribution, not a prediction of contact-centre
performance.

## System architecture (v5)

```
              audio (16 kHz mono)
                       |
         +-------------+--------------+
         |                            |
         v                            v
    Whisper-small                 wav2vec2-large
       (ASR)                     superb-er, fine-tuned
       transcript                P(anger / frust / sad /
         |                       neutral / satisfaction)
   +-----+-----+-------------+               |
   |           |             |               |
   v           v             v               |
 VADER     j-hartmann      RoBERTa-base      |
 (lexicon) emo-distil-     fine-tuned        |
           roberta-base    on MELD + synth   |
 4 dims    6 dims          5 dims            5 dims
   |           |             |               |
   +-----------+-------------+---------------+
                       |
                  20-dim feature vector
                       |
                       v
        Meta-classifier  (Stacking: LR + XGB soft-vote,
                          trained on SMOTE-resampled v5 features)
                       |
                       v
        handover_score = 0.6 * P(anger) + 0.4 * P(frustration)
                       |
        +--------------+---------------+
        |                              |
   instant rule:               sustained rule:
   score > 0.10               mean of last 3 scores
                              > 0.10 * 0.7
        |                              |
        +--------------+---------------+
                       |
                handover trigger
```

| Component                                       | Role                       | Trained by us? |
|--------------------------------------------------|----------------------------|----------------|
| `openai/whisper-small`                           | ASR (audio → text)         | No (pretrained)|
| `vaderSentiment`                                 | Lexicon sentiment, 4 dims  | No             |
| `j-hartmann/emotion-english-distilroberta-base`  | 7-class Ekman+ text classifier (anger, disgust, fear, joy, neutral, sadness, surprise; we use the first 6 columns) | No             |
| `roberta-base` fine-tuned                        | 5-class text classifier, weighted CE | **Yes** (MELD + 19 280 filtered synthetic texts, see `scripts/run_dayF8_retrain_roberta.py`) |
| `superb/wav2vec2-large-superb-er` fine-tuned     | 5-class audio classifier   | **Yes** (MELD + CREMA-D + 9 235 phone-degraded synthetic clips, see `scripts/run_dayF13_finetune_wav2vec2.py`) |
| Meta-classifier (v4 production)                  | Logistic regression with `class_weight="balanced"` on 20-dim features, SMOTE-resampled training set, isotonic calibration | **Yes** |
| Meta-classifier (v5 improvement)                 | Stacking: soft-vote average of LR + XGBoost (grid-searched); no calibration step | **Yes** |

The audio classifier replaced the original frozen
`superb/wav2vec2-large-superb-er` checkpoint after CREMA-D-only fine-tuning
produced **zero** frustration recall on MELD (`audio MELD + CREMA-D → MELD
test`, `cross_corpus_matrix.csv` row); the with-synth variant
(`MELD + CREMA-D + Synth`) is the one used in v4 and v5.

## Emotion model

A discrete categorical model (Ekman-style) reduced to the five classes
relevant for handover:

`anger`, `frustration`, `sadness`, `neutral`, `satisfaction`.

MELD's source labels are remapped in [src/data/load_meld.py](src/data/load_meld.py):

| MELD class | Target            |
|------------|-------------------|
| anger      | anger             |
| disgust    | anger             |
| fear       | **frustration**   |
| joy        | satisfaction      |
| neutral    | neutral           |
| sadness    | sadness           |
| surprise   | (dropped)         |

The `fear → frustration` mapping is the most important caveat in the
whole project; see the Limitations section.

## Data

| Use                        | Source                                                | Size                              |
|----------------------------|-------------------------------------------------------|-----------------------------------|
| Training + evaluation      | `ajyy/MELD_audio` on HuggingFace                      | 8 783 + 958 + 2 329 utterances    |
| Audio cross-corpus         | CREMA-D                                               | 6 171 clips (anger, sadness, neutral, satisfaction; no frustration) |
| Synthetic text             | OpenAI `gpt-4o-mini` (IAEDU pool) → heuristic + Ollama `mistral-small3.1` LLM-as-judge | 21 213 generated → 19 280 filtered |
| Synthetic audio            | OpenAI `gpt-4o-mini-tts` + telephony bandpass (300–3400 Hz, μ-law 8 kHz) | 9 235 phone-degraded clips (`data/synthetic/audio_phone_manifest.csv`) |

MELD class distribution after the 5-class remap (computed from
`src.data.load_meld.load_meld(...)`):

| Split      | n     | anger | frustration | sadness | neutral | satisfaction |
|------------|------:|------:|------------:|--------:|--------:|-------------:|
| train      | 8 783 | 1 380 |         268 |     683 |   4 709 |        1 743 |
| validation |   958 |   175 |          40 |     111 |     469 |          163 |
| test       | 2 329 |   413 |          50 |     208 |   1 256 |          402 |

Frustration is 3.1 % of the train split and 2.1 % of the test split, which
sets the upper bound on per-utterance frustration recall (a single missed
test sample costs 2 percentage points).

The synthetic text was filtered by a two-stage pipeline: length and
banned-phrase heuristics, then an LLM-as-judge (Ollama
`mistral-small3.1`, a different model family from the generator
`gpt-4o-mini`) requiring `judge_score ≥ 4` and intensity match
(`src/data/synthetic/filter_text.py`). The synthetic audio was generated
by `gpt-4o-mini-tts` across 11 voices with per-sample emotional
instructions, then passed through a phone-channel bandpass to match the
contact-centre acoustic conditions.

## Headline results

All numbers below come from JSON / CSV files produced by the scripts in
`scripts/`. Sources are noted next to each block.

### Per-utterance, MELD test (5-class)

| Model                                              | W-F1   | macro F1 | frust recall | frust F1 |
|----------------------------------------------------|-------:|---------:|-------------:|---------:|
| VADER alone (meta-LR on 4 VADER feats)             | 0.3967 |   0.2470 |       0.1600 |        — |
| GoEmotions alone (meta-LR on 6 GoEmo feats)        | 0.5591 |   0.4220 |       0.3800 |        — |
| wav2vec2 fine-tuned (with-synth) alone, argmax     | 0.4532 |   0.2784 |       0.1400 |        — |
| RoBERTa fine-tuned (MELD + synth, argmax of `prob_*`) | 0.6537 |   0.4873 |       0.1800 |        — |
| **v4** — LR + SMOTE + isotonic calibration         | 0.6532 |   0.4896 |       0.1600 |   0.1951 |
| **v5** — Stacking (LR + XGB soft-vote), no calibration | 0.6471 |   0.4746 |   **0.2400** |   0.1538 |

VADER / GoEmotions alone rows are from `data/processed/ablation_results.csv`
(meta-LR trained on only that modality's columns, not the lexicon raw
output). wav2vec2-with-synth row is from `data/processed/dayF13_results.csv`,
`with-synth` × `meld_test`. RoBERTa row is the argmax of the v4 feature
columns `prob_*` on the test set, the same probabilities that feed the
meta-classifier. v4 row is from `meta_classifier_v4_summary.json`
(`calibrated_test`). v5 row is from `meta_classifier_v5_summary.json`
(`calibrated_test`).

v5 trades macro F1 and weighted F1 for an absolute +8 points of
frustration recall.

### Handover (binary anger+frustration), MELD test, **utterance level**

| Config                          | w_anger | w_frust | t    | prec   | recall | F1     | frust recall |
|---------------------------------|--------:|--------:|-----:|-------:|-------:|-------:|-------------:|
| v4 (Rule A: prec floor ≥ 0.50)  |     0.6 |     0.4 | 0.20 | 0.504  | 0.646  | 0.566  |        0.460 |
| v5 (Rule B: max val F1)         |     0.6 |     0.4 | 0.10 | 0.425  | 0.704  | 0.530  |        0.540 |
| v5 (Rule A fallback)            |    0.55 |    0.45 | 0.10 | 0.424  | 0.702  | 0.528  |        0.540 |

v4 numbers from `configs/handover_threshold_v4.json` (`test_metrics`);
v5 numbers from `configs/handover_threshold_v5.json`. v5 sacrifices ~8
points of precision for ~6 points of recall and ~8 points of frustration
recall.

### Handover, MELD test, **conversation level**

| Config | conv. recall | conv. precision | false-handover rate on clean conv. | mean catch latency (utt.) |
|--------|-------------:|----------------:|-----------------------------------:|--------------------------:|
| v4     |       0.9118 |          0.7452 |                             0.4862 |                    −0.76  |
| v5     |       0.9588 |          0.7181 |                             0.5872 |                    −1.14  |

From `handover_simulation_v4_summary.json` and
`handover_simulation_v5_summary.json`. v5 catches 5 percentage points
more conversations but the false-handover rate on clean conversations
rose from 48.6 % to 58.7 %. Both versions fire **before** the first
labelled negative utterance on average (negative latency), which means
the system pre-emptively flags conversations that *will* go negative
based on early-utterance cues — useful for some operations, alarming for
others.

## Cross-corpus generalisation

From `data/processed/cross_corpus_matrix.csv`. The columns are the test
sets; the rows are what the model was trained on.

| Modality | Train on               | MELD test (W-F1) | MELD test (FrustR) | CREMA-D test | Synth test |
|----------|------------------------|-----------------:|-------------------:|-------------:|-----------:|
| text     | MELD only              |           0.6062 |             0.2600 |            — |     0.4656 |
| text     | Synth only             |           0.3934 |             0.4400 |            — |     0.8371 |
| text     | MELD + Synth           |           0.6537 |             0.1800 |            — |     0.8236 |
| audio    | Frozen IEMOCAP         |                — |                  — |       0.5364 |          — |
| audio    | MELD + CREMA-D         |           0.0933 |             0.6600 |       0.3653 |     0.1484 |
| audio    | MELD + CREMA-D + Synth |           0.4532 |             0.1400 |       0.5366 |     0.2843 |

`Synth only` text training collapses on MELD weighted F1 (0.39) even
though it has by far the best synthetic-test frustration recall (0.65),
which is the clearest available evidence that the synthetic distribution
is meaningfully different from the MELD distribution.

The `audio MELD + CREMA-D` row has 66 % MELD frustration recall but 9 %
W-F1, i.e. it predicts frustration on nearly everything; adding the
synthetic corpus regularises the model back to 45 % W-F1 at the cost of
that recall.

## Limitations

1. **MELD is sitcom dialogue.** Numbers here are not a prediction of
   contact-centre performance. *Friends* features stagey delivery,
   laugh-tracks and crosstalk; the acoustic and pragmatic distributions
   are far from a phone-band customer-support call.
2. **MELD's "frustration" is repurposed "fear".** The five-class remap
   folds `fear` into `frustration` (`src/data/load_meld.py`). Inspecting
   MELD's fear utterances shows most of them are sitcom startle (*"please
   don't hurt me"*), not service-context frustration. This is the
   dominant ceiling on per-utterance frustration recall — the labels we
   are scored against are systematically different from what a
   contact-centre annotator would call frustration.
3. **Severe class imbalance.** Frustration is 268/8 783 ≈ 3.1 % of train
   and 50/2 329 ≈ 2.1 % of test. The meta-classifier sees a SMOTE-balanced
   training set; the test numbers reflect the natural distribution.
4. **Whisper transcription is noisy on MELD.** Laugh tracks, music and
   crosstalk degrade ASR. The pipeline does not measure WER, so this is
   reflected only indirectly through the downstream metrics.
5. **The handover system has a high false-trigger rate.** On clean MELD
   conversations the v4 false-handover rate is **48.6 %** and v5 is
   **58.7 %** (`handover_simulation_v*_summary.json`). This is an
   operational concern — a system that hands over the majority of clean
   calls would be unusable. The threshold could be raised at the cost of
   recall; the current thresholds optimise for catching frustration on
   MELD's distribution, not for false-trigger control.
6. **The mean catch latency is negative.** v4 = −0.76 utterances, v5 =
   −1.14. The system flags conversations on early-utterance cues before
   the labelled negative-emotion utterance appears. This is *useful* if
   you want pre-emptive routing, but it also means "latency" as a metric
   is fragile — the system can "catch" by triggering on every long
   conversation.

## Ethics, privacy, bias

**Voice is biometric.** A pipeline that listens to customer audio, even
if it only emits a probability, has access to a biometric identifier. A
production deployment of anything like this needs the legal basis and
retention controls of a biometric data pipeline.

**Demographic narrowness.** MELD is an English-language American sitcom;
CREMA-D speakers are 91 actors with limited dialect coverage; the
synthetic TTS pool is 11 OpenAI voices, all also English. The system has
no exposure to non-native English, code-switching, or dialect variation,
and inherits whatever speaker-demographic biases sit inside
`wav2vec2-large-superb-er`. We do not measure or report fairness across
demographic axes — there is no validation set that supports it.

**Chilling effect of monitoring.** A live emotion classifier on a
customer call is a surveillance instrument; even ignoring accuracy, its
mere presence can suppress legitimate complaints (callers who know they
are being assessed may behave more compliantly). Honest deployment
requires telling the caller it is happening.

**Failure-mode trade-offs in threshold calibration.** The v5 threshold
(0.10) reflects a deliberate bias toward catching frustration. The cost
is the high false-handover rate. The opposite calibration —
high-precision, low-recall — would miss real frustration and reduce
agent caseload at the cost of caller experience. Neither extreme is
neutral; pretending the threshold is a "technical detail" hides the
moral choice.

## Reproducing the v5 results

Pre-requisites: Python 3.11, `pip install -r requirements.txt`,
a CUDA GPU (the focal-loss attempt and the wav2vec2 fine-tune both
need one). The HuggingFace MELD audio dataset is fetched on first
`load_meld`. The two large `.pt` checkpoints (RoBERTa + wav2vec2) are
not committed; download them with
`python scripts/download_checkpoints.py`. The synthetic source corpus
is generated offline (`data/synthetic/*.jsonl`, gitignored, requires
an OpenAI key); the post-filter MELD-side feature CSVs are committed.

```bash
# 0) one-time
pip install -r requirements.txt
python scripts/download_checkpoints.py        # only needed for end-to-end / demo

# 1) v5 pipeline (uses committed v4 feature CSVs as input)
python scripts/run_dayG1_train_text_v5.py     # focal-loss RoBERTa re-fine-tune (~10 min, GPU)
python scripts/run_dayG2_features_v5.py        # build 20-dim v5 feature CSVs
python scripts/run_dayG3_meta_v5.py            # train LR / XGB-grid / MLP / Stacking, select winner
python scripts/run_dayG4_threshold_v5.py       # threshold sweep + Rule A/B
python scripts/run_dayG5_simulate_handover_v5.py  # conversation-level simulation
python scripts/run_dayG8_final_summary.py      # consolidate all JSONs
```

To regenerate the v4 baseline as a fallback comparison (these are the
scripts whose outputs are already on disk and feed v5):

```bash
python scripts/run_dayF8_retrain_roberta.py    # 4 RoBERTa conditions (MELD / synth / combined / combined_cw)
python scripts/run_dayF13_finetune_wav2vec2.py # wav2vec2 fine-tune
python scripts/run_dayF14_cross_corpus_matrix.py
python scripts/run_dayF16_ensemble_v4.py        # builds v4 features + LR_balanced meta + isotonic
python scripts/run_dayF17_simulate_handover_v4.py
```

## Repository layout

```
checkpoints/
    meta_classifier_v4.pkl              uncalibrated v4 winner (LR_balanced)
    meta_classifier_v4_calibrated.pkl   v4 + isotonic, production fallback
    meta_classifier_v5.pkl              v5 winner (StackingSoftVote)
    meta_classifier_v5_calibrated.pkl   v5 winner (Stacking not calibrated; same model)

    # large .pt files (not in git, fetched by scripts/download_checkpoints.py):
    #   roberta_combined.pt              487 MB
    #   wav2vec2_finetuned.pt           1233 MB
    #   roberta_*.pt (ablation variants) 487 MB each

configs/
    config.yaml                         pipeline knobs (whisper size, batch)
    handover_threshold_v4.json          w_anger=0.6, w_frust=0.4, t=0.20
    handover_threshold_v5.json          w_anger=0.6, w_frust=0.4, t=0.10 (Rule B)
    iaedu_accounts.example.json         template for the synthetic-text API pool

data/processed/
    ensemble_features_{train,val,test}_v4.csv    20-dim, source for v4 meta
    ensemble_features_{train,val,test}_v5.csv    20-dim, same prob_* as v4 (Phase 1 fell back)
    meta_classifier_v4_summary.json              v4 candidate metrics + calibrated test
    meta_classifier_v5_summary.json              v5 candidate metrics + v4 comparison
    handover_simulation_v4_summary.json          conv-level v4
    handover_simulation_v5_summary.json          conv-level v5
    dayG1_summary.json                           focal-loss attempt result + abort reason
    dayG2_summary.json                           feature build provenance
    dayG3_summary.json                           XGB grid trace
    dayG4_threshold_sweep_v5.csv                  full (w_anger, t) sweep on val + test
    v5_final_summary.json                         single source of truth used by the README
    dayF8_*.csv                                   RoBERTa retraining condition predictions
    dayF13_*.json|csv                             wav2vec2 fine-tune history + results
    dayF16_summary.json, dayF16_threshold_sweep_v4.csv   v4 production artefacts
    cross_corpus_matrix.csv                       text/audio cross-train matrix (4×4)
    ablation_results.csv                          leave-one-modality-out
    cremad_baseline_summary.json                  frozen wav2vec2 IEMOCAP on CREMA-D
    audio_v3_predictions.csv                      wav2vec2 fine-tuned per-utterance probs
    vader_predictions.csv                          VADER baselines
    goemo_predictions.csv                          GoEmotions baselines
    speechbrain_predictions.csv, cremad_speechbrain_predictions.csv   audio baselines
    top_20_errors.csv                              qualitative error sample
    roberta_predictions_*_v5.csv                   v5 focal-loss RoBERTa predictions (unused)

data/synthetic/
    audio_phone_manifest.csv                       9 235-row manifest of phone-band TTS clips
    # text.jsonl, text_filtered.jsonl, audio/, audio_phone/  are gitignored
    # (regenerable via the src/data/synthetic/ pipeline, OpenAI key required)

scripts/
    download_checkpoints.py                        fetch the .pt weights from the GitHub Release
    run_dayF8_retrain_roberta.py                   v4 RoBERTa retraining
    run_dayF13_finetune_wav2vec2.py                v4 wav2vec2 fine-tune
    run_dayF14_cross_corpus_matrix.py
    run_dayF15_ensemble_v3.py                       v3 19-dim ensemble (intermediate, kept for traceability)
    run_dayF16_ensemble_v4.py                       v4 20-dim ensemble + LR + isotonic
    run_dayF17_simulate_handover_v4.py              v4 conversation simulation
    run_dayG1_train_text_v5.py                      Phase 1: focal-loss RoBERTa re-fine-tune
    run_dayG2_features_v5.py                        Phase 2: build v5 feature CSVs
    run_dayG3_meta_v5.py                            Phase 3: meta-classifier grid + selection
    run_dayG4_threshold_v5.py                       Phase 4: threshold sweep (Rule A + Rule B)
    run_dayG5_simulate_handover_v5.py               Phase 5: v5 conversation simulation
    run_dayG8_final_summary.py                      Phase 8: aggregated JSON for the README

src/
    classifiers/
        ensemble_trainer.py, ensemble.py            shared constants + meta-classifier scaffolding
        fusion_strategies.py                         late / decision / score fusion
        goemo_classifier.py, vader_classifier.py     pretrained wrappers
        speechbrain_classifier.py                    frozen wav2vec2 IEMOCAP wrapper
        wav2vec2_finetuned.py                        loader for the v4 fine-tuned audio model
        whisper_asr.py                               ASR wrapper
        pipeline.py                                  v1 reference pipeline
        pipeline_v4.py                               v4 end-to-end inference (audio → handover)
        stacking.py                                  StackingSoftVote, used by v5
    data/
        load_meld.py                                 MELD ingestion + 5-class remap
        load_cremad.py                               CREMA-D ingestion
        synthetic/                                    generate_text / filter_text / generate_audio
    decision/
        handover.py                                  instant + sustained rule
        simulate_handover.py                         simulation helpers
    evaluation/
        metrics.py, ablation.py, error_analysis.py
    models/
        text_encoder.py                              RoBERTa encoder with attention pooling
    training/
        train_text.py                                v1 / v4 weighted-CE training
        train_text_v5.py                             v5 focal-loss training
        train_audio.py                               wav2vec2 fine-tuning
        train_meta_balanced.py                       SMOTE + isotonic meta-trainer
        focal_loss.py                                multi-class focal loss with per-class alpha
    demo/
        app.py                                       Gradio demo on the v4 pipeline
        app_callcenter.py                            call-centre-flavoured demo

requirements.txt
README.md
```

## Honesty notes on v5

* Phase 1 (focal-loss RoBERTa re-fine-tune) **did not clear its stop bar**:
  best val frustration recall was 0.1500 after 10 epochs, equal to v4's
  weighted-CE RoBERTa val (0.1500) and below the 0.32 threshold set by
  the plan. Phase 2 therefore reused the v4 RoBERTa probability columns;
  the `prob_*` columns in `ensemble_features_*_v5.csv` are
  byte-identical to v4's. See `data/processed/dayG1_summary.json` for
  the trajectory and the explicit abort reason.
* What actually changed in v5 vs v4: the meta-classifier (Stacking
  instead of plain LR + isotonic), the selection rule
  (`0.5*frust_recall + 0.5*macro_f1` instead of `frust_recall s.t. W-F1 ≥ 0.55`),
  the threshold (0.10 instead of 0.20) and the threshold selection rule
  (max handover F1 instead of max frust recall under a precision floor).
* The wav2vec2-finetuned audio classifier and the RoBERTa text classifier
  are **shared** between v4 and v5. Only the downstream meta-classifier
  and threshold changed.

## License

Educational use only, academic coursework.
