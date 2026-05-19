# scripts/

Entry-point scripts that run from the project root. Each writes a JSON
summary (or CSV) to `data/processed/` and exits — every number quoted
in the top-level README comes from one of these outputs.

## v4 pipeline (baseline)

```bash
python scripts/run_dayF8_retrain_roberta.py    # 4-condition RoBERTa retrain
python scripts/run_dayF13_finetune_wav2vec2.py # wav2vec2 fine-tune
python scripts/run_dayF14_cross_corpus_matrix.py  # 4x4 cross-corpus eval
python scripts/run_dayF15_ensemble_v3.py        # 19-dim ensemble v3
python scripts/run_dayF16_ensemble_v4.py        # 20-dim ensemble v4 (LR + isotonic)
python scripts/run_dayF17_simulate_handover_v4.py # conversation-level sim
```

## v5 pipeline (current best)

```bash
python scripts/run_dayG1_train_text_v5.py       # focal-loss RoBERTa re-fine-tune
python scripts/run_dayG2_features_v5.py         # rebuild 20-dim feature CSVs
python scripts/run_dayG3_meta_v5.py             # LR / XGB-grid / MLP / Stacking
python scripts/run_dayG4_threshold_v5.py        # threshold sweep (Rule A + Rule B)
python scripts/run_dayG5_simulate_handover_v5.py # conversation-level sim
python scripts/run_dayG8_final_summary.py        # consolidated JSON
```

## Synthetic-data pipeline (text + audio, run once)

```bash
python -m src.data.synthetic.generate_text       # text via OpenAI / IAEDU
python -m src.data.synthetic.filter_text         # heuristics + LLM-judge filter
python -m src.data.synthetic.generate_audio      # TTS via gpt-4o-mini-tts
```

Synthetic outputs land in `data/synthetic/` and are gitignored (large +
require an API key to regenerate).

## Demo

```bash
python -m src.demo.app                           # Gradio demo on the v4 pipeline
```
