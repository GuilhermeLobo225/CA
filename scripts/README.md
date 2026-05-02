# scripts/

Thin entry-point scripts for each day of the sprint. They orchestrate
modules from `src/` and are designed to be run **from the project root**:

```bash
python scripts/run_day1_vader.py     # VADER baseline -> data/processed/vader_predictions.csv
python scripts/run_day2_goemo.py     # GoEmotions zero-shot
python scripts/run_day3_audio.py     # SpeechBrain (wav2vec2-IEMOCAP) + Whisper smoke test
python scripts/run_day4_train.py     # Fine-tune RoBERTa on MELD (text-only)
python scripts/run_day5_ensemble.py  # Week-1 weighted-average baseline
```

For Week 2+ deliverables run the modules directly:

```bash
python -m src.classifiers.ensemble_trainer       # meta-classifier (Day 6)
python -m src.evaluation.ablation                # leave-one-out study (Day 7)
python -m src.evaluation.error_analysis          # error analysis + threshold sweep (Day 8)
python -m src.classifiers.fusion_strategies      # 3-way fusion comparison
python -m src.classifiers.pipeline               # end-to-end smoke test
python -m src.decision.simulate_handover         # conversation-level handover sim (Day 10)
python -m src.training.train_meta_balanced       # SMOTE + isotonic calibration variant
python -m src.demo.app                           # Gradio demo (Day 11)
```

Synthetic-data pipeline (Phase 4):

```bash
python -m src.data.synthetic.generate_text       # text via OpenAI / IAEDU
python -m src.data.synthetic.filter_text         # heuristic + LLM-judge filter
python -m src.data.synthetic.generate_audio      # TTS via gpt-4o-mini-tts
python -m src.data.synthetic.validate sample     # listening-test sheet
python -m src.data.synthetic.validate annotate --name <you>
python -m src.data.synthetic.validate score      # Cohen's kappa report
```
