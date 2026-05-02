# Deprecated training code

`train_multimodal.py` was the original Phase-2 multimodal trainer
(RoBERTa + wav2vec2 fusion). It was abandoned in favour of the
ensemble approach because:

* The data ceiling on MELD (no real frustration, ~268 samples) made
  end-to-end multimodal fine-tuning unstable.
* The current pipeline reaches comparable F1 with a fraction of the
  GPU/training time by combining pre-trained models in an ensemble.

The file is kept here for traceability. It is broken on `import`
(references `src/models/fusion_model.py`, removed during the cleanup)
and is no longer reachable from any active script.
