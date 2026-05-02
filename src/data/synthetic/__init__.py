"""SmartHandover - Synthetic data generation pipeline.

Modules
-------
* config      : single source of truth for sizes, paths, models, prompts
* diversity   : controlled sampling over the 5-axis diversity space
* _openai_client : shared OpenAI client with retry / rate-limit / backoff
* generate_text  : LLM text generator (resumable, checkpointed)
* filter_text    : LLM-as-judge + heuristic quality filter
* generate_audio : TTS generator with per-sample emotional instructions
* validate       : CLI listening test + Cohen's kappa report
"""
