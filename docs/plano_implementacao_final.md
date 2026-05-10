# Plano de Implementação Final — SmartHandover

> **Para:** Claude Code (execução sequencial)
> **Prazo:** 22 Maio 2026 (sem penalização até 25 Maio); apresentação semana 27 Maio
> **Início:** 2026-05-05
> **Princípios:** validar a cada gate, não avançar se métricas não baterem; preservar `meld_only` baseline em todos os checkpoints para comparação.

---

## 0. Tabela master

| # | Fase | Tempo | Output crítico | Gate de avanço |
|---|---|---|---|---|
| 1 | Re-treino RoBERTa multi-corpus | 30-45 min | `checkpoints/roberta_combined.pt` | Frust recall ≥ 40% no MELD test |
| 2 | Re-build features + meta v2 | ~1h | `checkpoints/meta_classifier_v2.pkl` | W-F1 ensemble ≥ 68% |
| 3 | Smoke test TTS (50 amostras) | 1-2h | 50 .wav + relatório qualitativo | Subjective pass + frozen wav2vec2 confirma sinal |
| 4 | Gerar áudio completo | ~10h background | `data/synthetic/audio_clean/` (~19k WAV) | Manifest completo, < 5% falhas |
| 5 | Degradação telefónica + ruído | ~2h | `data/synthetic/audio_phone/` | Inteligibilidade preservada (smoke test) |
| 6 | Download CREMA-D + indexação | ~1h | `data/raw/CREMA-D/` indexado | Loader passa em 100% |
| 7 | Fine-tune wav2vec2 multi-corpus | ~3h GPU | `checkpoints/wav2vec2_finetuned.pt` | MELD test F1 ≥ frozen baseline |
| 8 | Listening test (kappa) | 2-3h × 3 pessoas | `validation_report.json` | Cohen's κ ≥ 0.4 |
| 9 | Cross-corpus matrix 4×4 | ~1h | `cross_corpus_results.{csv,png}` | Matriz produzida |
| 10 | Re-tune ensemble + handover threshold | ~1h | `handover_threshold_v2.json` | Frust recall (utterance) ≥ 50% |
| 11 | Relatório + slides + demo final | ~5 dias | PDF + .pptx + Gradio | Auto-review check |

---

## FASE 1 — Re-treino RoBERTa multi-corpus

**Objectivo:** Provar que o sintético resolve o ceiling de 14% de frust recall.

### Action items
1. Confirmar que `data/synthetic/text_filtered.jsonl` tem 19 280 amostras:
   ```bash
   wc -l data/synthetic/text_filtered.jsonl
   ```
2. Correr o script existente (já está pronto):
   ```bash
   python scripts/run_dayF8_retrain_roberta.py
   ```
   Treina 4 condições: `meld_only`, `synth_only`, `combined`, `combined_cw`.
3. Inspeccionar `data/processed/dayF8_results.csv`:
   - Coluna `meld_test_frust_recall` para cada condição
   - Coluna `meld_test_weighted_f1`
   - Coluna `synth_test_*` (cross-corpus eval)

### Outputs esperados
- `checkpoints/roberta_meld_only.pt` (baseline atual, 14% frust recall)
- `checkpoints/roberta_synth_only.pt` (cross-corpus puro)
- `checkpoints/roberta_combined.pt` (alvo: best frust recall)
- `checkpoints/roberta_combined_cw.pt` (com class weights)
- `data/processed/dayF8_results.csv` (4 linhas, ~12 colunas)
- `data/processed/dayF8_manifest.json`

### Gate
- **PASS:** condição `combined` ou `combined_cw` atinge frust recall ≥ 40% em MELD test, mantendo W-F1 ≥ 60%.
- **FAIL:** se frust recall < 30%, parar e investigar:
  - Verificar se `synth_only` consegue >50% (se sim, problema é de fusão de domínios — testar weighted sampler)
  - Se `synth_only` também falha, problema é do dataset sintético (não da fusão) — re-correr filtragem com judge mais permissivo

### Decisão a registar
- **Qual checkpoint vai para downstream?** O que tem **maior frust recall** sob condição W-F1 ≥ 65%. Documentar no manifest.

---

## FASE 2 — Re-build features 19-dim + meta-classifier v2

**Objectivo:** Propagar o ganho do RoBERTa para o ensemble.

### Action items
1. Re-correr predições do classificador de texto fine-tuned em todo o MELD com o novo checkpoint:
   ```bash
   python scripts/run_day3_roberta_predict.py --checkpoint checkpoints/roberta_combined.pt --output data/processed/roberta_predictions_v2.csv
   ```
   (Se o script não suportar `--checkpoint`, modificar para aceitar.)
2. Re-construir features 19-dim concatenadas:
   ```bash
   python scripts/run_day6_build_features.py --roberta-pred data/processed/roberta_predictions_v2.csv --output-prefix data/processed/ensemble_features_v2
   ```
3. Re-treinar meta-classifier (3 variantes para escolher):
   ```bash
   python -m src.training.train_meta --features data/processed/ensemble_features_v2_train.csv --output checkpoints/meta_classifier_v2.pkl
   python -m src.training.train_meta_balanced --features data/processed/ensemble_features_v2_train.csv --output checkpoints/meta_classifier_v2_balanced.pkl
   ```
4. Re-correr comparação 3-way de fusão (score / late / decision) com o novo RoBERTa:
   ```bash
   python -m src.classifiers.fusion_strategies --features-dir data/processed/ --output data/processed/fusion_comparison_v2.csv
   ```

### Outputs esperados
- `data/processed/roberta_predictions_v2.csv`
- `data/processed/ensemble_features_v2_{train,val,test}.csv`
- `checkpoints/meta_classifier_v2.pkl`
- `checkpoints/meta_classifier_v2_balanced.pkl`
- `data/processed/fusion_comparison_v2.csv`

### Gate
- **PASS:** ensemble (qualquer das 3 estratégias) atinge W-F1 ≥ 68% **e** frust recall ≥ 45%.
- **FAIL:** se W-F1 cair em relação aos 66.0% actuais, há regressão — investigar weights do late fusion (provavelmente roberta=0.5 já não é óptimo).

---

## FASE 3 — Smoke test TTS

**Objectivo:** Validar a configuração de TTS em pequena escala antes de gastar €30 + 10h.

### Action items
1. Criar script `scripts/run_dayF9_smoke_test_tts.py`:
   - Estratifica 50 amostras de `text_filtered.jsonl`: 10 por classe.
   - 3 voices (recomendado: `nova`, `onyx`, `shimmer` — cobrem registos distintos).
   - Para cada amostra, gerar 1 .wav usando `src/data/synthetic/generate_audio.py` em modo single-call.
   - Output: `data/synthetic/smoke_audio/<label>/<id>_<voice>.wav` (50 ficheiros) + `smoke_manifest.csv`.
2. **Listening manual** (humano, 30 min):
   - Correr `python -m src.data.synthetic.validate sample --input data/synthetic/smoke_audio/ --n 50`
   - Anotar para cada: `correct_class` (sim/não), `intensity_match` (1-5), `naturalness` (1-5).
   - Critério mínimo: ≥35/50 correct_class (70%).
3. **Smoke automático** com wav2vec2 frozen:
   ```bash
   python scripts/run_dayF9_smoke_eval.py --audio-dir data/synthetic/smoke_audio/ --output data/synthetic/smoke_frozen_eval.csv
   ```
   Logar P(angry), P(sad), P(happy), P(neu) por amostra. **Esperado:** anger sintético deve ter P(ang) > 0.3 em maioria; se for ~0.1 uniforme, TTS está fraco para emoção.

### Outputs esperados
- `data/synthetic/smoke_audio/` com 50 .wav
- `data/synthetic/smoke_manifest.csv`
- `data/synthetic/smoke_frozen_eval.csv`
- `data/synthetic/smoke_listening.csv` (anotação manual)
- `docs/smoke_tts_report.md` (1 página: decisões + samples problemáticas)

### Gate
- **PASS:** ≥70% correct_class na anotação humana **E** classes frust/anger têm P(ang) frozen > 0.25 mediano.
- **FAIL — refusal mode:** se ≥10 amostras saíram com instruções recusadas pelo TTS, suavizar prompt em `generate_audio.py`:
  - Substituir "shouting in rage" por "voice strained, near loss of composure"
  - Substituir "screaming" por "raised voice, audible tension"
  - Re-correr smoke.
- **FAIL — naturalidade baixa:** se naturalness mediana < 2.5, voices estão mal alinhadas com personas. Implementar mapping voice × persona em `generate_audio.py` (ver §3.2 abaixo).

### §3.2 — Voice × persona mapping (implementar se gate falhar)

Em `src/data/synthetic/generate_audio.py`, substituir round-robin puro por mapping:
```python
VOICE_BY_PERSONA = {
    "older_male":   ["onyx", "ash"],
    "younger_male": ["echo", "alloy"],
    "older_female": ["sage", "shimmer"],
    "younger_female": ["nova", "coral"],
    "neutral_adult": ["alloy", "fable"],
    "frustrated_professional": ["onyx", "echo"],
}
```
Round-robin **dentro** do mapping da persona (não global).

---

## FASE 4 — Geração áudio completa

**Objectivo:** Produzir as ~19 280 .wav.

### Action items
1. Confirmar que smoke test passou (Fase 3 gate).
2. Confirmar saldo OpenAI suficiente (~€35 com margem):
   ```bash
   python scripts/diagnose_api.py --check-tts-balance
   ```
3. Lançar geração em background:
   ```bash
   nohup python -m src.data.synthetic.generate_audio \
     --input data/synthetic/text_filtered.jsonl \
     --output-dir data/synthetic/audio_clean/ \
     --workers 4 \
     --resume \
     > logs/audio_gen.log 2>&1 &
   ```
4. Monitorização periódica (não a cada hora — confiar no `--resume`):
   ```bash
   tail -f logs/audio_gen.log | grep -E "(ERROR|done|cost)"
   wc -l data/synthetic/audio_manifest.csv
   ```
5. Após terminar, validar integridade:
   ```bash
   python scripts/validate_audio_corpus.py \
     --manifest data/synthetic/audio_manifest.csv \
     --audio-dir data/synthetic/audio_clean/
   ```
   Verifica: ficheiros existem, duração 1-15s, sample rate uniforme, nenhum corrupto.

### Outputs esperados
- `data/synthetic/audio_clean/<label>/<id>.wav` (~19 280 ficheiros, 24 kHz mono)
- `data/synthetic/audio_manifest.csv` (id, text, label, voice, instruction, duration, cost)
- `logs/audio_gen.log`

### Gate
- **PASS:** ≥95% das amostras geradas com sucesso (≥18 316), custo ≤ €35.
- **FAIL — alta taxa de falhas:** investigar logs por padrão (rate limit? content policy?). Se for content policy, refinar instruções nas amostras falhadas e re-correr só essas.

---

## FASE 5 — Degradação telefónica + ruído

**Objectivo:** Aproximar o áudio sintético da distribuição de áudio de call center real, evitando que o wav2vec2 aprenda artefactos de TTS.

> **Esta fase é o que distingue um pipeline ingénuo de um defensável no relatório.** Sem ela, o wav2vec2 aprende a reconhecer "TTS" em vez de "frustração".

### Action items
1. Criar `src/data/synthetic/degrade_audio.py` com:
   ```python
   def degrade_to_phone(audio_24k: np.ndarray, sr_in: int = 24000) -> np.ndarray:
       """24 kHz studio → 8 kHz μ-law → 16 kHz phone-band."""
       # Resample 24k → 8k (anti-alias)
       # μ-law encode/decode (torchaudio.functional.mu_law_encoding/decoding)
       # Resample 8k → 16k
       # Optional: bandpass 300-3400 Hz (Butterworth ordem 4)
       return audio_16k

   def add_call_center_noise(audio: np.ndarray, snr_db_range=(15, 25),
                              noise_dir: Path) -> np.ndarray:
       """Mix com sample aleatório de noise_dir a SNR aleatório."""
       # Carregar noise sample, ajustar duração (loop ou crop)
       # Calcular gain para SNR target
       # Sum
       return mixed
   ```
2. Descarregar fontes de ruído:
   - **MUSAN noise subset** (~6 GB): https://www.openslr.org/17/
     - Subset relevante: `musan/noise/free-sound/` (background noise variado)
   - **ESC-50 office sounds** (~600 MB, opcional): https://github.com/karolpiczak/ESC-50
     - Filtrar tags: `keyboard_typing`, `mouse_click`, `office`
3. Criar script `scripts/run_dayF10_degrade_audio.py`:
   - Para cada .wav em `audio_clean/`, aplicar `degrade_to_phone` + `add_call_center_noise`.
   - SNR aleatório por amostra em [15, 25] dB.
   - 50% das amostras com ruído de fundo, 50% só telefónico (variabilidade).
   - Output em `data/synthetic/audio_phone/<label>/<id>.wav` (mesma estrutura).
   - Gain ±6 dB aleatório para invariância.
4. Smoke test pós-degradação (5-10 amostras):
   - Ouvi-las: inteligibilidade preservada? Soa a chamada real?
   - Passar pelo Whisper: WER < 30% (o áudio limpo deve ter WER ~5-10%; phone-band tipicamente 15-25%).
5. **Manter os dois conjuntos:** `audio_clean/` (para listening test) e `audio_phone/` (para treino).

### Outputs esperados
- `src/data/synthetic/degrade_audio.py`
- `scripts/run_dayF10_degrade_audio.py`
- `data/synthetic/audio_phone/<label>/<id>.wav` (mesma cardinalidade que clean)
- `data/synthetic/audio_phone_manifest.csv` (com colunas `noise_added`, `snr_db`, `gain_db`)
- `data/raw/musan/` ou path equivalente

### Gate
- **PASS:** Whisper transcreve audio_phone com WER < 30% mediano; smoke listening confirma que parece "chamada real, não estúdio limpo".
- **FAIL — inteligibilidade colapsa:** SNR demasiado baixo ou bandpass demasiado agressivo. Subir SNR para [20, 30] dB e remover bandpass (ficar só com μ-law).

---

## FASE 6 — Download e indexação CREMA-D

**Objectivo:** Adicionar áudio actuado real ao pool de fine-tuning.

### Action items
1. Download (script já tem loader):
   ```bash
   wget -P data/raw/ https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip
   unzip data/raw/master.zip -d data/raw/
   mv data/raw/CREMA-D-master data/raw/CREMA-D
   ```
   (~580 MB, 7 442 .wav)
2. Indexar:
   ```bash
   python -m src.data.load_cremad --build-index --output data/processed/cremad_index.csv
   ```
3. Aplicar mapeamento já definido no plano:
   - ANG → anger; DIS → anger (proxy); FEA → DROP; HAP → satisfaction; NEU → neutral; SAD → sadness
4. Cross-corpus baseline com wav2vec2 frozen:
   ```bash
   python scripts/run_dayF11_cremad_baseline.py --output data/processed/cremad_speechbrain_predictions.csv
   ```

### Outputs esperados
- `data/raw/CREMA-D/AudioWAV/*.wav` (7 442 ficheiros)
- `data/processed/cremad_index.csv`
- `data/processed/cremad_speechbrain_predictions.csv`

### Gate
- **PASS:** loader indexa 100%, frozen wav2vec2 atinge ≥50% W-F1 em CREMA-D test (esperado, dado que CREMA-D é distribuição parecida com IEMOCAP).
- **FAIL:** se frozen wav2vec2 < 35% em CREMA-D, há mismatch maior do que esperado — verificar se o mapeamento de classes está correcto.

---

## FASE 7 — Fine-tune wav2vec2 multi-corpus

**Objectivo:** Levantar o componente acústico de "irrelevante" (+1.4 pp ablation) para "contributo real".

### Action items
1. Criar `scripts/run_dayF12_finetune_wav2vec2.py`:
   - Modelo base: `superb/wav2vec2-large-superb-er` (mesmo do frozen actual)
   - Cabeça de classificação: 5 classes (anger, frustration, sadness, neutral, satisfaction)
   - Encoder freeze 2 épocas → unfreeze top 4 layers (mesma estratégia do RoBERTa)
   - Training data:
     - MELD train (audio): peso 1.0
     - CREMA-D train (audio): peso 1.0
     - synth-phone (audio_phone): peso **0.5**
   - **Validação só em MELD val** (real). Nunca em sintético.
   - **Test em 3 splits separados:** MELD test, CREMA-D test, synth test.
   - Hiperparâmetros:
     - LR 1e-5 (cabeça), 5e-6 (encoder após unfreeze)
     - Batch 16, FP16, max 8 épocas
     - Early stopping em MELD val W-F1 (patience=4)
2. Correr (~3h GPU):
   ```bash
   python scripts/run_dayF12_finetune_wav2vec2.py
   ```
3. Avaliar:
   ```bash
   python scripts/eval_wav2vec2_finetuned.py \
     --checkpoint checkpoints/wav2vec2_finetuned.pt \
     --output data/processed/dayF12_results.csv
   ```

### Outputs esperados
- `checkpoints/wav2vec2_finetuned.pt` (~1.2 GB)
- `data/processed/dayF12_results.csv` (linhas: MELD, CREMA-D, synth; colunas: W-F1, macro F1, frust recall)
- `data/processed/dayF12_train_curves.png`

### Gate **(crítico — onde se detecta overfit a artefactos TTS)**
- **PASS:** MELD test W-F1 ≥ frozen baseline (44.8%) **E** gap (synth_test_F1 − MELD_test_F1) < 15 pp.
- **FAIL — gap > 15 pp:** modelo aprendeu artefactos de TTS, não emoção. Acções:
  - Reduzir peso do sintético para 0.25
  - Aumentar peso de MELD para 1.5
  - Re-correr
- **FAIL — MELD test cai abaixo do frozen:** sintético está a corromper o sinal. Acções:
  - Excluir sintético (treinar só em MELD + CREMA-D)
  - Documentar no relatório como limitação descoberta
  - Manter `wav2vec2_finetuned_no_synth.pt` como variante

### Output secundário (sempre fazer)
- Treinar variante `wav2vec2_finetuned_no_synth.pt` (só MELD + CREMA-D) para ablation no relatório. Mesmo se a versão com sintético passar, esta é a baseline para mostrar o ganho do sintético.

---

## FASE 8 — Listening test (validação humana)

**Objectivo:** Cohen's κ entre 3 anotadores, validar que o sintético soa como anotado.

### Action items
1. Sample estratificado:
   ```bash
   python -m src.data.synthetic.validate sample \
     --input data/synthetic/audio_clean/ \
     --n 200 \
     --output data/synthetic/listening_sample.csv
   ```
   (Usar `audio_clean`, não `audio_phone` — anotadores devem ouvir o sinal limpo.)
2. Distribuir entre os 3 elementos do grupo:
   - Cada um anota as **mesmas 200 amostras** independentemente (overlap total para κ).
   - Cada um regista: classe percebida (5 opções), intensidade (1-5), confiança (low/med/high).
   - Tempo estimado: 2-3h por pessoa.
3. Após anotações:
   ```bash
   python -m src.data.synthetic.validate score \
     --annotations data/synthetic/listening_*.csv \
     --output data/synthetic/validation_report.json
   ```
   Computa: Cohen's κ pairwise, Fleiss' κ global, agreement matrix por classe.

### Outputs esperados
- `data/synthetic/listening_sample.csv`
- `data/synthetic/listening_pedro.csv`, `_simao.csv`, `_lobo.csv`
- `data/synthetic/validation_report.json`

### Gate
- **PASS:** Fleiss' κ ≥ 0.4 (moderate agreement) **E** ≥60% das amostras com classe-modal igual à pretendida.
- **FAIL — κ < 0.4:** documentar no relatório como **limitação assumida**. Não significa abandonar o sintético — significa que se relata que o áudio sintético não atinge prosódia humana de forma fiável, e que os ganhos do wav2vec2 fine-tuned podem ser parcialmente artefactos. Esta é honestidade científica e **valoriza** o relatório.

---

## FASE 9 — Cross-corpus matrix 4×4

**Objectivo:** Figura central do relatório. Mostra exactamente quanto o sintético generaliza.

### Action items
1. Criar `scripts/run_dayF13_cross_corpus_matrix.py`:
   - **Linhas (treino):** `MELD`, `CREMA-D`, `Synthetic`, `All`
   - **Colunas (teste):** `MELD`, `CREMA-D`, `Synthetic`, `MELD+CallCenter (se houver)`
   - Para cada célula, treinar do zero RoBERTa text + (opcional) wav2vec2 nessa fonte e avaliar.
   - **Atalho viável:** correr só para o classificador de texto (RoBERTa) — para áudio, usar resultados da Fase 7.
2. Output: matriz 4×4 com 2 valores por célula (W-F1 / frust recall).
3. Heatmap em `cross_corpus_matrix.png`.

### Outputs esperados
- `scripts/run_dayF13_cross_corpus_matrix.py`
- `data/processed/cross_corpus_matrix.csv`
- `data/processed/cross_corpus_matrix.png` (heatmap)

### Gate
- Sempre PASS — esta fase produz a evidência (não há critério de aprovação técnico, apenas "está produzido").

### Para o relatório
- Se a célula `(treino=Synthetic, teste=MELD)` tiver bom recall mas baixo W-F1, o sintético está a empurrar para classes negativas (overconfident em frustração). Discutir.
- Se a célula `(treino=All, teste=MELD)` for o melhor, validação clara da estratégia.

---

## FASE 10 — Re-tuning final do ensemble + handover

**Objectivo:** Maximizar a métrica de cabeçalho (frust recall a nível de utterance).

### Action items
1. Re-correr ablation completa com novos componentes:
   ```bash
   python -m src.evaluation.ablation \
     --roberta-checkpoint checkpoints/roberta_combined.pt \
     --wav2vec2-checkpoint checkpoints/wav2vec2_finetuned.pt \
     --output data/processed/ablation_v2.csv
   ```
   Esperado: o componente áudio agora deve subir do +1.4 pp para algo entre +3 e +6 pp.
2. Re-tune late fusion weights (grid search):
   ```bash
   python -m src.classifiers.fusion_strategies \
     --tune-weights \
     --output data/processed/late_fusion_weights_v2.json
   ```
3. Re-tune handover threshold:
   ```bash
   python -m src.evaluation.error_analysis \
     --threshold-sweep \
     --output configs/handover_threshold_v2.json
   ```
   Manter optimização para `recall sob precision floor de 70%`.
4. Re-correr simulação a nível de conversa:
   ```bash
   python -m src.decision.simulate_handover \
     --threshold-config configs/handover_threshold_v2.json \
     --output data/processed/handover_simulation_v2.csv
   ```

### Outputs esperados
- `data/processed/ablation_v2.csv` + `.png`
- `data/processed/late_fusion_weights_v2.json`
- `configs/handover_threshold_v2.json`
- `data/processed/handover_simulation_v2.csv`

### Gate (final do trabalho técnico)
- **TARGET hit:** frust recall (utterance) ≥ 50%, frust recall (conversa) ≥ 90%, W-F1 ensemble ≥ 70%.
- **MINIMUM viable:** frust recall (utterance) ≥ 35%, W-F1 ≥ 67%. Se ficares aqui, ainda há história forte (ganho de 14% → 35% é 2.5× e mostra valor do sintético).

---

## FASE 11 — Relatório, slides, demo

### Action items
1. **Relatório** (alvo: 15-20 páginas):
   - Estrutura já tem em `docs/projeto_estado_atual.md` — material praticamente todo lá.
   - Secções obrigatórias: cenário, modelos de emoção, datasets, metodologia, resultados, ablations, **decisões técnicas (§14 do estado_atual)**, **limitações (§9)**, **ética (§10)**, trabalho futuro.
   - Figuras essenciais: arquitectura SVG (já existe), confusion matrix final, ablation barplot, **cross-corpus heatmap**, threshold sweep curve, learning curves wav2vec2.
   - Página 1: avaliação pelos pares (deltas).
2. **Slides** (~3 min apresentação):
   - 8-10 slides max.
   - Estrutura: problema (1) → solução (1) → arquitectura (2) → dataset sintético (1) → resultados (2) → demo live ou GIF (1) → limitações + ética (1).
3. **Demo Gradio** (`src/demo/app.py`):
   - Actualizar para usar `roberta_combined.pt` e `wav2vec2_finetuned.pt`.
   - Pre-carregar 3-4 amostras de exemplo: 1 neutral, 1 anger, 1 frustration, 1 ambíguo (texto-áudio discordam).
   - Smoke test antes da apresentação.

### Outputs esperados
- `docs/relatorio_final.pdf`
- `docs/apresentacao.pptx`
- Demo Gradio operacional
- `README.md` actualizado para o estado final
- ZIP `CA-GRUPO[G].zip` para Blackboard

### Gate
- Auto-review: cada métrica numerada está com fonte (script + .csv)? Cada figura tem caption + referência no texto? README descreve como reproduzir todos os resultados? Página 1 tem deltas que somam 0.00?

---

## Cronograma proposto (5 Maio → 22 Maio)

| Data | Fase(s) | Notas |
|---|---|---|
| 5 Mai (Ter) | 1 + 2 | RoBERTa retrain + meta v2. **Day-1 critical path.** |
| 6 Mai (Qua) | 3 + iniciar 4 | Smoke TTS de manhã, gen audio overnight |
| 7 Mai (Qui) | 4 (continua) + 6 | CREMA-D em paralelo enquanto audio gera |
| 8 Mai (Sex) | 5 | Degradação acústica (depende de Fase 4 terminar) |
| 11 Mai (Seg) | 7 | Fine-tune wav2vec2 (~3h GPU + análise) |
| 12 Mai (Ter) | 8 (cada um anota 200) | Distribuído pelos 3, em paralelo |
| 13 Mai (Qua) | 9 + 10 | Cross-corpus matrix + re-tune final |
| 14-15 Mai | 11 (escrita) | Primeira versão do relatório |
| 18-20 Mai | 11 (revisão) | Refinar relatório, slides, demo |
| 21 Mai (Qui) | Buffer + ensaio apresentação | |
| 22 Mai (Sex) | **ENTREGA** | |

**Buffer:** 23-25 Maio se algo derraparr (entrega sem penalização até 25).

---

## Decisões a registar à medida que avança

Manter um log em `docs/decisions_log.md` com formato:
```
## YYYY-MM-DD — <decisão>
**Contexto:** ...
**Opções consideradas:** ...
**Escolha:** ...
**Justificação:** ...
**Impacto medido (se aplicável):** ...
```

Decisões previsíveis que vão precisar de log:
- Qual checkpoint RoBERTa vai para downstream (Fase 1 gate)
- Voice × persona mapping (se Fase 3 falhar)
- Pesos do training mix wav2vec2 (Fase 7 gate)
- Manter ou descartar sintético no áudio (Fase 7 gate)
- Threshold final de handover (Fase 10)

---

## O que NÃO fazer

- Não otimizar wav2vec2 para sintético test. Validação só em MELD val real. Sempre.
- Não correr Fase 4 (€30) sem smoke test (Fase 3) ter passado.
- Não retreinar tudo de cada vez se uma fase falhar. Manter a tabela master e parar no gate.
- Não cortar a Fase 8 (listening test) por falta de tempo. Mesmo κ baixo é resultado defensável; ausência de listening test é fraqueza apontável.
- Não actualizar `roberta_meld_only.pt` — preservar para comparação no relatório.
- Não passar tempo a melhorar a demo Gradio em termos visuais. Funcionalidade > estética. 30 min máximo.

---

*Plano criado 2026-05-04 para arrancar 2026-05-05. Manter sincronia com `plano_3_semanas.md` (Session Log) e `projeto_estado_atual.md`.*
