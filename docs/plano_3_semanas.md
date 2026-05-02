# SmartHandover — Plano de Execucao (3 Semanas)

> **Projeto:** Detecao de frustracão em chamadas de suporte para handover automatico
> **Equipa:** Guilherme Lobo Pinto, Pedro Alexandre Silva Gomes, Simao Novais Vieira da Silva
> **Contexto:** Mestrado em IA — Computacao Afetiva, Universidade do Minho
> **Data de inicio:** 2026-04-07 (Segunda-feira)
> **Data de entrega estimada:** 2026-04-25 (Sexta-feira)

---

## 1. Visao Geral da Arquitetura

A abordagem anterior (RoBERTa + Wav2Vec2 multimodal treinado no MELD) e substituida
por um **ensemble de modelos pre-treinados + classificador de texto fine-tuned**, mais robusto
face ao desbalanceamento do dataset (268 amostras de frustracao vs 4709 neutral).

### 1.1 Pipeline Final

```
                        Chamada ao vivo (audio)
                               |
                  +------------+------------+
                  |                         |
                  v                         v
           Whisper (ASR)          SpeechBrain wav2vec2
           audio -> texto          (IEMOCAP, 4 classes)
                  |                         |
                  v                         |
                Texto                       |
                  |                         |
          +-------+--------+               |
          |       |        |               |
          v       v        v               v
      RoBERTa  VADER  GoEmotions    P(angry), P(sad)
      (MELD)  (score) (zero-shot)   P(happy), P(neutral)
          |       |        |               |
          +-------+--------+-------+-------+
                           |
                      Score Fusion
                   (meta-classificador)
                           |
                      +----+----+
                      |         |
                      v         v
                  Frustrado   OK
                  / Raiva?
                      |
                  HANDOVER
```

### 1.2 Componentes do Ensemble

| # | Componente | Tipo | Input | Output | Treino necessario |
|---|-----------|------|-------|--------|-------------------|
| 1 | **VADER** | Lexico (rule-based) | Texto | pos, neg, neu, compound (4 floats) | Nenhum |
| 2 | **GoEmotions** (`j-hartmann/emotion-english-distilroberta-base`) | Modelo pre-treinado | Texto | 6 probabilidades de emocao | Nenhum |
| 3 | **RoBERTa fine-tuned** | Modelo fine-tuned | Texto | 5 probabilidades (classes MELD) | ~1-2h no MELD |
| 4 | **SpeechBrain** (`speechbrain/emotion-recognition-wav2vec2-IEMOCAP`) | Modelo pre-treinado | Audio | 4 probabilidades de emocao | Nenhum |
| 5 | **Whisper** (`openai/whisper-small`) | ASR pre-treinado | Audio | Texto transcrito | Nenhum |
| 6 | **Meta-classificador** | Logistic Regression / MLP | Features dos 4 modelos acima | Decisao final (5 classes) | ~2 min no MELD |

### 1.3 Mapeamento de Emocoes entre Modelos

```
Classe-alvo (SmartHandover)   RoBERTa (MELD)    GoEmotions         SpeechBrain (IEMOCAP)
--------------------------    ---------------   ----------------   ----------------------
anger                         anger             anger              ang
frustration                   frustration       disgust+anger*     ang (proxy)
sadness                       sadness           sadness            sad
neutral                       neutral           neutral            neu
satisfaction                  satisfaction      joy                hap
```
*GoEmotions nao tem "frustration" — usamos disgust+anger como proxy.
*SpeechBrain nao tem "frustration" — usamos angry como proxy.

---

## 2. Pre-requisitos e Dependencias

### 2.1 Dependencias Python a adicionar

```
# Adicionar ao requirements.txt
vaderSentiment>=3.3.2         # Analise de sentimento lexico
speechbrain>=1.0.0            # Modelo pre-treinado IEMOCAP
openai-whisper>=20231117      # ASR (audio -> texto)
gradio>=4.0.0                 # Interface demo (semana 3)
xgboost>=2.0.0                # Alternativa para meta-classificador
```

### 2.2 Modelos a descarregar (primeira execucao)

| Modelo | Tamanho aprox. | Comando / auto-download |
|--------|---------------|------------------------|
| `roberta-base` | ~500 MB | Auto (HuggingFace) |
| `j-hartmann/emotion-english-distilroberta-base` | ~260 MB | Auto (HuggingFace) |
| `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` | ~360 MB | Auto (HuggingFace) |
| `openai/whisper-small` | ~460 MB | Auto (HuggingFace/whisper) |
| Dataset `ajyy/MELD_audio` | ~2 GB | Auto (HuggingFace datasets) |

**Total estimado em disco:** ~3.6 GB de modelos + ~2 GB dataset

### 2.3 Hardware

- **GPU:** RTX 5060 Ti 16GB (Blackwell) — mais do que suficiente para text-only fine-tuning
- **VRAM estimado em pico:**
  - Fine-tune RoBERTa text-only: ~2-3 GB
  - Inferencia ensemble completo: ~3-4 GB
  - (vs. ~6-8 GB da abordagem multimodal anterior)

### 2.4 Estrutura de Ficheiros (Alvo Final)

```
src/
  data/
    load_meld.py              # [MANTER] - dataset loader
    augmentation.py           # [REMOVER ou manter para ablation]
  models/
    text_encoder.py           # [MANTER] - RoBERTa encoder
    audio_encoder.py          # [DEPRECATED] - substituido pelo SpeechBrain
    fusion_model.py           # [REESCREVER] - novo ensemble fusion
    vram_test.py              # [MANTER]
  classifiers/                # [NOVO] pasta
    vader_classifier.py       # [NOVO] wrapper VADER
    goemo_classifier.py       # [NOVO] wrapper GoEmotions
    speechbrain_classifier.py # [NOVO] wrapper SpeechBrain
    whisper_asr.py            # [NOVO] wrapper Whisper
    ensemble.py               # [NOVO] meta-classificador (score fusion)
  training/
    train.py                  # [REESCREVER] - treino text-only RoBERTa
    train_ensemble.py         # [NOVO] - treino meta-classificador
  evaluation/
    metrics.py                # [MANTER] - metricas
    ablation.py               # [NOVO] - estudo de ablacao
  decision/
    handover.py               # [NOVO] - logica de handover com thresholds
  demo/
    app.py                    # [NOVO] - demo Gradio
configs/
  config.yaml                 # [ATUALIZAR] - nova arquitetura
notebooks/
  train.ipynb                 # [ATUALIZAR] - novo pipeline
  analysis.ipynb              # [NOVO] - analise de resultados e graficos
docs/
  fase1_report.md             # [MANTER]
  fase2_report.md             # [MANTER]
  fase3_report.md             # [NOVO] - documentacao desta fase
  plano_3_semanas.md          # Este ficheiro
```

---

## 3. SEMANA 1 — Baselines + Fine-tune Texto (07-11 Abril)

**Objetivo:** Ter 4 classificadores a funcionar e um ensemble baseline.

---

### Dia 1 (Segunda, 07 Abril) — Setup + VADER Baseline

**Manha: Preparacao do ambiente**
- [ ] Instalar novas dependencias (`pip install vaderSentiment speechbrain openai-whisper gradio xgboost`)
- [ ] Atualizar `requirements.txt`
- [ ] Criar a pasta `src/classifiers/`
- [ ] Criar `src/classifiers/__init__.py`

**Tarde: Implementar VADER**
- [ ] Criar `src/classifiers/vader_classifier.py`
  - Classe `VaderClassifier` com metodo `predict(text) -> dict`
  - Retorna: `{"pos": float, "neg": float, "neu": float, "compound": float}`
  - Mapear scores para as 5 classes-alvo:
    - compound < -0.3 → frustration/anger
    - compound entre -0.3 e 0.1 → sadness/neutral
    - compound > 0.3 → satisfaction
  - Nota: este mapeamento e simplista — servira principalmente como feature para o ensemble
- [ ] Correr VADER sobre **todo o MELD** (train + val + test)
- [ ] Guardar resultados em `data/processed/vader_predictions.csv`
- [ ] Calcular metricas com `src/evaluation/metrics.py`
- [ ] Registar resultados: accuracy, weighted_f1, macro_f1, confusion matrix

**Entregavel:** VADER baseline com metricas. Ficheiro `vader_classifier.py` funcional.

**Resultado esperado:** ~30-40% weighted F1 (VADER e fraco em emocoes finas, mas e o chao).

---

### Dia 2 (Terca, 08 Abril) — GoEmotions Zero-Shot

**Manha: Implementar wrapper GoEmotions**
- [ ] Criar `src/classifiers/goemo_classifier.py`
  - Classe `GoEmotionsClassifier`
  - Carregar `j-hartmann/emotion-english-distilroberta-base` com `pipeline("text-classification")`
  - Metodo `predict(text) -> dict` retorna probabilidades por emocao
  - Metodo `predict_batch(texts) -> list[dict]` para eficiencia
  - Mapear as 6 classes GoEmotions para as 5 classes-alvo:
    - anger → anger
    - disgust → frustration (proxy)
    - fear → frustration (proxy)
    - joy → satisfaction
    - neutral → neutral
    - sadness → sadness
    - surprise → neutral (ou drop)

**Tarde: Avaliar no MELD**
- [ ] Correr GoEmotions sobre todo o MELD (train + val + test)
- [ ] Guardar resultados em `data/processed/goemo_predictions.csv`
- [ ] Calcular metricas
- [ ] Comparar com VADER (tabela side-by-side)

**Entregavel:** GoEmotions zero-shot baseline. Tabela comparativa VADER vs GoEmotions.

**Resultado esperado:** ~45-55% weighted F1 (ja e um classificador treinado em emocoes).

---

### Dia 3 (Quarta, 09 Abril) — SpeechBrain Audio + Whisper ASR

**Manha: Implementar wrapper SpeechBrain**
- [ ] Criar `src/classifiers/speechbrain_classifier.py`
  - Classe `SpeechBrainClassifier`
  - Carregar `speechbrain/emotion-recognition-wav2vec2-IEMOCAP`
  - Metodo `predict(audio_array, sr=16000) -> dict`
  - Retorna: `{"ang": float, "hap": float, "sad": float, "neu": float}`
  - Mapear para classes-alvo:
    - ang → anger + frustration (proxy)
    - hap → satisfaction
    - sad → sadness
    - neu → neutral
- [ ] Correr SpeechBrain sobre todo o audio do MELD
- [ ] Guardar resultados em `data/processed/speechbrain_predictions.csv`
- [ ] Calcular metricas

**Tarde: Implementar Whisper ASR**
- [ ] Criar `src/classifiers/whisper_asr.py`
  - Classe `WhisperASR`
  - Carregar `openai/whisper-small` (ou `whisper-base` se VRAM apertada)
  - Metodo `transcribe(audio_array, sr=16000) -> str`
  - Metodo `transcribe_file(path) -> str`
- [ ] Testar Whisper em 50-100 amostras do MELD
- [ ] Comparar texto transcrito vs texto original do MELD
- [ ] Calcular WER (Word Error Rate) aproximado
- [ ] Documentar qualidade da transcricao

**Entregavel:** SpeechBrain audio baseline + Whisper ASR funcional. 3 baselines completos.

**Resultado esperado:** SpeechBrain ~50-60% weighted F1 no audio. Whisper WER < 15% no MELD.

---

### Dia 4-5 (Quinta-Sexta, 10-11 Abril) — Fine-tune RoBERTa Text-Only

**Dia 4 Manha: Preparar pipeline de treino text-only**
- [ ] Criar `src/training/train_text.py` (novo script, limpo)
  - Reutilizar `TextEncoder` existente de `src/models/text_encoder.py`
  - Criar `TextOnlyClassifier(nn.Module)`:
    - TextEncoder (RoBERTa) → [768] → Linear(768, 256) → ReLU → Dropout(0.3) → Linear(256, 5)
  - DataLoader simplificado (so texto + labels, sem audio)
  - `WeightedRandomSampler` para balanceamento (reutilizar logica existente)
  - Treino com:
    - Optimizer: AdamW (lr=2e-5, weight_decay=0.01)
    - Scheduler: linear warmup (10% steps)
    - FP16 (autocast)
    - Gradient accumulation (se necessario, mas com text-only batch_size pode ser 16-32)
    - Early stopping no weighted_f1 (patience=8)
    - Encoder freezing: 2 epocas frozen, depois unfreeze top 4 layers

**Dia 4 Tarde: Treinar**
- [ ] Correr treino (estimativa: 1-2h com batch_size=16, 40 epocas, early stopping)
- [ ] Monitorizar loss e metricas por epoca
- [ ] Guardar melhor modelo em `checkpoints/roberta_text_only.pt`
- [ ] Registar curvas de treino

**Dia 5 Manha: Avaliar e iterar**
- [ ] Avaliar no test set do MELD
- [ ] Guardar predicoes em `data/processed/roberta_predictions.csv`
- [ ] Analisar confusion matrix
- [ ] Se F1 < 55%: tentar learning rate diferente (1e-5, 3e-5) ou mais epocas
- [ ] Se frustration recall < 30%: aumentar peso da classe ou usar focal loss

**Dia 5 Tarde: Primeiro ensemble (texto)**
- [ ] Criar `src/classifiers/ensemble.py` (versao inicial)
  - Combinar: RoBERTa probs + GoEmotions probs + VADER scores
  - Metodo simples: media ponderada com pesos tunaveis
  - Grid search de pesos no validation set:
    ```
    para alpha em [0.3, 0.4, 0.5, 0.6]:
      para beta em [0.2, 0.3, 0.4]:
        gamma = 1 - alpha - beta
        score = alpha * roberta + beta * goemo + gamma * vader_mapped
        avaliar weighted_f1
    ```
- [ ] Registar melhor combinacao de pesos
- [ ] Comparar ensemble texto vs cada modelo individual

**Entregavel Semana 1:**
- 4 classificadores individuais funcionais (VADER, GoEmotions, SpeechBrain, RoBERTa)
- Ensemble de texto (3 modelos)
- Tabela comparativa completa
- Todas as predicoes guardadas em CSV

**Tabela de resultados esperados (fim semana 1):**

| Modelo | Weighted F1 | Frustration Recall | Notas |
|--------|------------|-------------------|-------|
| VADER | ~35% | ~20% | Baseline lexicon |
| GoEmotions | ~50% | ~35% | Zero-shot |
| SpeechBrain (audio) | ~55% | ~40% | Zero-shot audio |
| RoBERTa (text, fine-tuned) | ~62% | ~45% | Fine-tuned no MELD |
| Ensemble texto (3) | ~65% | ~50% | VADER + GoEmo + RoBERTa |

---

## 4. SEMANA 2 — Ensemble Final + Ablation (14-18 Abril)

**Objetivo:** Juntar texto + audio no ensemble final, otimizar, e documentar o impacto de cada componente.

---

### Dia 6 (Segunda, 14 Abril) — Meta-Classificador Completo

**Manha: Construir feature vectors**
- [ ] Para cada amostra do MELD (train/val/test), construir o vector de features:
  ```
  features = [
      roberta_prob_anger,       # 5 floats
      roberta_prob_frustration,
      roberta_prob_sadness,
      roberta_prob_neutral,
      roberta_prob_satisfaction,
      goemo_anger,              # 6 floats (ou mapeadas para 5)
      goemo_disgust,
      goemo_fear,
      goemo_joy,
      goemo_neutral,
      goemo_sadness,
      vader_pos,                # 4 floats
      vader_neg,
      vader_neu,
      vader_compound,
      speechbrain_ang,          # 4 floats
      speechbrain_hap,
      speechbrain_sad,
      speechbrain_neu,
  ]
  # Total: ~19 features por amostra
  ```
- [ ] Guardar em `data/processed/ensemble_features_train.csv`, `_val.csv`, `_test.csv`

**Tarde: Treinar meta-classificador**
- [ ] Testar varias opcoes de meta-classificador:
  - `LogisticRegression(max_iter=1000, class_weight='balanced')`
  - `XGBClassifier(n_estimators=100, max_depth=5)`
  - `MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)`
- [ ] Treinar em train features, validar em val, testar em test
- [ ] Selecionar o melhor meta-classificador
- [ ] Guardar modelo em `checkpoints/meta_classifier.pkl` (joblib/pickle)

**Entregavel:** Meta-classificador treinado. Resultado esperado: ~67-72% weighted F1.

---

### Dia 7 (Terca, 15 Abril) — Ablation Study

**Objetivo:** Provar que cada componente contribui para o resultado final.

- [ ] Criar `src/evaluation/ablation.py`
- [ ] Correr o meta-classificador removendo **um componente de cada vez**:

| Configuracao | Componentes | F1 esperado |
|-------------|------------|-------------|
| Full ensemble | RoBERTa + GoEmo + VADER + SpeechBrain | ~70% |
| Sem VADER | RoBERTa + GoEmo + SpeechBrain | ~68% |
| Sem GoEmotions | RoBERTa + VADER + SpeechBrain | ~65% |
| Sem SpeechBrain | RoBERTa + GoEmo + VADER | ~66% |
| Sem RoBERTa | GoEmo + VADER + SpeechBrain | ~58% |
| So RoBERTa | RoBERTa apenas | ~62% |
| So SpeechBrain | SpeechBrain apenas | ~55% |
| So GoEmotions | GoEmotions apenas | ~50% |
| So VADER | VADER apenas | ~35% |

- [ ] Gerar tabela e grafico de barras (matplotlib)
- [ ] Guardar graficos em `data/processed/ablation_*.png`
- [ ] Escrever analise: qual componente contribui mais? O audio acrescenta valor?

**Entregavel:** Tabela de ablation completa + graficos + analise escrita.

---

### Dia 8 (Quarta, 16 Abril) — Analise de Erros + Otimizacao

**Manha: Error analysis**
- [ ] Identificar os casos onde o ensemble falha:
  - Falsos negativos de frustration (mais perigosos — cliente frustrado nao detectado)
  - Falsos positivos de frustration (incomodo mas menos grave)
  - Confusoes entre classes proximas (anger vs frustration, sadness vs neutral)
- [ ] Para os top-20 erros mais graves, analisar manualmente:
  - O texto e ambiguo?
  - O audio contradiz o texto?
  - Que modelo individual acertou?
- [ ] Documentar padroes de erro

**Tarde: Otimizacao de thresholds**
- [ ] Em vez de argmax, testar **thresholds adaptados por classe**:
  - Para handover, o que importa e: `P(anger) + P(frustration) > threshold`
  - Testar thresholds de 0.3 a 0.7 no validation set
  - Otimizar para **maximizar frustration recall** mantendo precision > 50%
- [ ] Calcular metricas binarias de handover:
  - Precision: dos handovers que o sistema faz, quantos sao corretos?
  - Recall: dos clientes frustrados, quantos o sistema detecta?
  - F1 binario de handover

**Entregavel:** Error analysis documentado. Thresholds otimizados para handover.

---

### Dia 9 (Quinta, 17 Abril) — Pipeline End-to-End com Whisper

**Manha: Integrar Whisper no pipeline**
- [ ] Criar pipeline completo em `src/classifiers/pipeline.py`:
  ```python
  class SmartHandoverPipeline:
      def __init__(self, config):
          self.whisper = WhisperASR(model_size="small")
          self.vader = VaderClassifier()
          self.goemo = GoEmotionsClassifier()
          self.speechbrain = SpeechBrainClassifier()
          self.roberta = load_roberta_model(checkpoint_path)
          self.meta = load_meta_classifier(meta_path)

      def predict_from_audio(self, audio_array, sr=16000):
          # 1. Transcrever audio
          text = self.whisper.transcribe(audio_array, sr)
          # 2. Classificadores de texto
          vader_scores = self.vader.predict(text)
          goemo_scores = self.goemo.predict(text)
          roberta_scores = self.roberta.predict(text)
          # 3. Classificador de audio
          sb_scores = self.speechbrain.predict(audio_array, sr)
          # 4. Meta-classificador
          features = concat(vader_scores, goemo_scores, roberta_scores, sb_scores)
          prediction = self.meta.predict(features)
          return {
              "text": text,
              "emotion": prediction,
              "confidence": max_prob,
              "should_handover": is_negative(prediction, threshold),
              "details": {vader, goemo, roberta, speechbrain scores}
          }
  ```

**Tarde: Testar pipeline end-to-end**
- [ ] Testar com audio do MELD (input: audio raw → output: emocao + decisao)
- [ ] Comparar: texto original vs texto Whisper → impacto na classificacao
- [ ] Medir latencia: quanto tempo demora a processar 1 utterance?
  - Alvo: < 2 segundos por utterance (viavel para "tempo real")
- [ ] Documentar performance end-to-end vs performance com texto original

**Entregavel:** Pipeline completo audio → emocao → handover. Metricas de latencia.

---

### Dia 10 (Sexta, 18 Abril) — Logica de Handover + Documentacao

**Manha: Implementar decisao de handover**
- [ ] Criar `src/decision/handover.py`:
  ```python
  class HandoverDecision:
      def __init__(self, threshold=0.6, window_size=3):
          self.threshold = threshold
          self.window_size = window_size
          self.history = []  # ultimas N predicoes

      def update(self, prediction):
          self.history.append(prediction)
          if len(self.history) > self.window_size:
              self.history.pop(0)

      def should_handover(self):
          # Regra 1: Emocao forte instantanea
          latest = self.history[-1]
          if latest["anger"] + latest["frustration"] > self.threshold:
              return True, "emocao_forte_instantanea"

          # Regra 2: Tendencia negativa na janela
          if len(self.history) >= self.window_size:
              avg_negative = mean([
                  h["anger"] + h["frustration"] + h["sadness"]
                  for h in self.history
              ])
              if avg_negative > self.threshold * 0.7:
                  return True, "tendencia_negativa_sustentada"

          return False, "ok"
  ```

**Tarde: Simulacao de handover no MELD**
- [ ] Para cada conversa no MELD test set, simular o handover:
  - Processar utterances sequencialmente
  - Registar: em que ponto da conversa o sistema teria feito handover?
  - Quantas conversas com frustration real foram apanhadas?
  - Quantos falsos handovers?
- [ ] Gerar metricas de handover a nivel de conversa (nao so utterance)
- [ ] Documentar resultados

**Tarde (continuacao): Documentar semana 2**
- [ ] Escrever `docs/fase3_report.md` com:
  - Descricao da nova arquitetura (ensemble)
  - Justificacao para cada componente
  - Resultados do ablation study
  - Analise de erros
  - Metricas de handover

**Entregavel Semana 2:**
- Ensemble final (texto + audio) com meta-classificador
- Ablation study completo com graficos
- Pipeline end-to-end funcional (audio → emocao → handover)
- Logica de handover com sliding window
- Simulacao de handover no MELD
- Documentacao fase 3

**Tabela de resultados esperados (fim semana 2):**

| Metrica | Valor esperado |
|---------|---------------|
| Weighted F1 (ensemble) | ~68-72% |
| Frustration Recall | ~55-65% |
| Macro F1 | ~55-60% |
| Handover Precision | ~60-70% |
| Handover Recall | ~65-75% |
| Latencia por utterance | < 2 seg |

---

## 5. SEMANA 3 — Demo + Avaliacao Final + Relatorio (21-25 Abril)

**Objetivo:** Demo funcional, avaliacao rigorosa, e documentacao para entrega.

---

### Dia 11 (Segunda, 21 Abril) — Demo Gradio

- [ ] Criar `src/demo/app.py` com interface Gradio:
  ```
  Interface:
  +------------------------------------------+
  |  SmartHandover - Demo                     |
  |                                           |
  |  [Upload audio .wav]  ou  [Gravar audio]  |
  |                                           |
  |  --- Resultado ---                        |
  |  Transcricao: "I've been waiting for..."  |
  |  Emocao: Frustration (72.3%)              |
  |  Handover: SIM                            |
  |                                           |
  |  --- Detalhe dos Modelos ---              |
  |  VADER:      compound = -0.65             |
  |  GoEmotions: anger=0.4, disgust=0.3       |
  |  RoBERTa:    frustration=0.6, anger=0.2   |
  |  SpeechBrain: angry=0.7, neutral=0.1      |
  |                                           |
  |  [Grafico de barras por emocao]           |
  +------------------------------------------+
  ```
- [ ] Funcionalidades:
  - Upload de ficheiro `.wav`
  - Gravacao de audio pelo microfone (Gradio suporta nativamente)
  - Visualizacao de cada modelo individual + decisao final
  - Grafico de barras com probabilidades por emocao
- [ ] Testar com 10+ exemplos variados (frustrado, neutro, satisfeito, etc.)
- [ ] Gravar screenshots para o relatorio

**Entregavel:** Demo Gradio funcional.

---

### Dia 12 (Terca, 22 Abril) — Avaliacao Final Rigorosa

**Manha: Metricas completas no test set**
- [ ] Correr pipeline completo no MELD test set
- [ ] Gerar relatorio final de metricas:
  - Accuracy, Weighted F1, Macro F1
  - Per-class: Precision, Recall, F1, Support
  - Frustration Recall (metrica-chave)
  - Confusion matrix (5x5)
- [ ] Gerar graficos:
  - Confusion matrix heatmap (`data/processed/confusion_matrix_final.png`)
  - Barplot de F1 por classe (`data/processed/per_class_f1.png`)
  - Ablation study barplot (`data/processed/ablation_results.png`)
  - ROC curves para frustration detection (`data/processed/roc_frustration.png`)

**Tarde: Comparacao com abordagem anterior**
- [ ] Se existir resultado do modelo multimodal anterior (RoBERTa + Wav2Vec2):
  - Tabela comparativa lado a lado
  - Argumentar vantagens do ensemble (robustez, menos treino, melhor recall)
- [ ] Se nao existir: comparar ensemble vs melhor modelo individual
- [ ] Calcular melhoria percentual do ensemble sobre cada baseline

**Entregavel:** Todas as metricas finais + graficos + comparacao.

---

### Dia 13 (Quarta, 23 Abril) — Testes com CallCenterEN (Opcional/Bonus)

> Este dia e opcional. Se o tempo permitir, testar generalizacao do modelo em dados reais.

**Manha: Preparar amostra do CallCenterEN**
- [ ] Descarregar subconjunto do dataset CallCenterEN (10-50 transcricoes)
- [ ] Selecionar manualmente exemplos com emocao visivel no texto
- [ ] Anotar manualmente as emocoes (ground truth)

**Tarde: Testar generalizacao**
- [ ] Correr os classificadores de texto (VADER, GoEmotions, RoBERTa) nas transcricoes
- [ ] Comparar com anotacoes manuais
- [ ] Documentar: o modelo generaliza para call center real?
- [ ] Nota: CallCenterEN nao tem audio publico, so testa texto

**Entregavel:** Analise de generalizacao (mesmo que preliminar).

---

### Dia 14-15 (Quinta-Sexta, 24-25 Abril) — Relatorio + Entrega

**Dia 14: Escrita do relatorio**
- [ ] Atualizar `docs/fase3_report.md` (ou relatorio final) com:
  - **Motivacao:** Porquê ensemble em vez de multimodal fim-a-fim
  - **Arquitetura:** Diagrama do pipeline, descricao de cada componente
  - **Implementacao:** Decisoes tecnicas, mapeamentos de emocoes, meta-classificador
  - **Resultados:**
    - Tabela de baselines (VADER, GoEmo, SpeechBrain, RoBERTa)
    - Tabela de ensemble (com e sem audio)
    - Ablation study
    - Analise de erros
    - Metricas de handover
  - **Discussao:**
    - Limitacoes (MELD e dados actuados, nao call center real)
    - O audio acrescentou valor? Quanto?
    - VADER: util ou dispensavel?
    - Generalizacao para dados reais (CallCenterEN se testado)
  - **Trabalho futuro:**
    - Fine-tune SpeechBrain no MELD
    - Treinar em dados de call center reais (necessita labels)
    - Real-time streaming com VAD (Voice Activity Detection)
    - Domain-adaptive pre-training do RoBERTa com CallCenterEN

**Dia 15: Limpeza e entrega**
- [ ] Limpar codigo: remover prints de debug, organizar imports
- [ ] Verificar que todos os scripts correm sem erros
- [ ] Atualizar `README.md` com nova arquitetura
- [ ] Atualizar `config.yaml` com configuracao final
- [ ] Commit final e push
- [ ] Testar demo uma ultima vez
- [ ] Preparar apresentacao se necessario (slides)

**Entregavel Final:** Codigo limpo, relatorio completo, demo funcional.

---

## 6. Riscos e Plano de Contingencia

| Risco | Probabilidade | Impacto | Mitigacao |
|-------|:---:|:---:|-----------|
| RoBERTa fine-tune nao converge | Baixa | Alto | Usar learning rates diferentes; se nao funcionar, usar GoEmotions como classificador principal |
| SpeechBrain nao funciona no MELD (formato de audio incompativel) | Media | Medio | Converter audio para formato esperado; se falhar, usar features prosodicas manuais (librosa) como fallback |
| Whisper transcreve mal o MELD (audio curto, emocional) | Media | Baixo | Whisper e robusto; se WER > 25%, usar texto original do MELD e manter Whisper so para a demo |
| Ensemble nao melhora sobre RoBERTa individual | Baixa | Medio | Testar meta-classificadores diferentes; adicionar features engineered (tamanho texto, presenca de palavras-chave) |
| VRAM insuficiente com 4 modelos em simultaneo | Baixa | Medio | Carregar modelos sequencialmente, nao em paralelo; usar batch inference |
| Falta de tempo na semana 3 | Media | Alto | Priorizar: metricas finais > demo > relatorio > CallCenterEN test (por ordem) |
| SpeechBrain com interface desatualizada | Media | Medio | Verificar documentacao SpeechBrain; se API mudou, adaptar; em ultimo caso, usar `transformers` directamente com o modelo IEMOCAP |

---

## 7. Checklist de Entregaveis por Semana

### Semana 1 (07-11 Abril)
- [ ] `src/classifiers/vader_classifier.py` — funcional e testado
- [ ] `src/classifiers/goemo_classifier.py` — funcional e testado
- [ ] `src/classifiers/speechbrain_classifier.py` — funcional e testado
- [ ] `src/classifiers/whisper_asr.py` — funcional e testado
- [ ] `src/training/train_text.py` — RoBERTa text-only treinado
- [ ] `checkpoints/roberta_text_only.pt` — modelo guardado
- [ ] `src/classifiers/ensemble.py` — versao baseline (media ponderada)
- [ ] `data/processed/vader_predictions.csv`
- [ ] `data/processed/goemo_predictions.csv`
- [ ] `data/processed/speechbrain_predictions.csv`
- [ ] `data/processed/roberta_predictions.csv`
- [ ] Tabela comparativa de todos os baselines

### Semana 2 (14-18 Abril)
- [ ] `data/processed/ensemble_features_{train,val,test}.csv`
- [ ] `checkpoints/meta_classifier.pkl`
- [ ] `src/evaluation/ablation.py` + graficos
- [ ] `src/classifiers/pipeline.py` — pipeline end-to-end
- [ ] `src/decision/handover.py` — logica de handover
- [ ] `data/processed/ablation_results.png`
- [ ] `docs/fase3_report.md` — documentacao parcial
- [ ] Metricas de latencia documentadas

### Semana 3 (21-25 Abril)
- [ ] `src/demo/app.py` — demo Gradio funcional
- [ ] `data/processed/confusion_matrix_final.png`
- [ ] `data/processed/per_class_f1.png`
- [ ] `data/processed/roc_frustration.png`
- [ ] `notebooks/analysis.ipynb` — analise completa
- [ ] `docs/fase3_report.md` — relatorio completo
- [ ] `README.md` atualizado
- [ ] `config.yaml` atualizado
- [ ] Codigo limpo e funcional
- [ ] Commit final

---

## 8. Metricas de Sucesso

O projeto e considerado **bem-sucedido** se:

1. **Weighted F1 >= 65%** no ensemble final (test set MELD)
2. **Frustration Recall >= 50%** (pelo menos metade dos clientes frustrados sao detectados)
3. **Handover Precision >= 55%** (mais de metade dos handovers sao justificados)
4. **Latencia < 3 seg** por utterance (viavel para tempo real)
5. **Ablation study** demonstra que o ensemble supera qualquer modelo individual
6. **Demo funcional** que aceita audio e mostra resultado

O projeto e considerado **excelente** se adicionalmente:

7. Weighted F1 >= 70%
8. Frustration Recall >= 60%
9. Analise de generalizacao com CallCenterEN concluida
10. Comparacao formal com abordagem multimodal anterior documentada

---

## 9. Notas Tecnicas Importantes

### 9.1 Formato de Audio no MELD
O dataset `ajyy/MELD_audio` fornece audio como dicionarios:
```python
sample["audio"] = {
    "array": np.array([...]),   # waveform float32
    "sampling_rate": 16000      # 16kHz
}
```
Tanto o SpeechBrain como o Whisper esperam arrays NumPy a 16kHz — compativel directamente.

### 9.2 Modelos GoEmotions
O modelo `j-hartmann/emotion-english-distilroberta-base` retorna 7 classes:
`anger, disgust, fear, joy, neutral, sadness, surprise`
Precisam de ser mapeadas para as 5 classes-alvo do projeto.

### 9.3 SpeechBrain Interface
```python
from speechbrain.inference import EncoderClassifier
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="tmp_speechbrain"
)
out_prob, score, index, text_lab = classifier.classify_batch(audio_tensor)
# text_lab: ['ang'], ['hap'], ['sad'], ['neu']
```

### 9.4 CallCenterEN (Paper 1 — referencia)
- 91,706 transcricoes reais de call center (sem audio publico)
- Sem labels de emocao — util apenas para domain adaptation ou teste qualitativo
- Licenca CC BY-NC 4.0 — uso academico permitido
- HuggingFace: `AIxBlock/91706-real-world-call-center-scripts-english`

### 9.5 IEMOCAP (base do SpeechBrain)
- ~12h de fala emocional actuada (dialogos entre pares de actores)
- 4 classes: angry, happy, sad, neutral
- O SpeechBrain ja esta treinado — usamos apenas para inferencia

### 9.6 CREMA-D (Fase 4)
- 7 442 clips audio actuados, 91 actores (48 M + 43 F, idades 20-74)
- 12 frases fixas (lexicalmente neutras) -> dataset puramente acustico
- 6 emocoes: anger, disgust, fear, happy, neutral, sad
- 4 niveis de intensidade (LO/MD/HI/XX)
- 16 kHz mono WAV, 16-bit
- Licenca: Open Database License (ODbL) v1.0 — totalmente publico
- Download direto: https://github.com/CheyneyComputerScience/CREMA-D
- Mapeamento default para as 5 classes-alvo:
    ANG -> anger
    DIS -> anger          (proxy)
    FEA -> DROP           (acted fear != customer frustration)
    HAP -> satisfaction
    NEU -> neutral
    SAD -> sadness
- Loader: `src/data/load_cremad.py`, API analoga ao loader MELD
- NAO contribui para treino de texto (frases fixas) — usar apenas como
  augmentacao acustica do componente wav2vec2

---

## 10. FASE 4 — Dataset Augmentation (30 Abril — 8 Maio)

**Motivacao:** O ceiling de frustration recall (~14%) e provadamente um
problema de dados, nao de modelo. SMOTE + class weights + calibration nao
melhoram. As causas raiz:

1. MELD nao tem classe `frustration` — usamos `fear` como proxy mas os
   exemplos sao "please don't hurt me" tipo sitcom, nao frustracao de cliente.
2. Apenas 268 amostras de frustration no train (3% do total).
3. Dominio: sitcom Friends != call center.

A Fase 4 ataca estes 3 pontos com (a) datasets publicos adicionais que
nao requerem licenca academica, e (b) geracao sintetica controlada.

### 10.1 Estrategia em 3 fontes

```
                    +-------------------+
                    |   MELD (atual)    |  12 070 utterances reais
                    +-------------------+
                              |
                              v  uniao no training set
                    +-------------------+
                    |  CREMA-D (publico)|  ~7 442 clips audio
                    |  ODbL license     |  (apenas augmentacao audio)
                    +-------------------+
                              |
                              v
                    +-------------------+
                    | Sintetico (gerado)|  10 000 ou 20 000 amostras
                    |  LLM + TTS         |  (texto + audio)
                    +-------------------+
                              |
                              v
                    Multi-corpus training set
                    com cross-corpus eval
```

### 10.2 Estado actual (2026-05-01)

| Sub-fase | Estado | Notas |
|---|:---:|---|
| Loader CREMA-D | feito | `src/data/load_cremad.py`, smoke tests passam. Falta download dos .wav. |
| Pipeline sintetico - setup | **feito** | 8 modulos em `src/data/synthetic/`, todos resumiveis e testados. |
| Pipeline sintetico - text gen | **feito** | 21 213 / 21 217 amostras geradas via IAEDU/gpt-4o em ~2.5h. |
| Pipeline sintetico - filter | em curso | Adapter judge com Ollama + mistral-small3.1 cableado, falta correr. |
| Pipeline sintetico - audio | pendente | Aguarda fim do filter. |
| Validacao humana | pendente | Listening test (200 amostras) apos audio. |
| Re-treino RoBERTa multi-corpus | pendente | Apos sintetico validado. |
| Re-treino meta + audio | pendente | Final. |
| Ablation cross-corpus + docs | pendente | Final. |

---

### 10.3 Arquitetura final do pipeline sintetico

```
configs/iaedu_accounts.json     (4 contas IAEDU - api_key + channel_id)
.env                            (Ollama endpoint, modelo, etc.)
                |
                v
+-------------------------------+
| src/data/synthetic/           |
|   __init__.py                 |
|   config.py        <- BALANCE_TARGET_PER_CLASS, distribuicao, paths
|   diversity.py     <- 5 eixos: intensity, cause, style, persona, turn
|   _openai_client.py <- 3 pools: text(IAEDU), judge(Ollama), tts(OpenAI)
|   text_normalize.py <- curly -> straight quotes / dashes / ellipsis
|   generate_text.py <- 21k texto via IAEDU, --preview, resume incremental
|   filter_text.py   <- heuristics + LLM-judge (Ollama, modelo independente)
|   generate_audio.py <- TTS via OpenAI gpt-4o-mini-tts, 11 voices, instr.
|   validate.py      <- listening test CLI: sample/annotate/score (kappa)
+-------------------------------+
```

**Pools de provedores** (em `_openai_client.py`):

| Pool | Provider default | Razao | Fallback |
|---|---|---|---|
| `text` | IAEDU (4 contas, multipart streaming) | Free, ja usado no AP | OpenAI direta via `OPENAI_API_KEYS` |
| `judge` | Ollama (`mistral-small3.1`) | Modelo independente do gerador, evita self-preference bias | IAEDU ou OpenAI via `JUDGE_PROVIDER` |
| `tts` | OpenAI direta (`gpt-4o-mini-tts`) | IAEDU nao expoe TTS | - |

---

### 10.4 Distribuicao de classes (calculada para combinado balanceado)

Sintetico gerado para que **MELD + sintetico = 6 000 por classe** (combinado totalmente balanceado, alvo configuravel via `SYNTH_BALANCE_TARGET`):

| Classe | MELD train | Sintetico (gerado) | Combinado | % do gerado |
|---|---:|---:|---:|---:|
| anger | 1 380 | **4 618** | 5 998 | 21.8% |
| frustration | **268** | **5 730** | 5 998 | 27.0% |
| sadness | 683 | 5 317 | 6 000 | 25.1% |
| neutral | 4 709 | 1 291 | 6 000 | 6.1% |
| satisfaction | 1 743 | 4 257 | 6 000 | 20.1% |
| **Total** | 8 783 | **21 213** | 29 996 | |

(faltam 4 amostras = 0.02% por rate-limit transitorio; pode-se recuperar com re-run)

---

### 10.5 Cronograma actualizado (Fase 4)

| Dia | Data | Tarefa | Estado |
|---|---|---|:---:|
| F0 | 29 Abril | Setup pipeline sintetico (8 modulos) | feito |
| F1 | 30 Abril | Adapter IAEDU + 4-account pool | feito |
| F2 | 1 Maio (manha) | Geracao 21k textos via IAEDU | feito |
| F2 | 1 Maio (tarde) | Adapter Ollama judge + cabling | feito |
| F3 | 2 Maio | Filtro completo (heuristics + Ollama judge ~3-4h) | proximo |
| F4 | 3-4 Maio | Smoke test audio (50 clips) + decidir augmentacao canal | pendente |
| F5 | 5 Maio | Geracao audio completa (21k via OpenAI TTS, ~10h) | pendente |
| F6 | 6 Maio | Listening test (200 amostras, 3 anotadores, kappa) | pendente |
| F7 | 7 Maio | Download CREMA-D + cross-corpus baseline | pendente |
| F8 | 8 Maio | Re-treino RoBERTa em 4 condicoes + tabela comparativa | pendente |
| F9 | 9 Maio | Re-treino meta + ablation cross-corpus + graficos | pendente |
| F10 | 10-11 Maio | Atualizar README + relatorio + limpeza | pendente |

---

### 10.6 Resultados esperados

| Configuracao | Test W-F1 | Frust Recall | Comentario |
|---|---:|---:|---|
| Baseline (MELD only) | 65% | 14% | Estado atual (medido) |
| MELD + CREMA-D | 67% | 16% | +diversidade audio |
| MELD + Sintetico | 69% | 45% | +frustration genuina (estimado) |
| MELD + CREMA-D + Sintetico | 71% | 55% | All-in (alvo) |

### 10.7 Riscos e mitigacoes

| Risco | Probabilidade | Mitigacao |
|---|:-:|---|
| Self-preference bias do judge | (resolvido) | Judge usa Ollama mistral-small3.1, modelo diferente do gerador |
| Sintetico colapsa em padroes | Media | 5 eixos forcados por amostra (4608 combinacoes para frustration); judge filtra |
| TTS robotico -> aprendizagem de artefactos | Media | 11 voices distintas, instructions emocionais variadas, opcional telephone-band |
| OOM no Ollama judge | Baixa | Fallback `phi4:latest` (9 GB) ou `JUDGE_PROVIDER=iaedu` |
| Cross-corpus piora | Baixa | Test set fica MELD real; cross-corpus eval explicita |
| kappa < 0.4 no listening test | Media | Iterar prompt; aceitavel se kappa > 0.4 |

### 10.8 Checklist Fase 4

**Infrastructure:**
- [x] `src/data/load_cremad.py`
- [x] `src/data/synthetic/__init__.py`
- [x] `src/data/synthetic/config.py` (BALANCE_TARGET=6000, 5 axes, paths)
- [x] `src/data/synthetic/diversity.py` (5 eixos com per-class rules)
- [x] `src/data/synthetic/_openai_client.py` (3 pools: IAEDU/Ollama/OpenAI)
- [x] `src/data/synthetic/text_normalize.py` (curly -> straight)
- [x] `src/data/synthetic/generate_text.py` (com --preview, resume)
- [x] `src/data/synthetic/filter_text.py` (heuristics + Ollama judge)
- [x] `src/data/synthetic/generate_audio.py` (TTS, 11 voices, instructions)
- [x] `src/data/synthetic/validate.py` (sample/annotate/score)
- [x] `scripts/diagnose_api.py` (text + judge + tts pool tests)
- [x] `scripts/peek_synthetic.py` (inspector)
- [x] `configs/iaedu_accounts.{json,example.json}`
- [x] `.env` + `.env.example` actualizados

**Outputs:**
- [x] `data/synthetic/text.jsonl` (21 213 amostras)
- [ ] `data/synthetic/text_filtered.jsonl` (apos filter)
- [ ] `data/synthetic/text_rejected.jsonl`
- [ ] `data/synthetic/text_judged.jsonl` (cache)
- [ ] `data/synthetic/audio/<label>/*.wav`
- [ ] `data/synthetic/manifest.csv`
- [ ] `data/synthetic/validation_results.csv` (listening test)
- [ ] `data/synthetic/validation_report.json` (kappa)

**Datasets externos:**
- [ ] `data/raw/CREMA-D/AudioWAV/` (~580 MB, download manual)

**Re-treino e ablation:**
- [ ] `checkpoints/roberta_text_only_v2.pt`
- [ ] `checkpoints/meta_classifier_v2.pkl`
- [ ] `data/processed/cross_corpus_results.csv`
- [ ] `data/processed/cross_corpus_results.png`

---

## 11. Session Log

> Diario de sessoes de trabalho. Adicionar uma entrada NO TOPO depois de cada
> sessao significativa, com data, duracao aproximada e resumo das mudancas.
> Itens accionaveis para a sessao seguinte ficam em "**Proximo**".

### Sessao 2026-05-02 (~tarde) — Re-train RoBERTa script (Day F8)

**Feito:**
- Refactor minimalista do `src/training/train_text.py`:
  - Nova `load_synthetic_texts(jsonl_path)` -> (texts, labels).
  - Nova `split_synthetic(texts, labels, val_frac, test_frac, seed)` —
    split estratificado 80/10/10 por classe.
  - `train_model()` aceita agora `train_data`, `val_data`,
    `use_class_weights`, `checkpoint_name` (back-compat preservada).
  - `evaluate_on_test()` aceita `test_data` para cross-corpus eval.
- Criado `scripts/run_dayF8_retrain_roberta.py`:
  - 4 condicoes: `meld_only`, `synth_only`, `combined`, `combined_cw`.
  - Cada uma escreve `checkpoints/roberta_<condition>.pt`.
  - Avalia cada checkpoint em **MELD test (gold)** e **synth test
    (cross-corpus)**.
  - Tabela final em `data/processed/dayF8_results.csv` + manifest JSON.
- Smoke test: imports OK, CLI OK, split estratificado deterministico.
  Synth filtered: 19 280 -> train 15 424, val 1 928, test 1 928.

**Proximo (utilizador):**
- [ ] `python scripts/run_dayF8_retrain_roberta.py` (~30-45 min na RTX
  5060 Ti, treina as 4 condicoes em sequencia).
- [ ] Inspeccionar `data/processed/dayF8_results.csv` para a tabela
  comparativa.
- [ ] Apos confirmacao do ganho em frust recall, avancar para
  geracao de audio (B) e CREMA-D (C).

### Sessao 2026-05-02 — Filtragem completa do sintetico

**Feito:**
- Recuperadas as 4 amostras em falta (`text.jsonl` agora tem 21 217).
- `filter_text.py` correu com Ollama mistral-small3.1 como judge.
- Resultados:
  - 18 754 / 21 217 amostras tinham smart punctuation (normalizadas).
  - Stage A (heuristics): 386 rejeitadas (banned phrases + axis leak).
  - Stage B (judge): 1 551 rejeitadas (1545 low_judge_score + 4
    unparseable + 2 timeouts).
  - **Kept: 19 280 (90.9%)** -> `data/synthetic/text_filtered.jsonl`.
- `data/synthetic/text_judged.jsonl` cache populado (resume-safe).

**Distribuicao kept por classe:**
| Classe | Kept | % retido |
|---|---:|---:|
| anger | 4 494 | 97.3% |
| frustration | 5 621 | 98.1% |
| sadness | 3 673 | 69.1% |
| neutral | 1 235 | 95.7% |
| satisfaction | 4 257 | 100.0% |

Sadness teve filtragem mais forte - vale a pena investigar (judge mais
exigente vs gerador menos convincente nesta classe).

**Criado:**
- `docs/projeto_estado_atual.md` (briefing para o colega que vai
  escrever o relatorio). Cobre contexto, arquitectura, datasets,
  resultados medidos, limitacoes, decisoes, bibliografia.

**Proximo:**
- [ ] Re-treinar RoBERTa com MELD + sintetico filtrado (4 condicoes:
  MELD-only, synth-only, MELD+synth, MELD+synth+class_weights).
- [ ] Criar `scripts/run_dayF8_retrain_roberta.py`.
- [ ] Cross-corpus eval (treino MELD vs treino synth, test em ambos).
- [ ] Apos: gerar audio sintetico (~10h, OPENAI_TTS_API_KEY necessario).
- [ ] Em paralelo: download CREMA-D.

### Sessao 2026-05-01 (~tarde) — Judge com Ollama (mistral-small3.1)

**Feito:**
- Adicionado `JUDGE_PROVIDER` (default `ollama`), `OLLAMA_BASE_URL`,
  `SYNTH_JUDGE_MODEL` em `config.py`.
- `_openai_client.py`: novo `_build_judge_pool()` com tres provedores
  (ollama / iaedu / openai). Concurrency reduzida automaticamente para 2
  quando provider e Ollama (uma so GPU).
- `filter_text.py` agora pede `get_pool("judge")` em vez de `get_pool("text")`.
  Argumento academico: judge != gerador -> sem self-preference bias.
- `.env.example` e `scripts/diagnose_api.py` actualizados (3 stages: text,
  judge, tts).
- Smoke test offline passa (pool judge constroi sem erros).

**Decisao:** modelo recomendado `mistral-small3.1:latest` (~14 GB Q4) por
caber na RTX 5060 Ti 16 GB com folga. Alternativas no `.env.example`:
gemma3:27b (top, mas borderline), phi4 (rapido), qwen3:14b, llama3.2.

**Proximo:**
- [ ] `ollama pull mistral-small3.1`
- [ ] `python scripts/diagnose_api.py` → confirmar `Judge backend: OK`
- [ ] `python -m src.data.synthetic.filter_text` (~3-4h)
- [ ] Inspeccionar `text_rejected.jsonl` para ver razoes de filtragem
- [ ] Decidir se vale a pena correr o gerador de novo para apanhar as 4
  amostras em falta (0.02%, provavelmente nao)

### Sessao 2026-05-01 (~manha) — Geracao de texto sintetico completa

**Feito:**
- IAEDU adapter implementado em `_openai_client.py` (`IAEduClient` com
  interface OpenAI-compativel, multipart/form-data, NDJSON streaming,
  filtragem de UUIDs e mensagens de processing, deteccao 429).
- `configs/iaedu_accounts.{json,example.json}` com 4 contas (api_key +
  channel_id pairs). Ficheiro real esta gitignored.
- `scripts/diagnose_api.py` com 3 stages: env load, IAEDU per-account
  test, TTS smoke test.
- Pequeno bug fix: `peek_synthetic.py` deixou de usar mapping local e
  agora importa de `src/data/synthetic/text_normalize.py` (single source
  of truth). UTF-8 forcado no stdout para mostrar curly chars no Windows.
- `text_normalize.py`: utilitario que converte aspas/dashes/ellipsis
  smart para ASCII. Aplicado em "Stage 0" do `filter_text.py` para
  garantir consistencia com o tokenizer do MELD.

**Resultados:**
- Run completo `generate_text.py` correu em ~2.5h via IAEDU 4 contas.
- 21 213 / 21 217 amostras geradas (faltam 4, rate limits transitorios).
- 0 duplicados, 0 textos vazios, 0 demasiado curtos/longos.
- Word count: min=9, max=40, mediana=19, media=19.2.
- Coverage diversity axes: 5/5 intensities, 12 styles, 16 causes,
  6/6 personas, 4/4 turns.

**Decisao:**
- Distribuicao alterada para "balanceado pos-MELD" (alvo 6000 por classe
  no combinado). Sintetico gera o defice exacto: 4620 anger, 5732
  frustration, 5317 sadness, 1291 neutral, 4257 satisfaction.

### Sessao 2026-04-30 — Setup do pipeline sintetico

**Feito:**
- 8 modulos novos em `src/data/synthetic/`.
- `BALANCE_TARGET_PER_CLASS=6000` configuravel por env var.
- 5 eixos de diversidade com per-class rules (`diversity.py`).
- `--preview N` em `generate_text.py` para auditar prompts sem custo.
- Resume incremental por amostra (JSONL append-only).
- Pool round-robin com cooldown automatico em rate-limit.

**Proximo:**
- [x] Confirmar URL real do IAEDU (descoberto no extend.py do AP)
- [x] Adapter IAEDU
- [x] Run completo de texto

### Sessao 2026-04-29 — Reorganizacao do projecto

**Feito:**
- `run_day*.py` movidos para `scripts/` (raiz limpa).
- `src/training/train.py` (multimodal deprecated, importacoes partidas)
  arquivado em `src/training/_deprecated/train_multimodal.py` com README.
- READMEs adicionados em `scripts/` e `_deprecated/`.
- `.env` criado a partir do template, `.gitignore` actualizado.
- 26/26 modulos importam apos reorg.

**Feito antes (Sessao 2026-04-27):**
- Diagnostico do problema de dados: frustration=fear no MELD, ceiling 14%.
- SMOTE + isotonic calibration testados, sem efeito (`train_meta_balanced.py`).
- Comparacao 3 estrategias de fusao: late fusion bate score fusion (+1.2pp).
- Demo Gradio (`src/demo/app.py`).
- README reescrito.
- Loader CREMA-D (`src/data/load_cremad.py`).
- Plano original Fase 4.

---

*Documento criado em 2026-04-04. Ultima atualizacao: 2026-05-02 (filtragem completa).*
