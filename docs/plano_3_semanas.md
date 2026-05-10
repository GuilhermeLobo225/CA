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

### Sessao 2026-05-07 (~depois de F15) — Fase 10b: Ensemble v4 (frustration-recall focused)

**Motivacao:** v3 nao melhorou face a v2 (audio fine-tuned colapsado em
4 sb_* perdeu informacao). Ensemble v4 corrige isso e adiciona:

- **20 features** (vs 19): mantem 5 nativas do wav2vec2 fine-tuned
  (audio_anger, audio_frust, audio_sad, audio_neut, audio_satis) em vez
  de colapsar em 4 sb_*.
- **Selecao por frust_recall** (vs val W-F1): max val frust_recall com
  W-F1 floor >= 0.55. Da prioridade ao que importa para handover.
- **SMOTE oversampling** (5 classes -> 4709 cada).
- **Isotonic calibration** sobre o melhor candidate. Trade-off: baixa
  argmax frust recall mas uniforma probs -> melhor para threshold-based.
- **Threshold tuning multi-grid**: 4 weight pairs (w_anger, w_frust)
  x 51 thresholds (0.20-0.70 step 0.01) = 204 combinacoes.

**Resultados v4:**

Meta-classifier (test):
- LR_balanced: W-F1=65.21%, FrustR=28.0%
- XGB_balanced: W-F1=64.14%, FrustR=22.0%
- MLP_balanced: W-F1=62.27%, FrustR=22.0%
- Selecionado: LR_balanced (max val frust_recall).
- Apos isotonic calibration: W-F1=65.32%, FrustR=16% (calibration baixa
  argmax frust recall mas beneficia handover threshold).

Handover (versao v1/v2/v3 vs v4):
| Versao | thr | Prec | Rec | F1 | FrustR |
|---|---:|---:|---:|---:|---:|
| v1/v2 | 0.300 | 0.538 | 0.469 | 0.501 | 0.240 |
| v3 | 0.300 | 0.538 | 0.469 | 0.501 | 0.240 |
| **v4** | **0.200** (w_a=0.6, w_f=0.4) | 0.504 | **0.646** | **0.566** | **0.460** |

**+15.4 pp F1**, **+17.7 pp recall**, **+22 pp frust recall**.
~2x melhoria face a v1/v2/v3.

**Outputs:**
- `data/processed/ensemble_features_*_v4.csv` (20-dim)
- `checkpoints/meta_classifier_v4.pkl` (LR raw)
- `checkpoints/meta_classifier_v4_calibrated.pkl` (LR + isotonic)
- `data/processed/meta_classifier_v4_summary.json`
- `data/processed/dayF16_threshold_sweep_v4.csv` (todas as combinacoes
  weights x thresholds, val + test)
- `configs/handover_threshold_v4.json` (chosen weights + threshold)
- `data/processed/dayF16_summary.json` (consolidado v1->v4)

**Headline para o relatorio:**
> "Optimised threshold tuning over a binary anger/frustration score
> (w_anger=0.6, w_frust=0.4, t=0.20) raised handover frustration
> recall from 24% to 46% while maintaining handover precision above
> 50%. The ensemble v4 detects ~2x more frustrated callers than
> v1/v2/v3."

**Decisao sobre Fase 7-replay (SpecAugment / wav2vec2-base):**
- v4 ja atinge 46% frust handover recall com componentes existentes.
- Re-treino com SpecAugment teria ganho marginal estimado (+2-5pp) ao
  custo de 2-3h GPU + risco.
- **Skipped** para o relatorio. Documentado como trabalho futuro.

### Sessao 2026-05-07 (~depois de F14) — Fase 10: Ensemble v3 final (scripts)

**Decisao:** **Opcao A** (re-tune ensemble com componentes v2):
- Texto: `roberta_combined.pt` (fine-tune com sintetico, escolhido na Fase 1)
- Audio: `wav2vec2_finetuned.pt` (with-synth, escolhido na Fase 7)

Opcao B (manter frozen audio) descartada — defesa academica e mais
forte com componentes coerentes ("usamos a versao multi-corpus em
todos os componentes do ensemble" > "audio frozen, texto fine-tuned").

**Implementado:**
- Em `src/training/train_audio.py`:
  - `predict_records(checkpoint, records)` -> List[Dict] com 5-class probs
    (anger, frust, sad, neut, satis). Permite extrair predicoes para
    o ensemble sem re-treinar.
- `scripts/run_dayF15_ensemble_v3.py` orquestra 4 passos:
  1. **Predict wav2vec2 fine-tuned** sobre todo o MELD (train+val+test).
     Cache em `data/processed/audio_v3_predictions.csv`. ~10-15 min.
  2. **Build features v3**: substitui colunas `sb_*` em
     `ensemble_features_*_v2.csv` pelas novas. Mapeamento 5-class
     -> 4-class IEMOCAP-style:
       sb_ang = p_anger + p_frust   (negative-energy consolidado)
       sb_hap = p_satis
       sb_sad = p_sad
       sb_neu = p_neut
     Schema 19-dim mantido para compatibilidade total com meta-classifier.
  3. **Train meta v3 + fusion v3**: invoca os mesmos scripts modulares
     (`ensemble_trainer.py`, `fusion_strategies.py`) com `suffix=_v3`.
  4. **Threshold sweep v3**: replica o Day-8 mas sobre meta v3.
     Output: `configs/handover_threshold_v3.json` +
     `dayF15_threshold_sweep.csv`.

**Outputs esperados (apos correr):**
- `data/processed/audio_v3_predictions.csv` (~12 070 rows, drop-in
  para speechbrain_predictions.csv)
- `data/processed/ensemble_features_*_v3.csv`
- `checkpoints/meta_classifier_v3.pkl` + summary JSON
- `data/processed/fusion_comparison_v3.{csv,png}` +
  `fusion_late_weights_v3.json`
- `data/processed/dayF15_threshold_sweep.csv` +
  `configs/handover_threshold_v3.json`
- `data/processed/dayF15_summary.json` (consolidado v1 -> v2 -> v3)

**Tempo estimado:** ~20 min (10-15 min predict + ~5 min restante).
Custo: zero.

**Smoke test offline:** imports OK, CLI OK, features_v2 prontos,
checkpoint wav2vec2 fine-tuned (~1.2 GB) presente.

**Proximo:** correr `python scripts/run_dayF15_ensemble_v3.py`.

### Sessao 2026-05-07 (~depois de F13) — Fase 9: Cross-corpus matrix

**Estrategia:** com pipeline end-to-end completo, gerar **matriz central
do relatorio** sem re-treinar nada. Le os CSVs ja produzidos:
- `dayF8_results.csv` -> texto (3 condicoes x 2 testes)
- `dayF13_results.csv` -> audio (2 condicoes x 3 testes)
- `cremad_baseline_summary.json` -> frozen audio anchor

**Implementado:** `scripts/run_dayF14_cross_corpus_matrix.py`:
- `build_text_matrix()` + `build_audio_matrix()` (pivot pandas)
- 2 heatmaps lado-a-lado por metrica (W-F1, FrustR)
- CSV unificado para o relatorio

**Resultados Fase 9 — TEXTO (W-F1):**
| Train | MELD test | Synth test |
|---|---:|---:|
| MELD only | 0.606 | 0.466 |
| Synth only | 0.393 | 0.837 |
| MELD + Synth | **0.654** | **0.824** |

**Resultados Fase 9 — AUDIO (W-F1):**
| Train | MELD test | CREMA-D test | Synth test |
|---|---:|---:|---:|
| Frozen (IEMOCAP) | n/a | 0.536 | n/a |
| MELD + CREMA-D | 0.093 | 0.365 | 0.148 |
| MELD + CREMA-D + Synth | **0.453** | **0.537** | 0.284 |

**Achados centrais para o relatorio:**

1. **Texto: MELD+Synth ganha nos 2 dominios** (+4.7pp em MELD test,
   +35.8pp em Synth test face ao MELD-only). Argumento de tese **provado
   quantitativamente**.

2. **MELD frustration e fake**: `meld_only` em synth_test = 0.4% frust
   recall. Modelo treinado em fear-as-frustration nao consegue
   reconhecer frustration genuina de call-center. Cross-corpus expoe
   isto.

3. **Audio: sintetico FOI NECESSARIO** para convergir. Sem ele, head
   re-iniciado colapsa em frustration (W-F1=9% em MELD test). Com
   sintetico, recupera paridade com frozen (53.6% vs 53.7% em CREMA-D).
   Defesa academica forte: "the synthetic data was not just an
   augmentation - it was a stabiliser without which fine-tune did not
   converge under MELD's class imbalance."

4. **Audio fine-tune NAO ultrapassa frozen em CREMA-D** — paridade
   apenas. Dois argumentos honestos: (a) wav2vec2-large com 316M
   parametros e demasiado para o nosso volume de dados; (b) tecto da
   arquitectura/dados, motiva trabalho futuro com modelos mais
   pequenos (CNN+MLP sobre mel-spectrograms ou wav2vec2-base 95M).

**Outputs:**
- `data/processed/cross_corpus_matrix.csv` (12 linhas)
- `data/processed/cross_corpus_matrix.png` (heatmaps W-F1, central no relatorio)
- `data/processed/cross_corpus_matrix_frust.png` (heatmaps frust recall)

**Proximo:** Fase 10 — re-tune final do ensemble com componentes v2.

### Sessao 2026-05-07 (~noite, ainda mais tarde) — Fase 7: Fine-tune wav2vec2 multi-corpus (scripts)

**Decisao:** Opcao B (treinar 2 variantes para ablation):
- `wav2vec2_finetuned_no_synth.pt`: MELD train + CREMA-D train.
- `wav2vec2_finetuned.pt`: MELD train + CREMA-D train + synth phone-band
  (com sample weight 0.5 para manter batch maioritariamente real).

Justificativo: o relatorio precisa de ablation directa do impacto do
sintetico. Sem isto, nao temos prova quantitativa do valor da Fase 4.

**Implementado:**
- `src/training/train_audio.py`:
  - `AudioRecord` (path, label, source, weight) + `AudioDataset`.
  - Loaders por corpus: `load_meld_audio_records()`,
    `load_cremad_audio_records()`, `load_synth_phone_records()`.
  - MELD audio cache em `data/cache/meld_audio/<split>/*.wav`
    (HF dataset entrega arrays, nao paths -> materializamos uma vez).
  - `build_model()` carrega `superb/wav2vec2-large-superb-er` com
    `num_labels=5` e `ignore_mismatched_sizes=True` (descarta head
    IEMOCAP de 4 classes).
  - `freeze_encoder()` + `unfreeze_top_layers(n)` = mesma estrategia
    do RoBERTa (frozen 2 epocs -> unfreeze top 4).
  - `train_audio_model()`: AdamW com 2 grupos (head lr 1e-4, encoder
    lr 5e-6), warmup linear, FP16 autocast, gradient accumulation,
    early stopping na MELD val W-F1.
  - `evaluate_checkpoint()`: avaliar qualquer .pt em qualquer
    record set. Usado para os 3 test sets (meld/cremad/synth).
- `scripts/run_dayF13_finetune_wav2vec2.py`:
  - Opcao --variant {no-synth, with-synth, both} (default: both).
  - Carrega records uma vez, treina cada variante em sequencia.
  - Validacao SEMPRE em MELD val (nunca em sintetico, evita overfit
    a artefactos TTS).
  - Test em 3 splits separados: MELD test, CREMA-D test, synth_test
    (10% do sintetico, separado com seed=42).
  - Output: `dayF13_results.csv` (variant x test_set) + per-variant
    history JSON.
  - Gate automatico: PASS se with-synth meld_test_W-F1 >= 0.448 (frozen
    baseline) AND |synth_test - meld_test| < 15 pp.

**Smoke test:**
- Imports OK, CLI OK.
- CREMA-D test records: 476.
- Synth phone records: 9 235 (todos disponiveis).
- AudioRecord/Dataset/loaders prontos.

**Tempo estimado real:** ~2-4h total (2 variantes em sequencia) na
RTX 5060 Ti, depende de early stopping. wav2vec2-large = 316M params,
batch 4 + accumulation 4 = effective batch 16, FP16 ja optimizado.

**Proximo:** correr `python scripts/run_dayF13_finetune_wav2vec2.py`
quando puder ficar a correr ~3-4h.

### Sessao 2026-05-07 (~noite, mais tarde) — Fase 6: CREMA-D + script baseline

**Feito:**
- Download CREMA-D via sparse-checkout (Opcao A, ~600 MB):
  ```
  git init + sparse-checkout AudioWAV/ + VideoDemographics.csv
  ```
- Loader `src/data/load_cremad.py` indexou 6 171 clips (FEA dropped
  por mapeamento default).
- Distribuicao: anger=2 542 (41.2%), sadness=1 271, neutral=1 087,
  satisfaction=1 271. Speaker-disjoint split: train=5 355, val=340,
  test=476.
- Criado `scripts/run_dayF12_cremad_baseline.py`:
  - Itera CREMA-D (split configuravel: all/train/val/test).
  - Aplica `superb/wav2vec2-large-superb-er` frozen.
  - Mapeia IEMOCAP (ang/hap/sad/neu) -> 5 classes-alvo.
  - Output: `cremad_speechbrain_predictions.csv` + `cremad_baseline_summary.json`.
  - Console: weighted/macro F1, per-class P/R/F1, confusion matrix.
  - Gate: PASS se W-F1 >= 50%, BORDERLINE 35-50%, FAIL < 35%.

**Para que serve:**
- Estabelece **baseline do componente audio** (numero a bater na Fase 7).
- Mostra a generalizacao IEMOCAP -> CREMA-D (mesmo espaco 4-class,
  actores diferentes, gravacao diferente).
- Output sera secao no relatorio: "frozen wav2vec2-IEMOCAP atinge X%
  W-F1 cross-corpus em CREMA-D, validando aprendizagem de emocao
  generica e nao apenas distribuicao IEMOCAP".

**Proximo:** correr `python scripts/run_dayF12_cremad_baseline.py`.

**Resultados Fase 6 (after running):**
- 6 171 clips processados em ~2:48 min (36.6 clips/s na RTX 5060 Ti).
- Bug encontrado: `compute_metrics` falhava com `ValueError` quando o
  test set nao tinha todas as 5 classes (CREMA-D nao tem frustration).
  Fix: passar `labels=list(range(n))` ao sklearn em
  `classification_report` / `f1_score` / `confusion_matrix`.
  Predicoes ja estavam guardadas em CSV — re-corri so as metricas.
- **Headline metrics (todas as 6 171):**
  - accuracy = 53.1%
  - **weighted F1 = 53.6%** -> Gate PASS (>= 50%)
  - macro F1 = 41.5%
- **Per-class:**
  - anger:  P=71.9%  R=50.9%  F1=59.6%  (n=2542)
  - sadness: P=53.4%  R=61.8%  F1=57.3%  (n=1271)
  - neutral: P=49.4%  R=39.8%  F1=44.1%  (n=1087)
  - satisfaction: P=37.7%  R=60.0%  F1=46.3%  (n=1271)
- **Achados-chave para o relatorio:**
  - Sadness F1 = 57.3% em CREMA-D vs P(sad)=0 em TODOS os 50 smoke
    sinteticos. **Prova** que a sub-deteccao de sadness no TTS e um
    fenomeno de **distribuicao out-of-domain**, nao defeito do
    classificador.
  - Confusao maior: 728 amostras anger -> satisfaction. Modelo IEMOCAP
    confunde "alta energia" entre as duas classes (anger forte vs
    satisfaction enthusiastic). Sitio onde fine-tune deve ganhar.
  - Anger precision 71.9% (alto) mas recall 50.9% (medio). Satisfaction
    inverso: precision 37.7%, recall 60%.

### Sessao 2026-05-07 (~noite) — Fase 5: Degradacao telefonica (scripts)

**Decisao:** evitar download MUSAN (6 GB) para um trabalho de cadeira;
implementar **ruido sintetico** (pink + 60 Hz hum) que e bem definido
e suficientemente realista para a tarefa.

**Implementado:**
- `src/data/synthetic/degrade_audio.py`:
  - `degrade_to_phone(wav, sr)` -> 16k -> 8k mu-law -> bandpass 300-3400 Hz
    -> 16k. Codec G.711 standard de PSTN/VoIP.
  - `add_synthetic_noise(wav, snr_db, seed)` -> pink (1/f via FFT) +
    60 Hz hum + harmonica 120 Hz, mixado a SNR exacto.
  - `add_recorded_noise(wav, noise_dir, snr_db, seed)` -> drop-in para
    MUSAN se o utilizador quiser realismo extra mais tarde. Fallback
    automatico para sintetico se a pasta estiver vazia.
  - `apply_gain_jitter(wav, db, seed)` -> +/- N dB.
  - `degrade_pipeline(wav, sr_in, snr_db_range, ...)` -> orquestra os 3
    passos. Determinístico com seed.
- `scripts/run_dayF11_degrade_audio.py`:
  - Itera `data/synthetic/audio/<label>/*.wav` (clean).
  - Aplica pipeline -> `data/synthetic/audio_phone/<label>/*.wav`.
  - Manifest CSV com snr_db, noise_source, codec, bandpass_hz por clip.
  - Resume incremental (skip se ja existir + size > 0).
  - SNR aleatorio por clip em [snr_low, snr_high] (default 15-25 dB).
  - Per-clip seed = stable hash do audio_id -> re-runs sao
    bit-exact iguais.

**Smoke offline:**
- Pipeline determinístico OK (mesma seed -> mesmos samples bit-exact).
- SNR alvo 20 dB -> medido 20.0 dB (FFT pink shaping correcto).
- Fallback recorded -> synthetic se noise_dir invalido.

**Defesa academica:** "Aplicamos codec G.711 mu-law + bandpass
300-3400 Hz (banda telefonica nominal) + ruido pink/60 Hz hum mixado
a SNR aleatorio em [15, 25] dB + jitter de ganho +/-6 dB. Esta
degradacao deterministica simula condicoes PSTN/VoIP tipicas e
previne que o classificador acustico aprenda artefactos do TTS de
estudio em vez de sinal emocional."

**Custo:** zero. Processamento local na RTX 5060 Ti, ~30-60 min para
9 235 clips.

**Resultados Fase 5:**
- ok=9 235  fail=0  elapsed=96s (96.4 clips/s, muito rapido com synthetic noise)
- avg duration phone: 7.69s
- Output: `data/synthetic/audio_phone/<label>/<id>.wav` (9 235 ficheiros)
- Manifest: `data/synthetic/audio_phone_manifest.csv` (9 235 linhas)
- Tudo bit-exact reprodutível com seed=42

**Proximo:** Fase 6 — download CREMA-D + cross-corpus baseline.

### Sessao 2026-05-07 (~tarde) — Fase 4: Audio sintetico completo (Opcao C)

**Decisao:** com base no smoke test, custo realista revisto para
~€0.003/clip (vs estimativa inicial €0.0015). Para um trabalho de
cadeira opcional, run completo (19 280 clips, ~€58) e excessivo.
Adicionada flag `--max-per-class N` ao `generate_audio.py` que faz
**subsampling estratificado deterministico** com seed (resume-safe).

**3 opcoes de cost control oferecidas:**
| Opcao | --max-per-class | Total | Custo | Para |
|---|---:|---:|---:|---|
| A (minimo viavel) | 800 | 4 000 | ~€12 | Provar conceito |
| B (recomendada) | 1 500 | 7 500 | ~€23 | Equilibrio custo/qualidade |
| **C (escolhida)** | **2 000** | **~9 235** | **~€28** | **+frustration data** |
| D (run completo) | (sem flag) | 19 280 | ~€58 | Maximalismo |

**Resultados Opcao C:**
- 9 235 clips gerados (2000+2000+2000+1235+2000)
- 0 falhas em 1h 1min 43s (rate ~2.5 clips/s com 4 workers)
- avg clip duration: 7.69s
- custo real: ~€27.71 (alinhado com a estimativa)
- Output: `data/synthetic/audio/<label>/<id>.wav` + `manifest.csv`
- Voices: 11 disponiveis no gpt-4o-mini-tts (round-robin no
  `_next_voice` global, distinto do smoke onde so usamos 3).
- Instructions usadas: condicionais por classe (versao final do smoke
  test 4: `_INTENSITY_TONE_BY_LABEL`, `_STYLE_HINTS_BY_LABEL`,
  `_CLOSING_BY_LABEL`).

**Decisoes registadas:**
- `neutral` ficou a 1 235 amostras (todas as filtered disponiveis;
  gerador inicial parou em 1 291 e o filtro tirou ~56 mais).
- Distribuicao final no dataset combinado (MELD + sintetico):
  - anger: 1 380 + 2 000 = 3 380
  - frustration: 268 + 2 000 = 2 268 (8.5x o MELD)
  - sadness: 683 + 2 000 = 2 683
  - neutral: 4 709 + 1 235 = 5 944
  - satisfaction: 1 743 + 2 000 = 3 743

**Proximo:** Fase 5 — degradacao telefonica + ruido (~2-3h local,
sem custo).

### Sessao 2026-05-08 — Fase 3 concluida (4 iteracoes de smoke TTS)

**Smoke test 1** (`SMOKE_VOICES = ["nova", "onyx", "shimmer"]`, instruction
universal):
- median P(ang): anger=0.753, frust=0.704, neut=0.344, sad=0.303, satis=0.477
- Feedback subjectivo: `onyx` soa pausada/chata; `nova` e `shimmer` boas.

**Smoke test 2** (substituido `onyx` por `coral`; instruction reforcada
com "vividly emotionally expressive, AVOID long pauses"):
- median P(ang): anger=0.918, frust=0.836, neut=0.781, sad=0.624, satis=0.762
- **Problema**: instruction universal puxou TODAS as classes para "agitated".
  Sadness/neutral colapsam em P(ang) alto.

**Smoke test 3** (instruction CONDICIONAL por classe via
`_CLOSING_BY_LABEL`; voices mantidas `nova/coral/shimmer`):
- median P(ang): anger=0.942, frust=0.954, neut=0.658, sad=0.923, satis=0.856
- Sadness ainda saturada porque `_INTENSITY_TONE` e `_STYLE_HINTS`
  (universais) contradiziam o closing.
- Feedback: utilizador pediu para adicionar voice masculina.

**Smoke test 4** (versao final):
- `SMOKE_VOICES = ["nova", "verse", "shimmer"]` (1 masculina + 2 femininas).
- `_INTENSITY_TONE_BY_LABEL` (override para sadness/satisfaction/neutral).
- `_STYLE_HINTS_BY_LABEL` (override para sadness — "polite_but_firm with
  steel underneath" virou "controlled and quietly resigned").
- median P(ang): anger=0.961, frust=0.873, neut=0.807, sad=0.768, satis=0.753
- Feedback: sadness agora soa diferente de anger (subjectivo). Sadness
  desceu de 0.923 -> 0.768 no frozen, mas P(sad) continua 0.000.

**Achado importante para o relatorio:**
- `superb/wav2vec2-large-superb-er` (frozen) tem **bias massivo para
  classe `ang`** em distribuicao sintetica. P(sad)=0.000 em 50/50 amostras,
  mesmo as anotadas como sadness. Isto **NAO e falha do TTS** — e bias
  do modelo IEMOCAP-trained quando aplicado out-of-domain.
- Este achado justifica o fine-tuning multi-corpus (Fase 7) com CREMA-D
  (sadness humana actuada com sinal prosodico forte) + sintetico phone-band.

**Decisoes registadas:**
- Voice mix final para o run completo: **`nova`, `verse`, `shimmer`**
  (round-robin determinístico por classe).
- Instructions com class-conditional closing + intensity_by_label +
  style_by_label.
- **Fase 3 listening test formal: SKIPPED**. Justificativa: feedback
  subjectivo do utilizador (10 amostras ouvidas) confirma qualidade
  perceptiva; gate tecnico (anger/frust > 0.25 P(ang)) passa folgado.
  No relatorio sera registado como decisao de risco assumido + apontado
  como trabalho futuro.
- Sadness sera aceite como classe mais fraca; o foco do projecto e
  handover (anger + frustration), onde o sinal e robusto.

**Custo real do smoke** (4 iteracoes × ~50 amostras): ~€0.60 total.

**Proximo:** Fase 4 — geracao audio completa (com cost control, ver
seccao seguinte).

### Sessao 2026-05-07 — Day F10: Smoke TTS scripts (Fase 3)

**Feito:**
- Decisao Fase 2 (registada): meta_classifier_v2 = MLP, W-F1 65.78%
  (vs v1 64.74%), Frust-R 22% (vs v1 14%). Late fusion v2 = 67.35%
  (vs 65.98%). Gate FAIL tecnico (precisava 68%/45%) mas ganho
  consistente.
- Caminho critico: passar para componente audio. Texto saturou no
  MELD test devido a fear-as-frustration ceiling.
- Criado `scripts/run_dayF10_smoke_tts.py`:
  - Estratifica 50 amostras de `text_filtered.jsonl` (10 por classe).
  - 3 voices fixas: `nova`, `onyx`, `shimmer` (registos distintos).
  - Round-robin determinístico de voice por classe.
  - Reutiliza `build_instruction()` de `generate_audio.py` (mesma
    instrução emocional do run completo).
  - Salva 50 .wav em `data/synthetic/smoke_audio/<label>/<id>_<voice>.wav`
    + `data/synthetic/smoke_manifest.csv`.
  - Resume incremental (skip se ja existe no manifest).
  - Custo estimado: ~€0.10-0.20 (50 calls).
- Criado `scripts/run_dayF10_smoke_eval.py`:
  - Carrega `superb/wav2vec2-large-superb-er` (frozen, mesma do ensemble).
  - Predicoes IEMOCAP 4-class (ang/hap/sad/neu) por clip.
  - Output: `data/synthetic/smoke_frozen_eval.csv`.
  - Sumario por classe: median P(ang/hap/sad/neu) + match rate vs
    label esperado.
- Smoke test offline: 50 amostras estratificadas OK, voice round-robin
  balanceado (4/3/3 por classe).

**Proximo (utilizador):**
- [ ] `python scripts/run_dayF10_smoke_tts.py` (~5-8 min, ~€0.15)
- [ ] `python scripts/run_dayF10_smoke_eval.py` (~30s na RTX 5060 Ti)
- [ ] Listening test manual (~30 min):
  - `python -m src.data.synthetic.validate sample --manifest data/synthetic/smoke_manifest.csv --sheet data/synthetic/smoke_listening.csv --n 50`
  - `python -m src.data.synthetic.validate annotate --sheet data/synthetic/smoke_listening.csv --name <yourname>`
- [ ] Gate Fase 3:
  - PASS: ≥35/50 correct_class no listening + median P(ang) > 0.25
    para anger e frustration no frozen eval.
  - FAIL refusal: instruções suavizadas em `generate_audio.py`.
  - FAIL flat: implementar voice × persona mapping (plan §3.2).

### Sessao 2026-05-05 — Day F9: Meta-classifier v2 + Fusion v2 (script)

**Feito:**
- Refactor de `src/classifiers/ensemble_trainer.py`:
  - `argparse` com `--roberta-checkpoint`, `--output-suffix`, `--force-regenerate`.
  - `_roberta_csv_path(split, suffix)` permite ter cache `_v2` em paralelo
    com o original.
  - `build_ensemble_features(roberta_suffix, roberta_checkpoint, force_regenerate)`.
  - `save_feature_csvs(df, suffix)` -> `ensemble_features_<split><suffix>.csv`.
  - `meta_classifier{suffix}.pkl` + `meta_classifier{suffix}_summary.json`.
  - Manifest agora regista o checkpoint do RoBERTa usado.
- Refactor de `src/classifiers/fusion_strategies.py`:
  - `argparse` com `--features-suffix`, `--meta-checkpoint`, `--output-suffix`.
  - `score_fusion_predict(df, meta_path)` aceita override.
  - Outputs com suffix: `fusion_comparison<suffix>.{csv,png}`,
    `fusion_late_weights<suffix>.json`.
- Novo `scripts/run_dayF9_meta_v2.py`:
  - Orquestra os dois passos (meta v2 + fusion v2).
  - Default usa `roberta_combined.pt` da Fase 1.
  - `--skip-meta` / `--skip-fusion` para granularidade.
  - Resumo final com checklist dos 11 outputs esperados.

**Outputs esperados (apos correr):**
- `data/processed/roberta_*_predictions_v2.csv` (re-gerados com novo .pt)
- `data/processed/ensemble_features_*_v2.csv`
- `checkpoints/meta_classifier_v2.pkl`
- `data/processed/meta_classifier_v2_summary.json`
- `data/processed/fusion_comparison_v2.{csv,png}`
- `data/processed/fusion_late_weights_v2.json`

**Decisao Fase 1 (registada):**
- `combined` escolhido como checkpoint principal (W-F1 65.4% no MELD,
  82.4% no synth - melhor compromisso). `combined_cw` descartado
  (frust recall 0% no MELD insustentavel).
- Frust recall do MELD test (18%) e enganador: o gold standard tem
  fear-rotulado-como-frustration. **Cross-corpus eval mostra que
  `meld_only` recolhe 0.4% no synth_test, prova que o MELD frustration
  e fake.** Documentar isto no relatorio.

**Proximo (utilizador):**
- [ ] `python scripts/run_dayF9_meta_v2.py` (~1h: ~30 min predicoes
  RoBERTa em CPU/GPU + ~5 min meta + ~30s fusion).
- [ ] Comparar `meta_classifier_v2_summary.json` com
  `meta_classifier_summary.json`.
- [ ] Comparar `fusion_comparison_v2.csv` com `fusion_comparison.csv`.
- [ ] Gate: PASS se W-F1 >= 68% e Frust-R >= 45% em qualquer
  estrategia de fusao.
- [ ] Avancar para Fase 3 (Smoke TTS) se gate PASS, ou diagnosticar
  pesos do late fusion se W-F1 cair.

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
