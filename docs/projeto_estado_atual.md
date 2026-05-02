# SmartHandover — Estado Atual do Projeto

> Documento de briefing para escrita do relatório.
> Estado em **2026-05-02**.
> NÃO é o relatório — é o material de partida para quem o vai escrever.

---

## 1. Contexto e objectivo

| Item | Valor |
|---|---|
| Unidade curricular | Computação Afetiva |
| Mestrado | Mestrado em Inteligência Artificial — Universidade do Minho |
| Ano letivo | 2025/2026 |
| Equipa | Guilherme Lobo Pinto (PG60225), Pedro Alexandre Silva Gomes (PG60289), Simão Novais Vieira da Silva (PG60393) |
| Data limite | 22 Maio 2026 (sem penalização até 25 Maio); apresentação semana 27 Maio |

**Problema:** detecção em (quasi-)tempo-real de **frustração** em chamadas
de apoio ao cliente, para automatizar o **handover** (encaminhamento para
agente humano) quando a emoção negativa atinge um limiar.

**Pergunta de investigação:** Pode um ensemble de modelos de fala e
linguagem pré-treinados, complementado com geração sintética
condicionada, atingir recall de frustração suficientemente alto para
justificar handover automático em call center?

---

## 2. Modelo de emoção adoptado

Abordagem **híbrida** com 3 modelos teóricos complementares (declarado no
`README.md` desde a Fase 1):

| Modelo | Uso no projeto |
|---|---|
| **Ekman (discreto)** | Classificação primária em 5 classes: `anger`, `frustration`, `sadness`, `neutral`, `satisfaction`. |
| **Russell (dimensional)** | A "soma" `P(anger)+P(frustration)` é tratada como proxy unidimensional de valência negativa, usada para o threshold de handover. |
| **Plutchik (gradação)** | A geração sintética usa 5 níveis de intensidade (`1=mild`...`5=extreme`) para cobrir o espectro de cada emoção. |

A escolha discreta foi por compatibilidade com os datasets disponíveis
(MELD, IEMOCAP, CREMA-D usam categorias). A dimensão de Russell é usada
indirectamente; Plutchik aparece principalmente nos dados sintéticos
(distribuição balanceada por intensidade).

---

## 3. Arquitetura técnica

```
                       audio (16 kHz mono)
                              |
           +------------------+------------------+
           |                                     |
           v                                     v
     Whisper (ASR)                    wav2vec2 - IEMOCAP
     openai/whisper-small        superb/wav2vec2-large-superb-er
           |                              P(ang/hap/sad/neu)
           v                                     |
         texto                                   |
           |                                     |
   +-------+-------+-------+                     |
   |       |       |       |                     |
   v       v       v       v                     |
 VADER   GoEmo  RoBERTa  ...                     |
 (4)     (6)    (5, fine-tuned MELD)             |
   |       |       |                             |
   +-------+-------+----------+------------------+
                              |
                       feature 19-D
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
       Score-fusion      Late fusion    Decision fusion
       MLP (Day 6)      (avg pesado)   (majority vote)
              |
              v
       5-class emotion + handover trigger
       (P(anger)+P(frustration) > t)
```

**Componentes activos** (ver código em `src/classifiers/` e `src/training/`):

| Componente | Modelo / técnica | Treinado por nós? |
|---|---|:---:|
| ASR | `openai/whisper-small` | não |
| Sentimento lexical | VADER | não (rule-based) |
| Emoção zero-shot (texto) | `j-hartmann/emotion-english-distilroberta-base` (treinado em GoEmotions) | não |
| Classificador texto fine-tuned | `roberta-base` + cabeça 256→5 (MELD) | **sim** |
| Emoção (áudio) | `superb/wav2vec2-large-superb-er` (treinado em IEMOCAP) | não |
| Meta-classificador | MLP / Logistic Regression / XGBoost | **sim** |
| Decisão de handover | regras com sliding window (3 utterances) | regras |

A escolha do `superb/wav2vec2-large-superb-er` em vez do `speechbrain/
emotion-recognition-wav2vec2-IEMOCAP` original do plano deveu-se a
incompatibilidades de symlinks no Windows e da dependência `k2`.

---

## 4. Datasets

### 4.1 MELD (real, fonte principal)
- `ajyy/MELD_audio` (HuggingFace), 16 kHz mono FLAC + texto + emoção.
- ~12 070 utterances (train 8 783, val 958, test 2 329).
- **Origem: série Friends** — sitcom americano. Emoção atuada, com
  laugh-track e música de fundo.
- Não tem classe `frustration`. Mapeamos:
  - `anger`+`disgust` → anger
  - `fear` → frustration (proxy questionável; ver §9)
  - `joy` → satisfaction
  - `sadness` → sadness
  - `neutral` → neutral
  - `surprise` → DROP (valência ambígua)

### 4.2 CREMA-D (real, audio enrichment, **não descarregado ainda**)
- 7 442 clips, 91 atores, 12 frases fixas.
- Licença ODbL pública, download direto sem formulário.
- 6 emoções (anger, disgust, fear, happy, neutral, sad) em 4 intensidades.
- **Loader implementado** (`src/data/load_cremad.py`), smoke tests passam.
- Será usado **só como augmentação acústica** (não para texto, dado que
  todas as 7 442 amostras são uma de 12 frases fixas).
- Mapeamento default: ANG→anger, DIS→anger (proxy), FEA→DROP,
  HAP→satisfaction, NEU→neutral, SAD→sadness.

### 4.3 Dataset sintético (gerado por nós, foco da Fase 4)
- **21 217** utterances geradas via **IAEDU/gpt-4o** com pool de 4 contas.
- Cada amostra condicionada num ponto único do espaço de diversidade de 5 eixos:
  - `intensity` (1-5)
  - `cause` (8 valores negativos / 5 neutros / 4 positivos)
  - `style` (12 estilos)
  - `persona` (6 perfis demográficos)
  - `turn_position` (4 posições na chamada)
  - **Total combinações distintas:** 2 880 (anger), 4 608 (frust), 2 304 (sad), 720 (neut), 1 152 (sat).
- **Após filter (heurísticas + Ollama mistral-small3.1 judge):** **19 280**
  amostras kept (**90.9%**); 1 937 rejeitadas.
- **Pendente:** geração de áudio via `gpt-4o-mini-tts` com 11 voices,
  instruções emocionais por amostra. Estimativa €30, ~10h.

### 4.4 Distribuição de classes (combinado)

| Classe | MELD train | Sintético kept | Combinado | % combinado |
|---|---:|---:|---:|---:|
| anger | 1 380 | 4 494 | 5 874 | 22% |
| frustration | **268** | **5 621** | **5 889** | 22% |
| sadness | 683 | 3 673 | 4 356 | 16% |
| neutral | 4 709 | 1 235 | 5 944 | 22% |
| satisfaction | 1 743 | 4 257 | 6 000 | 22% |
| **Total** | 8 783 | 19 280 | 28 063 | |

`frustration` salta de **3.1% no MELD** para **21% no combinado** (>21x mais sinal).

---

## 5. Pipeline sintético — implementação

Localizado em `src/data/synthetic/` (8 módulos, todos resumíveis e testados):

| Módulo | Função |
|---|---|
| `config.py` | Source of truth: distribuição alvo, modelos, paths, concorrência |
| `diversity.py` | Sampler dos 5 eixos com regras por classe (e.g. `neutral` não tem intensity 5) |
| `_openai_client.py` | 3 pools: `text` (IAEDU), `judge` (Ollama), `tts` (OpenAI). Round-robin com cooldown automático. |
| `text_normalize.py` | Normalização curly→straight (apóstrofos, dashes, ellipsis) para alinhar com MELD |
| `generate_text.py` | Loop principal: amostra ponto da diversidade → chama LLM → grava JSONL incrementalmente. Resumível, com `--preview` (audita prompts sem gastar API). |
| `filter_text.py` | 3 stages: normalização → heurísticas (length, banned phrases, axis leak) → LLM judge (score 1-5 + intensity match). |
| `generate_audio.py` | TTS com `gpt-4o-mini-tts`. Voices em round-robin. Instruções emocionais geradas por amostra a partir dos eixos de diversidade. |
| `validate.py` | CLI listening test: `sample` (estratificado), `annotate` (interactivo), `score` (Cohen's kappa). |

**Provedores:**
- **Texto:** IAEDU (4 contas, free, custom multipart streaming protocol — mesmo
  que foi usado no projecto AP do mestrado).
- **Judge:** Ollama local com `mistral-small3.1:latest` (~14 GB Q4) — modelo
  **independente** do gerador para evitar self-preference bias documentado em
  Zheng et al. ("Judging LLM-as-a-Judge", 2023) e Panickssery et al. ("LLM
  Evaluators Recognize and Favor Their Own Generations", 2024).
- **TTS:** OpenAI direto (IAEDU não expõe TTS); custo ~€30 para 19 280 amostras.

---

## 6. Estado dos componentes

### Texto (treino próprio)
- `src/training/train_text.py` — RoBERTa fine-tuned no MELD train.
  - Class weights inversos, `WeightedRandomSampler`.
  - FP16, AdamW, linear warmup, gradient clipping.
  - Encoder freeze 2 épocas, depois unfreeze top 4 layers.
  - Early stopping em `weighted_f1` (patience=8).
  - Best model em `checkpoints/roberta_text_only.pt` (~500 MB).
- `src/training/train_meta_balanced.py` — variante SMOTE + isotonic
  calibration para tentar melhorar frust recall (sem efeito significativo —
  ver §7).

### Áudio (todos zero-shot)
- VADER, GoEmotions, SpeechBrain (wav2vec2), Whisper — ver tabela §3.

### Ensemble e fusão
- `src/classifiers/ensemble_trainer.py` — score-fusion via MLP/LR/XGBoost
  sobre features 19-dim concatenadas. Default: **MLP**.
- `src/classifiers/fusion_strategies.py` — comparação 3-way:
  score / late (média ponderada) / decision (majority vote).

### Decisão de handover
- `src/decision/handover.py` — sliding window de 3 utterances + 2 regras:
  1. Instantânea: `P(anger)+P(frustration) > t`.
  2. Tendência: média rolling `> 0.7 * t`.
- `src/decision/simulate_handover.py` — simulação a nível de conversa no MELD test.

### Avaliação
- `src/evaluation/metrics.py` — accuracy, weighted/macro F1, frust recall, confusion matrix.
- `src/evaluation/ablation.py` — leave-one-out por componente.
- `src/evaluation/error_analysis.py` — top-20 erros mais críticos +
  threshold sweep para handover.

### Demo
- `src/demo/app.py` — Gradio: upload .wav ou microfone → pipeline →
  emoção + handover + breakdown por modelo + plot probabilidades.

---

## 7. Resultados actuais (medidos)

### 7.1 Baselines individuais (MELD test)

| Modelo | Test W-F1 | Macro F1 | Frust Recall |
|---|---:|---:|---:|
| VADER | 39.7% | 24.7% | 16.0% |
| GoEmotions zero-shot | 55.9% | 42.2% | 38.0% |
| Audio (wav2vec2-IEMOCAP) | 44.8% | 30.3% | 0.0% |
| **RoBERTa fine-tuned (MELD)** | **64.4%** | 46.7% | 14.0% |

### 7.2 Ensemble (3 estratégias de fusão)

| Estratégia | Test W-F1 | Frust Recall | Comentário |
|---|---:|---:|---|
| Score fusion (MLP, Day 6) | 64.7% | 14.0% | meta sobre 19-dim feature |
| **Late fusion (avg pesado)** | **66.0%** | 14.0% | **vencedor — pesos: roberta=0.5, goemo=0.5, sb=0.5, vader=0.3** |
| Decision fusion (majority) | 63.1% | 12.0% | tie-break por roberta |

### 7.3 Ablation study

Ordenado por queda de W-F1 quando o componente é removido:

| Removido | Δ W-F1 | Δ Frust Recall |
|---|---:|---:|
| RoBERTa | -8.2 pp | -8.0 pp |
| GoEmotions | -3.5 pp | -2.0 pp |
| SpeechBrain | -1.4 pp | +0.5 pp |
| VADER | -0.3 pp | +0.0 pp |

**RoBERTa é o componente dominante**, GoEmotions é second-best.
SpeechBrain (áudio) e VADER contribuem marginalmente.

### 7.4 Tentativas de melhorar frust recall (todas falharam)

| Tentativa | Frust Recall | Resultado |
|---|---:|---|
| Class weights (CrossEntropy) | 14% | sem efeito |
| `WeightedRandomSampler` | 14% | sem efeito |
| Focal loss | 14% | sem efeito |
| SMOTE oversampling (1380→4709) | 14% | sem efeito |
| LR balanced + SMOTE | 14% | sem efeito |
| XGBoost balanced + SMOTE | 12% | piorou |
| MLP + SMOTE + isotonic calibration | 14% | sem efeito |

**Conclusão:** o ceiling é dos dados (MELD não tem frustração real), não
do algoritmo. Esta foi a evidência decisiva para iniciar a Fase 4
(geração sintética).

### 7.5 Handover (a nível de conversa, threshold tuned)

| Métrica | Valor |
|---|---:|
| Recall de conversas com emoção negativa | **88.8%** |
| Precisão de conversas accionadas | 75.5% |
| Latência média até trigger | -0.5 utterances (frequentemente apanha **antes** da turn negativa) |

A nível de **utterance** o frust recall é ~14%; a nível de **conversa** o
recall sobe para 88.8% porque o sliding window agrega sinal ao longo do
tempo. Esta é uma das histórias mais fortes para o relatório.

---

## 8. Pipeline sintético — resultados (2026-05-02)

### 8.1 Geração de texto
- **21 217 amostras** geradas via IAEDU/gpt-4o em ~2.5h (4 contas
  paralelas).
- Throughput médio: 2.6 utterances/s.
- Falhas: 4 amostras iniciais por rate-limit transitório, recuperadas em
  re-run (0.02% loss → 0%).
- 0 duplicados, 0 textos vazios, word count 9-40 (mediana 19, média 19.2).

### 8.2 Filtragem
| Stage | Output | % retido |
|---|---:|---:|
| Stage 0 — normalização (curly→straight) | 18 754 / 21 217 modificados | — |
| Stage A — heurísticas | 20 831 / 21 217 | 98.2% |
| Stage B — Ollama judge | 19 280 / 20 831 | 92.6% |
| **Total final** | **19 280 / 21 217** | **90.9%** |

### 8.3 Razões de rejeição (top)

| Motivo | Contagem |
|---|---:|
| `low_judge_score(2)` (judge deu 2/5) | 1 518 |
| `banned_phrase('I'm sorry, but')` | 274 |
| `banned_phrase('I cannot')` | 72 |
| `axis_leak` (style/persona escapou no texto) | 32 |
| `low_judge_score(1)` (judge deu 1/5) | 27 |
| `banned_phrase('the customer')` (narrador) | 8 |
| `judge_unparseable` (mistral devolveu JSON malformado) | 4 |
| `judge_api_error:APITimeoutError` | 2 |

A maior parte das rejeições são **legítimas**: o gpt-4o ocasionalmente
fugia para "I cannot/I'm sorry, but" (modo refusal) em vez de gerar a
utterance pedida — capturado pelas heurísticas. O judge rejeitou ~7.4%
adicionais por baixa qualidade ou intensidade desalinhada.

### 8.4 Distribuição final por classe (sintético kept)

| Classe | Gerado | Kept | % retido |
|---|---:|---:|---:|
| anger | 4 620 | 4 494 | 97.3% |
| frustration | 5 732 | 5 621 | 98.1% |
| **sadness** | 5 317 | **3 673** | **69.1%** |
| neutral | 1 291 | 1 235 | 95.7% |
| satisfaction | 4 257 | 4 257 | 100.0% |

**Sadness teve 31% de rejeição** — vale a pena investigar se o judge é
mais exigente com tristeza, ou se o gerador produziu sadness pouco
convincente. Pode ser uma secção de discussão no relatório.

---

## 9. Limitações conhecidas (importante para honestidade do relatório)

1. **MELD não tem frustração real.** Mapeamos `fear` → frustration, mas
   o `fear` no MELD é maioritariamente susto/medo de sitcom ("please
   don't hurt me"), não frustração de cliente. Esta é a causa-raiz do
   ceiling de 14% no frust recall do baseline.

2. **Sitcom vs call center: domain shift forte.** MELD é Friends; o
   problema-alvo é call center. Emoção atuada, presence de laugh-track,
   conversação multi-character vs cliente-agente.

3. **Dataset sintético tem provenance única (gpt-4o).** Mesmo com 5 eixos
   forçados, há viés do modelo gerador. O judge independente (Ollama
   mistral-small3.1) mitiga em parte mas não elimina.

4. **TTS sintético soa distinguível de fala humana.** Mesmo o gpt-4o-mini-tts
   com instruções emocionais não atinge a prosódia natural de actores.
   Discutir como limitação do componente acústico.

5. **CREMA-D tem só 12 frases fixas.** Útil só para o componente
   acústico; não dá sinal para texto.

6. **Validação humana ainda não foi feita.** Cohen's kappa entre os 3
   anotadores no listening test ainda está por medir.

7. **Cross-corpus eval (MELD ↔ sintético) ainda não foi feito.** É a
   métrica que valida se o sintético generaliza.

8. **Componente acústico contribui pouco** (~1.4 pp W-F1). É uma fraqueza
   conhecida; CREMA-D + síntese de áudio podem mudar isto, mas só será
   sabido após Fase 4 completa.

9. **Whisper transcreve mal o MELD.** Background music + laugh-track
   degradam a ASR. Para o relatório, métricas devem distinguir entre
   "com texto MELD ground-truth" vs "com texto Whisper".

---

## 10. Considerações éticas (RGPD, viés, chilling effect)

1. **Voz é dado biométrico.** Qualquer deployment exigiria consentimento
   informado no início da chamada e política de retenção clara.
   Modelos e features extraídas devem ser apagáveis a pedido (Direito
   ao Esquecimento, RGPD).

2. **Viés demográfico.** MELD é US English, sitcom (atores
   maioritariamente brancos, idade 25-35). IEMOCAP é similar. CREMA-D
   tem mais diversidade (91 atores, idades 20-74) mas continua a ser
   English americano. **Não testámos em sotaques não-nativos,
   code-switching, ou outras línguas.**

3. **Chilling effect.** Um sistema que escala "frustrados"
   automaticamente pode também desincentivar agentes a gravarem
   queixas legítimas. O threshold deve ser transparente para o
   utilizador.

4. **Falsos positivos vs negativos.** Falsos positivos (escalar
   utilizadores não-frustrados) inconvenenciam; falsos negativos
   (perder frustração real) são o failure mode que o sistema deve
   resolver. O threshold sweep optimiza explicitamente para
   recall sob restrição de precisão (ver `error_analysis.py`).

---

## 11. Trabalho ainda por fazer

Em ordem de prioridade:

| # | Tarefa | Tempo | Output |
|---|---|---|---|
| 1 | Re-treinar RoBERTa em **MELD train + sintético filtrado** (com cross-corpus eval) — script `scripts/run_dayF8_retrain_roberta.py` pronto | ~30-45 min | `checkpoints/roberta_<condition>.pt` (4 condições) + `data/processed/dayF8_results.csv` |
| 2 | Re-construir features 19-dim, re-treinar meta-classifier | ~5 min | `meta_classifier_v2.pkl` |
| 3 | Gerar **áudio sintético** via `gpt-4o-mini-tts` (~€30, ~10h) | ~10h em background | `data/synthetic/audio/<label>/*.wav` + manifest |
| 4 | Download CREMA-D + indexar + cross-corpus baseline do componente áudio | ~1h | `cremad_speechbrain_predictions.csv` |
| 5 | Listening test: 200 amostras, 3 anotadores, Cohen's kappa | ~2-3h dos 3 | `validation_report.json` |
| 6 | Fine-tune wav2vec2 em MELD audio + CREMA-D + sintético áudio | ~3h GPU | `wav2vec2_finetuned.pt` |
| 7 | Ablation cross-corpus (matriz 4×4): MELD/CREMA-D/sintético/all | ~1h | `cross_corpus_results.{csv,png}` |
| 8 | (Opcional) Atualizar demo Gradio com novos modelos | ~30 min | `app.py` revisto |
| 9 | Atualizar README.md, requirements.txt, configs/ para o estado final | ~30 min | docs alinhados |

---

## 12. Estrutura do repositório

```
CA/
├── README.md                                    # geral, atualizado para arquitetura ensemble
├── .env.example                                 # template (chaves não vão para o git)
├── .gitignore
├── requirements.txt                             # openai>=1.50, imbalanced-learn, gradio, etc.
│
├── scripts/                                     # entry points executáveis
│   ├── run_dayN_*.py                            # Day 1-5 (Week 1)
│   ├── diagnose_api.py                          # testa pools text/judge/tts
│   └── peek_synthetic.py                        # inspector do .jsonl sintético
│
├── checkpoints/                                 # (gitignored)
│   ├── roberta_text_only.pt                     # fine-tuned MELD
│   ├── meta_classifier.pkl                      # MLP score-fusion
│   └── meta_classifier_balanced.pkl             # SMOTE + isotonic (não melhora)
│
├── configs/
│   ├── config.yaml                              # config geral
│   ├── handover_threshold.json                  # threshold optimal para handover
│   └── iaedu_accounts.example.json              # template (real é gitignored)
│
├── data/
│   ├── raw/                                     # MELD streamed (HF), CREMA-D pendente
│   ├── processed/                               # predicões dos modelos, ablations, fusões
│   └── synthetic/
│       ├── text.jsonl                           # 21 217 amostras geradas
│       ├── text_filtered.jsonl                  # 19 280 kept após filter
│       ├── text_rejected.jsonl                  # 1 937 com razão
│       ├── text_judged.jsonl                    # cache do judge
│       └── audio/<label>/*.wav                  # PENDENTE
│
├── docs/
│   ├── fase1_report.md                          # relatório fase 1 (já entregue)
│   ├── fase2_report.md                          # relatório fase 2 (já entregue)
│   ├── plano_3_semanas.md                       # plano executivo, com session log
│   └── projeto_estado_atual.md                  # ESTE ficheiro
│
├── notebooks/                                   # 5 notebooks Day 1-5 (Week 1 deliverables)
│
└── src/
    ├── data/
    │   ├── load_meld.py
    │   ├── load_cremad.py
    │   └── synthetic/                           # 8 módulos: ver §5
    ├── models/
    │   └── text_encoder.py                      # wrapper RoBERTa
    ├── classifiers/                             # 8 módulos: vader/goemo/sb/whisper/
    │                                            # ensemble/ensemble_trainer/fusion/pipeline
    ├── training/
    │   ├── train_text.py                        # fine-tune RoBERTa
    │   ├── train_meta_balanced.py               # SMOTE + calibration
    │   └── _deprecated/                         # train_multimodal.py (Phase-2 abandonado)
    ├── evaluation/                              # metrics, ablation, error_analysis
    ├── decision/                                # handover.py, simulate_handover.py
    └── demo/
        └── app.py                               # Gradio
```

---

## 13. Métricas de sucesso (definição original do plano)

> **Bem-sucedido** se cumprir 1-6:

1. ✅ Weighted F1 ≥ 65% no ensemble final → **66.0% (late fusion)** — feito
2. ❌ Frust Recall ≥ 50% → **14% (utterance) / 88.8% (conversa)** — pendente sintético
3. ✅ Handover Precision ≥ 55% → **75.5%** — feito
4. ✅ Latência < 3 seg por utterance → ~1.5-2 seg — feito
5. ✅ Ablation prova superioridade do ensemble → demonstrado
6. ✅ Demo funcional → `src/demo/app.py` operacional

> **Excelente** se cumprir adicionalmente 7-10:

7. ❓ Weighted F1 ≥ 70% → ainda não medido (após Fase 4)
8. ❓ Frust Recall ≥ 60% → meta da Fase 4
9. ❌ Generalização CallCenterEN → não feito (fora do scope final)
10. ❌ Comparação com abordagem multimodal anterior → não feito

A Fase 4 está focada em fechar **#2** e **#7-#8**. Tudo o resto está
provado.

---

## 14. Decisões importantes e justificações

Lista das escolhas técnicas mais relevantes (úteis na secção "Decisões"
do relatório):

| Decisão | Alternativa rejeitada | Justificação |
|---|---|---|
| Ensemble vs multimodal end-to-end | RoBERTa + wav2vec2 fim-a-fim com fusão | Robustez face ao desbalanceamento (268 frustration), menos VRAM, mais explicável |
| Late fusion vs score fusion | Score fusion como default | +1.2 pp W-F1 (medido), interpretabilidade |
| MELD `fear` → frustration | Drop fear ou outro mapping | Mantém >0 amostras de frustração (alternativa daria 0). **Limitação assumida no relatório.** |
| IAEDU para texto sintético | OpenAI direta | Custo zero (free para alunos), 4 contas paralelas, mesma stack do projecto AP do mestrado |
| Ollama judge (≠ gerador) | gpt-4o judge | Evita self-preference bias documentado em 2 papers (Zheng 2023, Panickssery 2024) |
| `mistral-small3.1` para judge | gemma3:27b | Cabe folgado em 16 GB Q4, JSON output fiável |
| OpenAI TTS (não Coqui local) | Coqui XTTS-v2, ElevenLabs | gpt-4o-mini-tts aceita instruções emocionais por amostra (Coqui não); ElevenLabs mais caro |
| 5 eixos de diversidade no prompt | Single prompt repetido | Evita colapso de modos; combinatorial space >2.8k por classe |
| BALANCE_TARGET = 6 000 | Distribuição uniforme 4 000 por classe | Combinado MELD+sintético fica balanceado a 6k; preserva sinal real do MELD |
| Threshold sweep (val) optimizado para frust recall sob precision floor | argmax | Recall é a métrica crítica para handover; o threshold actual é 0.30 |

---

## 15. Referências bibliográficas relevantes

(para o relatório — completar à medida que escrevem)

- **MELD:** Poria et al. (2018), "MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations".
- **CREMA-D:** Cao et al. (2014), "CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset", IEEE TAC.
- **IEMOCAP:** Busso et al. (2008), "IEMOCAP: Interactive Emotional Dyadic Motion Capture Database".
- **wav2vec2:** Baevski et al. (2020), "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations".
- **RoBERTa:** Liu et al. (2019).
- **GoEmotions:** Demszky et al. (2020).
- **VADER:** Hutto & Gilbert (2014).
- **Whisper:** Radford et al. (2022), "Robust Speech Recognition via Large-Scale Weak Supervision".
- **LLM-as-Judge:** Zheng et al. (2023), "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena".
- **Self-preference bias:** Panickssery et al. (2024), "LLM Evaluators Recognize and Favor Their Own Generations".
- **Ekman:** Ekman (1992), "An argument for basic emotions".
- **Russell:** Russell (1980), "A circumplex model of affect".
- **Plutchik:** Plutchik (1980), "Emotion: A psychoevolutionary synthesis".
- **Frustração em call center:** Devillers & Vidrascu (2006), "Real-Life Emotions Detection with Lexical and Paralinguistic Cues on Human-Human Call Center Dialogs".
- **Domain adaptation com sintético:** Shen et al. (2021), "Synthetic Data Augmentation for Robust Speech Recognition".

---

*Documento criado em 2026-05-02 como briefing para escrita do relatório.
Mantido em sincronia com `plano_3_semanas.md` (Session Log na §11).*
