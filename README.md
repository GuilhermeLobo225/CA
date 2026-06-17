# 🎧 SmartHandover

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFCA28)
![Whisper](https://img.shields.io/badge/ASR-Whisper--small-6C5CE7)
![wav2vec2](https://img.shields.io/badge/SER-wav2vec2--large-00BFC4)
![XGBoost](https://img.shields.io/badge/Meta-XGBoost-00B050)
![Gradio](https://img.shields.io/badge/Demo-Gradio-F97316)
![License](https://img.shields.io/badge/License-Academic-lightgrey)
![Grade](https://img.shields.io/badge/Grade-19/20-green)

> **Computação Afetiva** | Mestrado em Inteligência Artificial | Universidade do Minho | 2025/26

Ensemble multimodal com *late fusion* para **deteção de frustração** em apoio ao cliente mediado por voz e **encaminhamento automático para operador humano** (*handover*). Combina quatro componentes textuais e um acústico sobre um esquema unificado de cinco classes, agregando a decisão ao nível da conversa via janela deslizante.

---

## 🏆 Destaques

* **Pipeline ponto-a-ponto real:** áudio bruto → Whisper-small (ASR) + wav2vec 2.0 (SER) → 3 classificadores textuais → meta-classificador Stacking → módulo de decisão temporal → *handover*.
* **Decisão a dois níveis:** classificação ao nível do enunciado, escalada ao nível da conversa via janela deslizante de 3 enunciados com regra instantânea + sustentada.
* **Aumento sintético controlado:** corpus de **19 280 textos** (gpt-4o-mini, filtrados por *LLM judge*) + **9 235 áudios** (gpt-4o-mini-tts, degradados para banda telefónica 300–3400 Hz) — compensa o desbalanceamento severo do MELD (17,6:1) e o desalinhamento de domínio (sitcom vs. call center).
* **Calibração honesta:** quatro candidatos de meta-classificador (LR, XGB grid, MLP, Stacking) treinados sobre as **mesmas features** e selecionados por regra composta `0.5·recall_frust + 0.5·macro_F1 s.t. W-F1 ≥ 0.55` na validação.
* **Resultados negativos reportados:** focal loss e calibração probabilística do meta-classificador foram testadas e **não** melhoraram — informativo para localizar o teto na qualidade dos labels.
* **Governance explícita:** alinhamento com RGPD, AI Act e ODS 8/9/10 discutidos como condições da implementação, não como benefícios automáticos.

> 📄 **Relatório:** [Ver PDF do Relatório](Relatório_CA_G1.pdf)  
> 🔗 **Repositório:** [github.com/GuilhermeLobo225/CA](https://github.com/GuilhermeLobo225/CA)  
> 📦 **Checkpoints:** [GitHub Releases](https://github.com/GuilhermeLobo225/CA/releases) (RoBERTa fine-tuned + wav2vec2 fine-tuned)

---

## 📊 Arquitetura

```
                    Áudio do cliente (16 kHz, mono)
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
     Whisper-small      wav2vec2-large    (canal acústico
        (ASR)           superb-er          alimenta direto)
            │           fine-tuned 2 fases
            ▼                 │
   ┌──────────────────┐       │
   │ Texto transcrito │       │
   └────────┬─────────┘       │
            │                 │
   ┌────────┼─────────┐       │
   ▼        ▼         ▼       ▼
 VADER  Hartmann   RoBERTa  audio_*
 (4)    Ekman+    fine-tuned   (5)
         (6)       MELD (5)
   └────────┼─────────┘       │
            └─────┬───────────┘
                  ▼
        Vetor de 20 dimensões
                  │
                  ▼
    Meta-classificador: Stacking soft-vote
       (LR + XGBoost, SMOTE k=5)
                  │
                  ▼
   P(anger, frustration, sadness, neutral, satisfaction)
                  │
                  ▼
     score = 0.6·P(anger) + 0.4·P(frustration)
                  │
                  ▼
    ┌─────────────┴─────────────┐
    ▼                           ▼
 Regra instantânea      Regra sustentada
 score > τ=0.10         média(janela 3) > 0.7·τ
    │                           │
    └─────────────┬─────────────┘
                  ▼
        HANDOVER → operador humano
```

---

## ⚙️ Implementações

### Caminho de texto (3 classificadores em paralelo)

- **VADER** (`src/classifiers/vader_classifier.py`): léxico zero-shot, 4 *features* (positiva, negativa, neutra, *compound*).
- **Ekman+ Hartmann** (`src/classifiers/goemo_classifier.py`): `j-hartmann/emotion-english-distilroberta-base` zero-shot, 6 *features* (anger, disgust, fear, joy, neutral, sadness). *Nota: nas colunas do CSV o prefixo é `goemo_*` por retro-compatibilidade — é um DistilRoBERTa Ekman+, não o GoEmotions original.*
- **RoBERTa-MELD** (`src/training/train_text.py`): `roberta-base` afinado com CE ponderada + `WeightedRandomSampler` + suplemento sintético; *freeze* 2 épocas, depois desbloqueio das 4 camadas superiores. **Núcleo do sinal textual.**

### Caminho de áudio

- **wav2vec 2.0** (`src/classifiers/wav2vec2_finetuned.py`): `superb/wav2vec2-large-superb-er` (pré-treinado em IEMOCAP) afinado em duas fases sobre MELD + CREMA-D + 9 235 sintéticos degradados, em `scripts/run_dayF13_finetune_wav2vec2.py`. Produz 5 *features* (`audio_anger`, `audio_frust`, `audio_sad`, `audio_neut`, `audio_satis`).

### Meta-classificador (4 candidatos avaliados)

`scripts/run_dayG3_meta_v5.py` treina sobre as 20 dimensões com SMOTE k=5:

| Candidato | val W-F1 | val M-F1 | val rec. frust. | test W-F1 | test rec. frust. |
|---|:---:|:---:|:---:|:---:|:---:|
| LR_balanced | 0.630 | 0.504 | 0.225 | 0.652 | 0.28 |
| XGB_v5 (n=300, d=5, η=0.08) | 0.628 | 0.502 | 0.175 | 0.641 | 0.22 |
| MLP_balanced | 0.617 | 0.494 | 0.200 | 0.623 | 0.22 |
| **Stacking (LR + XGB) ✅** | **0.631** | **0.511** | **0.225** | **0.647** | **0.24** |

**Regra de seleção (v5):** `max 0.5·val_frust_recall + 0.5·val_macro_f1 s.t. val_W-F1 ≥ 0.55`.
Vencedor: **Stacking soft-vote**.

### Módulo de decisão temporal

`scripts/run_dayG5_simulate_handover_v5.py`:

- **Score:** `s_t = 0.6·P(anger) + 0.4·P(frustration)`
- **Regra A (instantânea):** dispara se `s_t > τ` num único enunciado.
- **Regra B (sustentada):** dispara se a média de `s_t` sobre 3 enunciados consecutivos excede `0.7·τ`.
- **Calibração (`scripts/run_dayG4_threshold_v5.py`):** grelha em `wa ∈ {0.5..0.7}`, `τ ∈ linspace(0.10, 0.40, 16)`. Vencedor: `(wa=0.6, wf=0.4, τ=0.10)`.

---

## 🏆 Resultados (MELD test, n = 2 329 enunciados, 279 conversas)

### Ao nível do enunciado

| Métrica | v4 (LR + isotonic) | **v5 (Stacking)** | Δ |
|---|:---:|:---:|:---:|
| W-F1 ponderado | 0.6532 | 0.6471 | −0.0061 |
| F1 macro | 0.4896 | 0.4746 | −0.0150 |
| Recall frustração | 0.16 | **0.24** | **+0.08** |
| F1 frustração | 0.1951 | 0.1538 | −0.0413 |
| Score composto (0.5·rec_frust + 0.5·M-F1) | 0.325 | **0.357** | **+0.033** |

### Ao nível da conversa (métrica operacionalmente relevante)

| Métrica | v4 (τ=0.20) | **v5 (τ=0.10)** |
|---|:---:|:---:|
| Conversas com negativo capturadas (170) | 155 | **163** |
| Recall ao nível da conversa | 91.18% | **95.88%** 📈 |
| Precisão ao nível da conversa | 74.52% | 71.81% |
| F1 ao nível da conversa | 0.820 | 0.821 |
| Falsos disparos em limpas (109) | 53 (48.6%) | 64 (58.7%) ⚠️ |
| Latência média de captura | −0.76 | **−1.13** 🔮 |

> 💡 **Insight:** v5 antecipa a frustração em **1.13 enunciados** antes do primeiro enunciado anotado como negativo (latência negativa = deteção pré-emptiva). A escolha v4↔v5 é de **política** (que custo é mais aceitável: 4.7 pp de cobertura adicional vs. 10 pp adicionais de falsos disparos), não de qualidade — o F1 ao nível da conversa é praticamente idêntico.

---

## ⚠️ Limitações Reportadas Honestamente

- **Teto de recall é uma limitação de DADOS, não de modelo**, sustentado por três linhas independentes:
  1. **Focal loss não produziu melhoria** (`dayG1_summary.json`: `"aborted": true`, val frust_recall=0.15 após 10 épocas).
  2. **Candidatos de meta-classificador saturam** num intervalo de ≈3 pp em W-F1.
  3. **Sintético→Sintético** dá recall 0.65 vs **Sintético→MELD** 0.44 — o sinal está lá quando o teste alinha com a distribuição.
- **Mapeamento `fear → frustração` no MELD é ruidoso**: a maioria dos enunciados *fear* do MELD são sobressalto cómico (*"please don't hurt me"*, *"oh my god, no!"*), não frustração de cliente.
- **MELD é diálogo de sitcom**, não chamadas reais — numbers acima não são preditivos de desempenho em contact center real.
- **Língua única (inglês americano)**, dominantemente atores brancos.
- **Assimetria sintético**: cobertura textual (19,3 k) é o dobro da acústica (9,2 k).
- **Taxa de falsos disparos em conversas limpas (58.7%) é elevada** — calibração mais conservadora (τ ≥ 0.25) é decisão de política em produção.

---

## 🛡️ Considerações Éticas

- **RGPD / AI Act:** voz é categoria especial; consentimento explícito e revogável, minimização de dados, retenção limitada.
- **Direito incondicional a operador humano** disponível sempre — não condicionado à inferência do sistema.
- **Política de utilização exclui** repropósito para classificação de operadores ou para preços dinâmicos.
- **Reporte por subgrupo** (género, idade, sotaque) é requisito de implementação responsável.
- **Alinhamento com ODS:** 8 (trabalho digno em contact centers), 9 (inovação responsável), 10 (acesso democrático a apoio humano de qualidade) — apresentados como **condições** da implementação, não benefícios automáticos.

---

## 📂 Estrutura do Repositório

```
guilhermelobo225-ca/
├── configs/                                # Configurações
│   ├── config.yaml                         #   Hiperparâmetros (legacy)
│   ├── handover_threshold_v4.json          #   τ=0.20, regra A
│   ├── handover_threshold_v5.json          #   τ=0.10, regra B (atual)
│   └── iaedu_accounts.example.json         #   Pool de contas para geração sintética
│
├── checkpoints/                            # Modelos persistidos
│   ├── meta_classifier_v4.pkl
│   └── meta_classifier_v4_calibrated.pkl
│
├── data/processed/                         # Métricas e resultados (auditáveis)
│   ├── ablation_results.csv                #   Tabela 3
│   ├── cross_corpus_matrix.csv             #   Tabela 8
│   ├── dayG1_summary.json                  #   Focal loss aborted
│   ├── meta_classifier_v5_summary.json     #   Tabela 5
│   ├── handover_simulation_v5_summary.json #   Tabela 7
│   ├── v5_final_summary.json               #   Síntese final
│   └── top_20_errors.csv                   #   Análise qualitativa de erros
│
├── scripts/                                # Pipeline reproduzível (numerados)
│   ├── run_dayF13_finetune_wav2vec2.py     #   Fine-tune acústico em 2 fases
│   ├── run_dayF14_cross_corpus_matrix.py   #   Matriz cross-corpus
│   ├── run_dayF16_ensemble_v4.py           #   Ensemble v4 (LR + isotonic)
│   ├── run_dayF17_simulate_handover_v4.py  #   Sim. ao nível da conversa v4
│   ├── run_dayG1_train_text_v5.py          #   RoBERTa focal loss (abortado)
│   ├── run_dayG2_features_v5.py            #   Features 20-D v5
│   ├── run_dayG3_meta_v5.py                #   LR/XGB-grid/MLP/Stacking
│   ├── run_dayG4_threshold_v5.py           #   Threshold sweep (Regra A + B)
│   ├── run_dayG5_simulate_handover_v5.py   #   Sim. ao nível da conversa v5
│   └── run_dayG8_final_summary.py          #   Consolidação JSON
│
├── src/
│   ├── classifiers/                        # Componentes do ensemble
│   │   ├── vader_classifier.py             #   VADER (léxico)
│   │   ├── goemo_classifier.py             #   Hartmann Ekman+
│   │   ├── wav2vec2_finetuned.py           #   Wav2Vec2 acústico
│   │   ├── speechbrain_classifier.py       #   SpeechBrain (baseline v4)
│   │   ├── whisper_asr.py                  #   Whisper ASR
│   │   ├── stacking.py                     #   Stacking soft-vote
│   │   ├── ensemble.py                     #   Composição
│   │   ├── pipeline.py                     #   Pipeline genérico
│   │   ├── pipeline_v4.py                  #   Pipeline anterior
│   │   ├── pipeline_v5.py                  #   Pipeline servido pelo demo
│   │   └── fusion_strategies.py            #   Estratégias de fusão alternativas
│   │
│   ├── data/
│   │   ├── load_meld.py                    #   MELD + label unification
│   │   ├── load_cremad.py                  #   CREMA-D
│   │   └── synthetic/                      #   Pipeline de geração sintética
│   │       ├── generate_text.py            #     OpenAI / IAEDU
│   │       ├── filter_text.py              #     Heurísticas + LLM-judge
│   │       ├── generate_audio.py           #     gpt-4o-mini-tts
│   │       ├── degrade_audio.py            #     Banda telefónica
│   │       └── ...
│   │
│   ├── decision/
│   │   ├── handover.py                     #   API genérica de referência
│   │   └── simulate_handover.py            #   Simulação ao nível da conversa
│   │
│   ├── demo/
│   │   ├── app.py                          #   Demo base
│   │   └── app_callcenter.py               #   Demo principal (pipeline v5)
│   │
│   ├── evaluation/
│   │   ├── ablation.py                     #   Leave-one-modality-out
│   │   ├── error_analysis.py               #   Top-20 erros, severity tiers
│   │   └── metrics.py
│   │
│   ├── models/
│   │   └── text_encoder.py                 #   RoBERTa-base wrapper
│   │
│   └── training/
│       ├── train_text.py                   #   CE ponderada (v4, retido)
│       ├── train_text_v5.py                #   Focal loss (abortado)
│       ├── train_audio.py
│       ├── train_meta_balanced.py
│       └── focal_loss.py
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Reprodução

### Pré-requisitos

- **Python 3.11+** com `pip`.
- **GPU consumer recomendada** (testado em 16 GB VRAM); CPU corre, mas demoroso.
- Chave OpenAI / portal IAEDU **apenas se quiseres regenerar o sintético** (não necessária para reproduzir os resultados finais — basta descarregar os *checkpoints*).

### Passos

1. **Clonar e instalar dependências:**
   ```bash
   git clone https://github.com/GuilhermeLobo225/CA.git
   cd CA
   pip install -r requirements.txt
   ```

2. **Descarregar *checkpoints* pré-treinados (se disponíveis):**
   ```bash
   python scripts/download_checkpoints.py
   ```

3. **Reproduzir pipeline v5 (a partir das *features* v4 já calculadas):**
   ```bash
   # Reconstruir features 20-D
   python scripts/run_dayG2_features_v5.py
   # Treinar 4 candidatos de meta-classificador
   python scripts/run_dayG3_meta_v5.py
   # Calibrar limiar (regra A + regra B)
   python scripts/run_dayG4_threshold_v5.py
   # Simular ao nível da conversa
   python scripts/run_dayG5_simulate_handover_v5.py
   # Sumário consolidado
   python scripts/run_dayG8_final_summary.py
   ```

4. **Demo Gradio (apresentação ao vivo):**
   ```bash
   python -m src.demo.app_callcenter
   # Abre http://localhost:7860
   # Abre public URL
   ```

5. **(Opcional) Regenerar corpus sintético:**
   ```bash
   cp .env.example .env  # depois preencher chaves
   cp configs/iaedu_accounts.example.json configs/iaedu_accounts.json
   python -m src.data.synthetic.generate_text
   python -m src.data.synthetic.filter_text
   python -m src.data.synthetic.generate_audio
   ```

> Cada *script* numerado escreve um sumário JSON/CSV em `data/processed/`. **Cada número citado no relatório vem de um destes ficheiros** — auditável célula a célula.

---

## 🔮 Trabalho Futuro

- **Recolher corpus real e consentido** de chamadas de contact center (impacto isolado esperado mais alto).
- **Re-anotar manualmente o subconjunto `fear` do MELD** para isolar empiricamente o teto imposto pelo ruído de label.
- Extensão para **português** (XLM-RoBERTa + recriação do sintético em PT-PT).
- Substituir o meta-classificador por **Transformer pequeno** sobre embeddings unimodais.
- **Análise facial** quando canal vídeo disponível.
- **Intervalos de confiança bootstrap** sobre as métricas reportadas.
- **Reporte por subgrupo** (género, idade, sotaque) — requisito ético explicitado em §7.

---


## 👥 Grupo — MIA

| Nome | Nº | Email | DELTA |
|------|----|-------|:---:|
| Guilherme Lobo Pinto | PG60225 | pg60225@alunos.uminho.pt | 0.00 |
| Pedro Alexandre Silva Gomes | PG60289 | pg60289@alunos.uminho.pt | 0.00 |
| Simão Novais Vieira da Silva | PG60393 | pg60393@alunos.uminho.pt | 0.00 |

---

## 📜 Licença

Trabalho académico. Universidade do Minho, Escola de Engenharia, Departamento de Informática. Unidade Curricular de **Computação Afetiva**, Mestrado em Inteligência Artificial, 2025/2026.
