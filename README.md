# 🤖 SmartHandover: Deteção Multimodal de Frustração em Apoio ao Cliente

## 📖 Descrição do Projeto

O **SmartHandover** é um protótipo de computação afetiva que monitoriza, em tempo real, o estado emocional do utilizador durante chamadas de apoio ao cliente. Em vez de prender um cliente frustrado num ciclo infinito de respostas automáticas, o sistema analisa sinais multimodais — **texto** (transcrição) e **áudio** (prosódia) — para detetar frustração e raiva. Quando um limiar crítico de emoção negativa é ultrapassado, a chamada é reencaminhada automaticamente para um agente humano (*handover*).

## 🎓 Contexto Académico

Projeto desenvolvido no âmbito da unidade curricular de **Computação Afetiva** do **Mestrado em Inteligência Artificial** da **Universidade do Minho** (ano letivo 2025/2026).

## 🧠 Arquitetura e Metodologia

### Modelo de Emoção

Adota-se uma abordagem **híbrida** que combina três modelos teóricos complementares:

- **Ekman (discreto):** classificação primária num subconjunto de 5 categorias — *raiva, frustração, tristeza, neutro, satisfação*.
- **Russell (dimensional):** estimação contínua da dimensão de *arousal* (ativação) para monitorizar a intensidade emocional ao longo da interação.
- **Plutchik (gradação):** escala de intensidade (*aborrecimento → raiva → fúria*) para definir diferentes níveis de urgência no reencaminhamento.

### Pipeline Multimodal

O sistema segue um pipeline modular em cadeia:

1. **ASR (Transcrição):** conversão fala → texto via **Whisper** (small/medium).
2. **Módulo Textual:** tokenização e classificação emocional com **RoBERTa-base** (~125M parâmetros), com *fine-tuning* sobre dados conversacionais.
3. **Módulo Acústico:** extração de embeddings com **wav2vec 2.0 base** (~95M parâmetros), complementados com features prosódicas tradicionais (F0, energia, jitter, shimmer) via **openSMILE**.
4. **Módulo de Fusão:** comparação de três estratégias — fusão tardia por concatenação (*baseline*), atenção cruzada e fusão ao nível da decisão.
5. **Módulo de Decisão (*Handover*):** janela temporal com média ponderada exponencial; o reencaminhamento é acionado quando o *arousal* ultrapassa um limiar calibrado ou N enunciados consecutivos são classificados como *raiva*.

### Estratégia Linguística

O desenvolvimento segue duas fases:

1. **Fase 1 — Inglês:** validação da arquitetura sobre datasets de referência, permitindo comparação direta com benchmarks da literatura.
2. **Fase 2 — Português:** adaptação com modelos multilingues (XLM-RoBERTa-base para texto; wav2vec 2.0 multilingue para áudio).

## 📊 Datasets

| Dataset | Modalidades | Dimensão | Utilização |
|---|---|---|---|
| **IEMOCAP** | Texto + Áudio | ~12h de diálogos | Treino e avaliação da fusão multimodal |
| **MELD** | Texto + Áudio + Vídeo | ~13.000 enunciados | Avaliação de dinâmicas conversacionais |
| **GoEmotions** | Texto | ~58.000 comentários | Pré-treino complementar do módulo textual |

## 📐 Métricas de Avaliação

- **Métrica primária:** Weighted F1-score (adequada ao desbalanceamento de classes).
- **Métricas complementares:** Accuracy, Precision/Recall por classe, F1-score macro.
- **Módulo de handover:** Curvas Precision-Recall e PR-AUC para calibração do limiar de reencaminhamento.
- **Validação:** *Stratified k-fold cross-validation*.

## ⚖️ Considerações Éticas

- **Privacidade e Consentimento:** anonimização dos dados de voz e conformidade com o RGPD.
- **Viés:** análise de desempenho por género, sotaque e faixa etária.
- **Chilling Effect:** discussão sobre o impacto da monitorização emocional no comportamento natural dos utilizadores.

## 🗂️ Estrutura do Repositório

> *A completar à medida que o desenvolvimento avança.*

```
SmartHandover/
├── README.md
├── data/                  # Scripts de download e pré-processamento dos datasets
├── src/
│   ├── asr/               # Módulo de transcrição (Whisper)
│   ├── text/              # Módulo textual (RoBERTa)
│   ├── audio/             # Módulo acústico (wav2vec 2.0)
│   ├── fusion/            # Estratégias de fusão multimodal
│   └── decision/          # Módulo de decisão e handover
├── notebooks/             # Notebooks de experimentação e análise
├── results/               # Resultados experimentais, gráficos e tabelas
└── requirements.txt
```

## 👥 Equipa

| Número | Nome |
|---|---|
| PG60225 | Guilherme Lobo Pinto |
| PG60289 | Pedro Alexandre Silva Gomes |
| PG60393 | Simão Novais Vieira da Silva |
