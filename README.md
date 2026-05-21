# 🚀 SmartHandover — Deteção Multimodal de Frustração

> 🎓 **Projeto de Mestrado** | Computação Afetiva, MIA (U.Minho, 2025/2026)
> 👥 **Equipa:** Guilherme Pinto, Pedro Gomes, Simão Silva

---

## 🎯 O que faz?
Analisa chamadas de suporte em tempo real e decide quando **transferir a chamada para um humano (*handover*)**. Combina **1 canal de áudio** e **3 de texto** através de um meta-classificador.

*📊 **Nota:** Validado no dataset MELD (diálogos da série "Friends").*

---

## 🏗️ Arquitetura e Regras (v5)

1. 🎧 **Áudio (16 kHz mono)** → Transcrito por `whisper-small`.
2. 🧠 **Features (20-dim):**
   * **Texto:** VADER + Hartmann RoBERTa + RoBERTa-base (fine-tuned).
   * **Áudio:** wav2vec2-large (fine-tuned).
3. ⚖️ **Meta-Classificador:** Stacking (Regressão Logística + XGBoost).
4. 📈 **Score de Handover:** `Score = 0.6 * P(raiva) + 0.4 * P(frustração)`
5. 🚨 **Gatilhos:**
   * **Imediato:** Score > 0.10 numa única fala.
   * **Sustentado:** Média das últimas 3 falas > 0.07.

---

## 💾 Dados Sintéticos
* **Problema:** A frustração original representa apenas ~3% do MELD.
* **Solução:** Gerados 19k textos (gpt-4o-mini) e 9k áudios (gpt-4o-mini-tts) com filtro telefónico (300–3400 Hz) para simular um contact center real.

---

## 🏆 Resultados (MELD Test)

| Métrica (Nível Conversa) | v4 (Reg. Logística) | v5 (Stacking) |
| :--- | :---: | :---: |
| **Recall (Detetar Frustração)** | 91.18% | **95.88%** 📈 |
| **Precisão (Acerto)** | 74.52% | 71.81% |
| **Falsos Alarmes** | 48.62% | 58.72% |
| **Latência Média (Falas)** | -0.76 | **-1.14** 🔮 |

> 💡 **Insight:** A v5 antecipa a frustração antes de ela ser explicitamente dita (latência negativa), mas gera mais falsos alarmes.

---

## ⚠️ Limitações e Ética
* **Focal Loss Abortado:** O re-treino da RoBERTa falhou nos testes. A v5 melhora apenas no meta-classificador e nos thresholds.
* **Dados Enviesados:** Mapeamento artificial de "medo" para "frustração" no MELD. Foco exclusivo em inglês americano.
* **Privacidade:** A voz é um dado biométrico (exige conformidade estrita com o RGPD).

---

## 💻 Como Executar

```bash
# Setup inicial
pip install -r requirements.txt
python scripts/download_checkpoints.py

# Pipeline v5
python scripts/run_dayG3_meta_v5.py
python scripts/run_dayG4_threshold_v5.py
python scripts/run_dayG5_simulate_handover_v5.py
