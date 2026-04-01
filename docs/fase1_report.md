# Smart Handover -- Relatorio Fase 1: Setup e Validacao

## 1. Estrutura do Projeto (Scaffolding)

```
CA/
├── configs/config.yaml        -- configuracoes centralizadas
├── data/raw/                  -- dados brutos (armazenamento local)
├── data/processed/            -- features pre-processadas (Fase 2+)
├── src/
│   ├── __init__.py            -- torna src/ um package Python importavel
│   ├── data/                  -- scripts de ingestao e pre-processamento
│   │   ├── __init__.py
│   │   └── load_meld.py       -- script de download + remapeamento de emocoes
│   ├── models/                -- definicoes de modelos e testes
│   │   ├── __init__.py
│   │   └── vram_test.py       -- script de validacao de VRAM
│   ├── decision/              -- logica de decisao/fusao (Fase 2+)
│   ├── evaluation/            -- metricas, confusion matrix (Fase 2+)
│   └── training/              -- training loops (Fase 2+)
├── notebooks/                 -- Jupyter notebooks para exploracao
└── requirements.txt           -- dependencias do projeto
```

### Porque e que esta estrutura e importante?

Esta e a convencao standard em projetos de Machine Learning -- separa dados, modelos, treino e avaliacao em modulos independentes. Cada pasta dentro de `src/` contem um ficheiro `__init__.py`, que diz ao Python para tratar aquela pasta como um **package**, permitindo imports como:

```python
from src.data.load_meld import load_meld, TARGET_LABELS
```

Sem os `__init__.py`, o Python nao reconhece os imports e o projeto nao funciona como um todo integrado.

### Porque e que criamos esta estrutura?

Quando o projeto cresce (e vai crescer -- Fase 2, 3, etc.), ter tudo organizado desde o inicio evita o caos de ter scripts soltos. Cada modulo tem uma responsabilidade clara:
- `src/data/` -- tudo o que toca em dados
- `src/models/` -- tudo o que define redes neurais
- `src/training/` -- tudo o que treina modelos
- `src/evaluation/` -- tudo o que avalia resultados

---

## 2. Dependencias (requirements.txt)

```
torch>=2.1.0                    # Framework de deep learning (o "motor" das redes neurais)
torchaudio>=2.1.0               # Extensao do PyTorch para processamento de audio
transformers>=4.35.0            # Biblioteca da HuggingFace -- acesso a modelos pre-treinados
datasets>=2.14.0,<3.0.0        # Biblioteca da HuggingFace para carregar datasets
librosa>=0.10.1                 # Processamento de audio (espectrogramas, MFCCs, etc.)
soundfile>=0.12.1               # Leitura/escrita de ficheiros de audio
opensmile>=2.5.0                # Extracao de features acusticas de baixo nivel (eGeMAPS)
scikit-learn>=1.3.0             # Metricas, utilidades de ML classico
pandas>=2.1.0                   # Manipulacao de dados tabulares
tqdm>=4.66.0                    # Barras de progresso
pyyaml>=6.0                    # Leitura de ficheiros de configuracao YAML
```

### Decisoes importantes:

**O pin `datasets<3.0.0`** -- A versao 4.x da biblioteca `datasets` **removeu** o suporte a "loading scripts" (ficheiros `.py` customizados dentro de repos HuggingFace). O dataset `ajyy/MELD_audio` depende de um desses scripts. Sem este pin, o erro era: `RuntimeError: Dataset scripts are no longer supported, but found MELD_audio.py`.

**PyTorch com CUDA 12.8** -- O `pip install torch` por defeito instala a versao **CPU-only**. Para usar a GPU, e necessario instalar com um indice especifico: `pip install torch --index-url https://download.pytorch.org/whl/cu128`. A RTX 5060 Ti usa arquitectura **Blackwell** (compute capability 12.0), que exige **CUDA 12.8 no minimo**. Com CUDA 12.6, o PyTorch dava warnings sobre `sm_120` nao suportado.

### Porque e que este ficheiro e importante?

O `requirements.txt` e o "contrato" das dependencias do projeto. Qualquer pessoa que clone o repositorio pode correr `pip install -r requirements.txt` e ter exactamente o mesmo ambiente. Sem ele, cada pessoa teria de adivinhar que bibliotecas instalar e em que versoes.

---

## 3. Script de Ingestao de Dados (load_meld.py)

### 3.1 -- O problema do dataset original

O plano inicial era usar `declare-lab/MELD` directamente do HuggingFace. Contudo, esse repositorio contem apenas dois ficheiros `.tar.gz` gigantes sem nenhum script de carregamento. A biblioteca `datasets` encontrou um **TAR dentro de outro TAR** e crashou:

```
NotImplementedError: Extraction protocol for TAR archives like
'memory://MELD.Raw/train.tar.gz' is not implemented in streaming mode
```

**Solucao**: Usar `ajyy/MELD_audio` -- o **mesmo dataset MELD**, mas com:
- Audio ja extraido dos videos em formato FLAC
- Resampleado a 16 kHz mono (exactamente o que o wav2vec2 espera)
- Um script de carregamento funcional no HuggingFace

### 3.2 -- O mapeamento de emocoes (a parte mais importante conceptualmente)

O MELD tem **7 emocoes** (retiradas da serie Friends). O nosso sistema de detecao de frustracao precisa de **5 classes**:

| MELD Original | Classe Alvo | Justificacao |
|---|---|---|
| anger | anger | Mapeamento directo |
| disgust | anger | Emocao adjacente no modelo circumplexo (valencia negativa, alta ativacao) |
| fear | frustration | Melhor proxy -- em contextos de servico, medo de nao resolver = frustracao |
| joy | satisfaction | Renomeacao directa -- alegria em contexto de servico = satisfacao |
| neutral | neutral | Mapeamento directo |
| sadness | sadness | Mapeamento directo |
| surprise | **ELIMINADA** | Valencia ambigua (pode ser positiva ou negativa), nao util para detecao de frustracao |

```python
MELD_LABEL_MAP = {
    "anger":    "anger",
    "disgust":  "anger",
    "fear":     "frustration",
    "joy":      "satisfaction",
    "neutral":  "neutral",
    "sadness":  "sadness",
    "surprise": None,  # eliminada
}
```

### 3.3 -- Dicionarios de lookup

```python
TARGET_LABELS = ["anger", "frustration", "sadness", "neutral", "satisfaction"]
TARGET_LABEL2ID = {"anger": 0, "frustration": 1, "sadness": 2, "neutral": 3, "satisfaction": 4}
TARGET_ID2LABEL = {0: "anger", 1: "frustration", 2: "sadness", 3: "neutral", 4: "satisfaction"}
```

As redes neurais trabalham com **numeros inteiros**, nao strings. O modelo preve `3` e nos traduzimos para `"neutral"`. Estes dicionarios sao usados em todo o projeto para fazer essa traducao.

### 3.4 -- A funcao `load_meld()`

```python
ds = load_dataset(_HF_DATASET, split=split, streaming=streaming, trust_remote_code=True)
```

- `split` -- escolhe train/validation/test
- `streaming=True` -- carrega os dados incrementalmente em vez de tudo para RAM
- `trust_remote_code=True` -- permite executar o script `.py` do repositorio HuggingFace

A funcao de remapeamento:

```python
def remap(example):
    original_str = example["emotion"].lower().strip()  # ex: "Fear" -> "fear"
    mapped = MELD_LABEL_MAP.get(original_str)           # "fear" -> "frustration"
    example["original_emotion"] = original_str
    example["target_emotion"] = mapped if mapped else "DROP"
    example["target_label"] = TARGET_LABEL2ID[mapped] if mapped else -1
    return example
```

- `.lower().strip()` e uma **sanitizacao** -- garante que mesmo `"Fear"` ou `" fear "` funcionam
- `.map(remap)` aplica esta funcao a **cada linha** do dataset
- `.filter(lambda x: x["target_label"] != -1)` remove todas as linhas de "surprise"

**Bug encontrado**: Na primeira versao, o campo `emotion` era tratado como inteiro (`ClassLabel`) quando na realidade era uma string. O dicionario `MELD_ID2EMOTION` nunca fazia match, tudo recebia `"unknown"`, tudo era filtrado, resultando em 0 amostras. Corrigido ao ler directamente como string.

### 3.5 -- Estatisticas do dataset

```
Split: train      | Total: 8.783 amostras
  anger              1.380  (15.7%)
  frustration          268  ( 3.1%)  << ALERTA: severamente sub-representada
  sadness              683  ( 7.8%)
  neutral            4.709  (53.6%)  << classe dominante
  satisfaction       1.743  (19.8%)

Split: validation  | Total:   958 amostras
Split: test        | Total: 2.329 amostras
```

O dataset esta **muito desequilibrado**. `neutral` tem 53.6% enquanto `frustration` tem apenas 3.1%. Um modelo naive que previsse sempre "neutral" acertaria 53.6% do tempo. Isto vai exigir mitigacao na Fase 2 (weighted loss, oversampling, ou focal loss).

### Porque e que este script e importante?

E a **fundacao de tudo**. Sem dados correctamente carregados e mapeados, nada funciona. Este script garante que:
1. O dataset e descarregado automaticamente do HuggingFace
2. As 7 emocoes originais sao convertidas nas 5 classes do nosso sistema
3. As constantes `TARGET_LABELS`, `TARGET_LABEL2ID`, `TARGET_ID2LABEL` sao definidas aqui e reutilizadas por **todos** os outros modulos

---

## 4. Script de VRAM (vram_test.py)

### 4.1 -- Porque e que este teste existe?

Temos **16 GB de VRAM** (RTX 5060 Ti). Na Fase 2 vamos carregar **dois modelos ao mesmo tempo** (RoBERTa para texto + Wav2Vec2 para audio) e treina-los. Se excedermos a VRAM, o PyTorch crasha com `CUDA out of memory`. Este script simula o cenario real **antes** de perdermos horas a descobrir na fase de treino.

### 4.2 -- Os dois modelos

```python
text_model = RobertaModel.from_pretrained("roberta-base").to(device).eval()
audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()
```

| Modelo | Parametros | Funcao |
|---|---|---|
| `roberta-base` | 125M | Encoder de texto (baseado em Transformers), processa transcricoes das utterances |
| `wav2vec2-base` | 95M | Encoder de audio, aprende representacoes directamente da waveform em bruto |

- `.to(device)` -- move os pesos do modelo da RAM do CPU para a VRAM da GPU
- `.eval()` -- modo de inferencia (desliga dropout, coloca batch norm em modo fixo)

### 4.3 -- Inputs dummy (simulam dados reais)

```python
# Texto: 8 frases de 128 tokens cada
input_ids = torch.randint(0, 50265, (batch_size, seq_len), device=device)
attention_mask = torch.ones_like(input_ids)

# Audio: 8 clips de 10 segundos a 16kHz = 160.000 amostras cada
audio_input = torch.randn(batch_size, sample_rate * audio_seconds, device=device)
```

- `50265` e o tamanho do vocabulario do RoBERTa
- `attention_mask` de 1s diz ao modelo "presta atencao a todos os tokens"
- `16000 * 10 = 160.000` amostras por clip -- wav2vec2 espera audio raw a 16kHz

### 4.4 -- FP16 autocast (a parte mais critica para VRAM)

```python
with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    text_out = text_model(input_ids=input_ids, attention_mask=attention_mask)
```

- **`torch.no_grad()`** -- desliga o calculo de gradientes (poupa ~50% de memoria). Aqui so queremos testar inferencia, nao treinar.
- **`torch.amp.autocast(..., dtype=torch.float16)`** -- **Automatic Mixed Precision**. Em vez de usar FP32 (32 bits por numero = 4 bytes), usa FP16 (16 bits = 2 bytes). Reduz a VRAM quase para metade e e **mais rapido** em GPUs modernas como a RTX 5060 Ti (que tem tensor cores optimizados para FP16).

### 4.5 -- Resultados obtidos

```
VRAM apos text model:     476.7 MB    -- pesos do RoBERTa em memoria
VRAM apos ambos modelos:  837.2 MB    -- + pesos do Wav2Vec2
VRAM apos text forward:   854.3 MB    -- + activacoes intermedias do texto
VRAM apos audio forward:  873.8 MB    -- + activacoes intermedias do audio

Pico alocado:  2.06 GB               -- pico maximo durante o forward pass
Pico reservado: 2.53 GB              -- memoria reservada pelo alocador CUDA
Headroom:       13.94 GB             -- sobram ~14 GB dos 16 GB
```

**2.06 GB de pico** com `torch.no_grad()`. No treino real (com gradientes), sera ~3-4x mais, ou seja **~6-8 GB**. Ainda assim, dentro dos 16 GB. Os ~14 GB de headroom dao margem para gradient accumulation e batches maiores.

### 4.6 -- Shapes do output

```
Text  output: [8, 128, 768]   -> 8 frases x 128 tokens x 768 dimensoes por token
Audio output: [8, 499, 768]   -> 8 clips x 499 frames x 768 dimensoes por frame
```

Ambos os encoders produzem vectores de **768 dimensoes** -- isto e conveniente para a fusao na Fase 2, porque partilham a mesma dimensionalidade de embedding.

### Porque e que este script e importante?

Sem ele, arriscaramos chegar a Fase 2, lancar o treino, esperar 20 minutos a carregar tudo, e receber um `CUDA out of memory`. Este teste demora 30 segundos e garante que o hardware suporta a arquitectura antes de investirmos tempo no treino.

---

## 5. Resumo dos Problemas Encontrados e Solucoes

| # | Problema | Causa Raiz | Solucao |
|---|---|---|---|
| 1 | Crash na extracao de TAR | `declare-lab/MELD` = TAR aninhado sem loading script | Mudamos para `ajyy/MELD_audio` |
| 2 | "Dataset scripts no longer supported" | `datasets 4.x` removeu a funcionalidade | Pin `datasets>=2.14.0,<3.0.0` |
| 3 | 0 amostras apos filtragem | Campo `emotion` e string, nao inteiro | Removemos `MELD_ID2EMOTION`, lemos string directamente |
| 4 | "CUDA not available" | PyTorch instalado sem suporte CUDA | Reinstalado com `--index-url .../cu128` |
| 5 | Warning sm_120 incompativel | CUDA 12.6 nao suporta arquitectura Blackwell | Upgrade para CUDA 12.8 (`cu128`) |
| 6 | UnicodeEncodeError no checkmark | Windows cp1252 nao suporta `✓` | Substituido por `[PASS]` em ASCII |
