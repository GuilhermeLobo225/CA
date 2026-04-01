# Smart Handover -- Relatorio Fase 2: Arquitectura e Pipeline de Treino

## Visao Geral

A Fase 1 preparou os alicerces: dados carregados, emocoes remapeadas, VRAM validada. A Fase 2 constroi o **sistema completo de detecao de frustracao multimodal** -- desde os encoders individuais (texto e audio) ate a fusao, o treino e a avaliacao.

O pipeline completo:

```
Texto ("I can't believe this!")     Audio (waveform 16kHz)
        |                                    |
   RobertaTokenizer                   Wav2Vec2Processor
        |                                    |
   TextEncoder (roberta-base)         AudioEncoder (wav2vec2-base)
        |                                    |
   attention pool                     attention pool
        |                                    |
   [batch, 768]                       [batch, 768]
        |                                    |
        +-------- FUSAO (concat) -----------+
                      |
                 [batch, 1536]
                      |
                 MLP classifier
                      |
                 [batch, 5] logits
                      |
           anger | frustration | sadness | neutral | satisfaction
```

---

## 1. Ficheiro de Configuracao (configs/config.yaml)

### O que faz

Centraliza **todos os hiperparametros** do projeto num unico ficheiro YAML. Em vez de ter numeros magicos espalhados pelo codigo, tudo e definido aqui e lido pelo `train.py` no inicio.

### Porque e que e importante

1. **Reprodutibilidade** -- qualquer experiencia pode ser reproduzida exactamente ao guardar o `config.yaml` que foi usado.
2. **Experimentacao rapida** -- para testar batch_size=16 em vez de 8, basta mudar uma linha no YAML em vez de editar codigo Python.
3. **Transparencia** -- qualquer pessoa que leia o ficheiro percebe imediatamente como o sistema esta configurado.

### Seccoes explicadas

```yaml
data:
  max_text_length: 128          # maximo de tokens por frase (RoBERTa suporta ate 512,
                                # mas 128 e suficiente -- utterances do MELD sao curtas)
  max_audio_length_sec: 15      # truncar audio acima de 15 segundos (a maioria das
                                # utterances do MELD tem 2-8 segundos)
  audio_sample_rate: 16000      # wav2vec2 exige audio a 16kHz
```

O `max_text_length: 128` e uma decisao de eficiencia. O RoBERTa aceita ate 512 tokens, mas as utterances do MELD sao falas curtas de dialogo ("You must've had your hands full."). 128 tokens sao mais que suficientes e poupam VRAM.

```yaml
fusion:
  type: "concat"                # tipo de fusao (explicado na seccao 4)
  hidden_dim: 512               # dimensao da camada escondida do classificador
  dropout: 0.3                  # 30% dropout para regularizacao
```

O `dropout: 0.3` significa que durante o treino, 30% dos neuronios sao aleatoriamente "desligados" em cada forward pass. Isto forca a rede a nao depender demasiado de nenhum neuronio individual, reduzindo o overfitting.

```yaml
training:
  batch_size: 8                 # amostras processadas por passo da GPU
  accumulation_steps: 4         # batch efectivo = 8 x 4 = 32
  learning_rate: 2.0e-5         # taxa de aprendizagem para o classificador
  encoder_lr_multiplier: 0.1    # encoders aprendem 10x mais devagar
  freeze_encoders: true         # congelar encoders nas primeiras 3 epochs
  freeze_epochs: 3
  unfreeze_top_n_layers: 2      # descongelar as 2 ultimas camadas apos epoch 3
```

**Porque batch=8 com accumulation=4?** O batch real na GPU e 8 (cabe na VRAM). Mas batches de 8 sao pequenos -- os gradientes sao "ruidosos". O gradient accumulation simula um batch de 32: acumula gradientes de 4 mini-batches antes de actualizar os pesos. Resultado: estabilidade de batch=32 com a memoria de batch=8.

**Porque freeze 3 epochs?** Com apenas 8.783 amostras de treino e ~220M parametros nos encoders, fine-tunar tudo desde o inicio causa overfitting. Nas 3 primeiras epochs, so a "cabeca" (fusao + classificador) treina, aprendendo a usar as features pre-treinadas. Depois, desbloqueamos as 2 ultimas camadas dos encoders para adaptacao ao dominio.

**Porque `encoder_lr_multiplier: 0.1`?** Os encoders ja vem pre-treinados com conhecimento linguistico/acustico de milhoes de amostras. Queremos ajusta-los subtilmente, nao destruir esse conhecimento. Uma learning rate 10x menor (2e-6 vs 2e-5) garante que as actualizacoes sao pequenas e cautelosas.

```yaml
training:
  use_weighted_loss: true       # pesos inversamente proporcionais a frequencia
  early_stopping_patience: 5    # parar se nao melhorar em 5 epochs
  early_stopping_metric: "frustration_recall"  # metrica chave
```

**Porque weighted loss?** Como vimos na Fase 1, `frustration` tem 3.1% dos dados e `neutral` tem 53.6%. Sem pesos, o modelo aprende a dizer "neutral" para quase tudo (e tem 53.6% de accuracy). Com pesos inversamente proporcionais a frequencia, cada amostra de `frustration` "vale" ~6.5x mais que uma de `neutral`.

**Porque early stopping por frustration_recall?** O objectivo principal do sistema e detectar frustracao. Accuracy global nao serve -- um modelo que ignore frustracao pode ter 90% de accuracy. O recall de frustracao mede: "de todas as amostras realmente frustradas, quantas o modelo detectou?". Se esta metrica nao melhora em 5 epochs seguidas, paramos.

---

## 2. Text Encoder (src/models/text_encoder.py)

### O que faz

Recebe texto tokenizado (uma frase convertida em numeros inteiros) e produz um **vector de 768 dimensoes** que representa o significado emocional dessa frase.

### Porque e que e importante

Este e o "cerebro" linguistico do sistema. O RoBERTa foi pre-treinado em milhares de milhoes de palavras e ja "sabe" que "I'm so angry!" tem conotacao negativa. Nos reutilizamos esse conhecimento em vez de treinar do zero.

### Arquitectura interna

```
Texto: "I can't believe this!"
         |
    RobertaTokenizer
         |
    [101, 146, 64, 75, 519, 42, 328, 102]    -- token IDs
         |
    RoBERTa (12 camadas Transformer)
         |
    [batch, 128, 768]                         -- 128 tokens, cada um com 768 dimensoes
         |
    Attention Pooling (aprendido)
         |
    [batch, 768]                              -- UM vector por frase
```

### O mecanismo de Attention Pooling (o detalhe mais importante)

O RoBERTa produz um vector de 768 dimensoes **para cada token**. Mas nos precisamos de **um unico vector por frase**. Como combinar 128 vectores num so?

**Opcao simples (mean pooling)**: media de todos os tokens. Problema: trata todas as palavras igualmente. "The" e "angry" teriam o mesmo peso.

**A nossa opcao (attention pooling aprendido)**: uma camada linear `768 -> 1` aprende a atribuir um "peso de importancia" a cada token. Depois de um softmax, os pesos somam 1. O vector final e a media ponderada.

```python
# Cada token recebe um score de importancia
attn_scores = self.attn_pool(hidden_states).squeeze(-1)   # [batch, 128]

# Tokens de padding recebem -infinito (peso zero apos softmax)
attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

# Normalizar para obter pesos que somam 1
attn_weights = F.softmax(attn_scores, dim=-1)             # [batch, 128]

# Media ponderada
pooled = torch.bmm(attn_weights.unsqueeze(1), hidden_states).squeeze(1)  # [batch, 768]
```

O modelo pode aprender, por exemplo, que a palavra "frustrated" e mais importante que "the" para a classificacao emocional. Isto adiciona apenas 769 parametros treinaveis (768 pesos + 1 bias) -- custo negligivel.

### Os metodos freeze/unfreeze

```python
def freeze(self):
    for param in self.roberta.parameters():
        param.requires_grad = False

def unfreeze_top_n(self, n):
    total_layers = len(self.roberta.encoder.layer)  # 12
    for i in range(total_layers - n, total_layers):
        for param in self.roberta.encoder.layer[i].parameters():
            param.requires_grad = True
```

`requires_grad = False` significa "nao calcules gradientes para estes parametros" -- eles ficam fixos durante o treino. Isto poupa VRAM (nao precisa de guardar gradientes) e evita overfitting.

Quando chamamos `unfreeze_top_n(2)`, as camadas 10 e 11 (as duas ultimas) voltam a ser treinaveis. As camadas superiores do RoBERTa codificam informacao mais abstracta e tarefa-especifica, enquanto as inferiores codificam sintaxe basica. Faz sentido adaptar as superiores ao nosso dominio emocional.

---

## 3. Audio Encoder (src/models/audio_encoder.py)

### O que faz

Recebe audio em bruto (waveform a 16kHz) e produz um **vector de 768 dimensoes** que representa as caracteristicas acusticas emocionais (tom de voz, velocidade, intensidade, etc.).

### Porque e que e importante

O texto sozinho nao apanha tudo. A frase "Fine, whatever" pode ser neutra ou extremamente frustrada -- depende do tom de voz. O audio captura essas nuances prosodicas que o texto nao consegue.

### Arquitectura interna

```
Audio: [0.02, -0.01, 0.03, ...]   -- waveform raw a 16kHz
         |
    CNN Feature Extractor (7 camadas convolucionais)
         |
    [batch, 499, 512]              -- 499 frames, cada um com 512 dimensoes
         |
    Transformer Encoder (12 camadas)
         |
    [batch, 499, 768]              -- 499 frames, cada um com 768 dimensoes
         |
    Attention Pooling (aprendido)
         |
    [batch, 768]                   -- UM vector por clip de audio
```

### O problema do downsampling da mascara

O wav2vec2 tem uma peculiaridade que o RoBERTa nao tem: o **CNN feature extractor** reduz a resolucao temporal. Um clip de 10 segundos a 16kHz tem 160.000 amostras, mas apos o CNN, ficam ~499 frames. Quando clips num batch tem comprimentos diferentes, o mais curto e preenchido com zeros (padding).

A attention mask do processador esta na resolucao da waveform (160.000 posicoes). Mas apos o CNN, precisamos de uma mask na resolucao dos frames (499 posicoes). O metodo `_compute_frame_mask` faz esta conversao:

```python
def _compute_frame_mask(self, attention_mask, hidden_states):
    if attention_mask is None:
        return torch.ones(hidden_states.shape[:2], ...)  # sem padding, tudo valido

    # Comprimentos reais da waveform (ex: [160000, 80000, 48000, ...])
    input_lengths = attention_mask.sum(dim=-1)

    # Comprimentos apos o CNN (ex: [499, 249, 149, ...])
    output_lengths = self.wav2vec2._get_feat_extract_output_lengths(input_lengths)

    # Construir mask: 1 para posicoes < comprimento, 0 caso contrario
    frame_indices = torch.arange(num_frames, device=hidden_states.device)
    frame_mask = (frame_indices.unsqueeze(0) < output_lengths.unsqueeze(1)).long()
    return frame_mask
```

Sem isto, o attention pooling incluiria frames de padding no calculo, poluindo o embedding com "lixo".

### Porque e que o CNN feature extractor fica sempre congelado?

```python
def freeze(self):
    self.wav2vec2.feature_extractor._freeze_parameters()  # CNN SEMPRE congelado
    for param in self.wav2vec2.parameters():
        param.requires_grad = False
```

O CNN do wav2vec2 aprende filtros acusticos fundamentais (semelhantes a filtros de frequencia). Estes sao extremamente genericos e uteis para qualquer tarefa de audio. Fine-tuna-los com apenas ~8.000 amostras destruiria essas features sem ganho. E uma best practice reconhecida na comunidade de speech processing.

---

## 4. Modelo de Fusao (src/models/fusion_model.py)

### O que faz

Combina os embeddings de texto e audio numa unica previsao de emocao. E o "cerebro" central que decide: dada esta frase E este tom de voz, qual e a emocao?

### Porque e que e importante

Este e o componente que torna o sistema **multimodal**. Sem ele, teriamos dois classificadores separados sem comunicacao. A fusao permite que o modelo aprenda que certas combinacoes de texto+audio sao mais indicativas de frustracao do que cada modalidade sozinha.

### Estrategia de fusao: Concat

```python
# text_emb:  [batch, 768]  -- "o que a pessoa disse"
# audio_emb: [batch, 768]  -- "como a pessoa disse"

combined = torch.cat([text_emb, audio_emb], dim=-1)  # [batch, 1536]
logits = self.classifier(combined)                     # [batch, 5]
```

A fusao por concatenacao e a mais simples: junta os dois vectores lado a lado (768+768=1536) e passa por um MLP (Multi-Layer Perceptron) que aprende a combina-los:

```python
self.classifier = nn.Sequential(
    nn.Linear(1536, 512),     # projectar de 1536 para 512
    nn.ReLU(),                # activacao nao-linear
    nn.Dropout(0.3),          # regularizacao
    nn.Linear(512, 5),        # 5 classes de saida
)
```

### Estrategia alternativa: Gated Fusion

```python
combined = torch.cat([text_emb, audio_emb], dim=-1)  # [batch, 1536]
gate = torch.sigmoid(self.gate(combined))             # [batch, 768] valores entre 0 e 1
fused = gate * text_emb + (1 - gate) * audio_emb     # [batch, 768]
logits = self.classifier(fused)                        # [batch, 5]
```

O gate e um vector de 768 dimensoes com valores entre 0 e 1. Cada dimensao decide independentemente: "para esta amostra, confio mais no texto ou no audio?" Se `gate[i] = 0.9`, a dimensao `i` usa 90% texto e 10% audio. Se `gate[i] = 0.2`, usa 20% texto e 80% audio.

Isto e util quando as modalidades tem fiabilidades diferentes. Por exemplo, se o audio esta ruidoso mas o texto e claro, o gate pode aprender a confiar mais no texto.

### Focal Loss (para o desequilibrio de classes)

```python
ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
pt = torch.exp(-ce_loss)                    # probabilidade da classe correcta
focal_loss = ((1 - pt) ** self.gamma) * ce_loss  # gamma=2.0 por defeito
```

A Cross-Entropy normal trata todos os erros igualmente. A Focal Loss adiciona um factor `(1 - pt)^gamma` que:
- Quando `pt` e alto (predicao confiante e correcta): `(1 - 0.9)^2 = 0.01` -- loss quase zero
- Quando `pt` e baixo (predicao errada): `(1 - 0.1)^2 = 0.81` -- loss alta

Resultado: o modelo foca-se nos exemplos dificeis (muitas vezes da classe minoritaria `frustration`) em vez de optimizar infinitamente os exemplos faceis de `neutral`.

---

## 5. Metricas de Avaliacao (src/evaluation/metrics.py)

### O que faz

Calcula todas as metricas de avaliacao do modelo: accuracy, F1-score (weighted e macro), precision/recall por classe, e confusion matrix.

### Porque e que e importante

Accuracy sozinha e **enganadora** com dados desequilibrados. Um modelo que diga sempre "neutral" tem 53.6% de accuracy. Precisamos de metricas que revelem o desempenho por classe, especialmente para `frustration`.

### A funcao `compute_class_weights()`

```python
weight_c = total_samples / (num_classes * count_c)
```

Com os dados do treino:
- anger:        8783 / (5 * 1380) = **1.27**
- frustration:  8783 / (5 * 268)  = **6.55**  << peso muito alto
- sadness:      8783 / (5 * 683)  = **2.57**
- neutral:      8783 / (5 * 4709) = **0.37**  << peso muito baixo
- satisfaction: 8783 / (5 * 1743) = **1.01**

Cada amostra de `frustration` "vale" 6.55x no calculo da loss, enquanto cada amostra de `neutral` vale apenas 0.37x. Isto obriga o modelo a prestar atencao a classe minoritaria.

### As metricas chave

- **Accuracy**: percentagem global de acertos. Util mas enganadora com classes desequilibradas.
- **Weighted F1**: media do F1 de cada classe, ponderada pelo numero de amostras. Da mais peso as classes frequentes.
- **Macro F1**: media simples do F1 de cada classe. Trata todas as classes igualmente -- mais justa para classes minoritarias.
- **Frustration Recall**: de todas as amostras realmente frustradas, quantas o modelo detectou? Esta e a **metrica mais importante** do sistema, porque um false negative (frustracao nao detectada) e pior que um false positive (falso alarme).

### A confusion matrix

```
           anger  frust  sadne  neutr  satis
  anger      85      3      5     10      2
  frust       2     15      3     18      2
  sadne       1      2     50     40      5
  neutr       3      1      8    430      8
  satis       2      0      3     15    140
```

Cada celula `[i,j]` diz: "quantas amostras da classe `i` foram classificadas como classe `j`". A diagonal principal sao os acertos. As outras celulas sao os erros. Isto permite ver, por exemplo, que muitas amostras de `frustration` sao classificadas como `neutral` -- informacao crucial para melhorar o modelo.

---

## 6. Pipeline de Treino (src/training/train.py)

### O que faz

E o "maestro" que orquestra tudo: carrega dados, inicializa modelos, treina, valida, guarda checkpoints, e para quando o modelo nao melhora.

### Porque e que e importante

E o ficheiro que efectivamente **executa** todo o trabalho. Todos os outros ficheiros definem componentes; este junta-os e corre o treino de ponta a ponta.

### 6.1 -- A funcao collate_fn (o elo entre dados e modelo)

```python
def collate_fn(batch, tokenizer, audio_processor, max_text_length, max_audio_samples):
    texts = [item["text"] for item in batch]
    audios = [item["audio"]["array"] for item in batch]
    labels = [item["target_label"] for item in batch]
```

O DataLoader do PyTorch agrupa amostras individuais em batches. Mas as nossas amostras sao "complexas" (texto + audio + label), e o texto/audio tem comprimentos variaveis. A `collate_fn` customizada:

1. **Extrai** o texto, audio (array numpy), e label de cada amostra
2. **Trunca** audio acima do limite (15s * 16000 = 240.000 amostras)
3. **Tokeniza** o texto com o RobertaTokenizer (padding + truncation a 128 tokens)
4. **Processa** o audio com o Wav2Vec2Processor (padding + normalizacao)
5. **Retorna** um dicionario de tensores prontos para o modelo

Sem esta funcao, o PyTorch nao saberia como juntar amostras de texto e audio com tamanhos diferentes num batch uniforme.

### 6.2 -- O optimizer com parameter groups

```python
param_groups = [
    {"params": head_params, "lr": 2e-5},            # classificador: lr normal
    {"params": encoder_params, "lr": 2e-6},          # encoders: lr 10x menor
]
optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
```

**Porque AdamW?** E o optimizer standard para fine-tuning de Transformers. Combina Adam (learning rates adaptativas por parametro) com weight decay correcto (regularizacao L2 desacoplada).

**Porque parameter groups separados?** Os encoders pre-treinados precisam de actualizacoes cautelosas (lr baixa) para nao destruir o conhecimento pre-existente. A cabeca de classificacao e treinada do zero e pode aprender mais rapido (lr alta).

**Porque se reconstroi apos unfreeze?** O AdamW so rastreia parametros que lhe foram passados na construcao. Quando desbloqueamos camadas no epoch 3, esses parametros nao existiam no optimizer original. E preciso criar um novo optimizer que inclua os parametros recem-desbloqueados.

### 6.3 -- O ciclo de treino (o coracao do ficheiro)

```python
for epoch in range(epochs):
    # Descongelar encoders no epoch certo
    if epoch == freeze_epochs:
        model.unfreeze_encoders(2)
        optimizer = build_optimizer(model, config)  # reconstruir

    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        # Forward pass com FP16
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(text_ids, text_mask, audio_vals, audio_mask)
            loss = criterion(logits, labels)
            loss = loss / accum_steps           # normalizar para acumulacao

        # Backward
        scaler.scale(loss).backward()

        # Actualizar pesos a cada 4 passos (gradient accumulation)
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
```

Passo a passo:

1. **`torch.amp.autocast`** -- converte automaticamente operacoes para FP16, reduzindo memoria e acelerando o calculo.

2. **`loss / accum_steps`** -- se acumulamos 4 micro-batches, cada loss contribui 1/4. Sem esta divisao, os gradientes acumulados seriam 4x maiores que o desejado.

3. **`scaler.scale(loss).backward()`** -- o GradScaler multiplica a loss por um factor grande antes do backward. Isto evita **underflow** em FP16 (numeros demasiado pequenos que se tornam zero). Depois, o `unscale_` desfaz a escala antes do optimizer step.

4. **`clip_grad_norm_(model.parameters(), 1.0)`** -- gradient clipping. Se algum gradiente explode (valor absurdamente grande), e cortado para norma maxima 1.0. Evita instabilidade no treino.

5. **A cada 4 passos**: unscale, clip, step, update, zero_grad. Entre passos, os gradientes vao **acumulando** (somando) no `.grad` de cada parametro.

### 6.4 -- Early stopping

```python
current_metric = val_metrics["frustration_recall"]
if current_metric > best_metric:
    best_metric = current_metric
    patience_counter = 0
    save_checkpoint(model, ...)
else:
    patience_counter += 1
    if patience_counter >= 5:
        print("Early stopping triggered.")
        break
```

Se o frustration recall no conjunto de validacao nao melhora durante 5 epochs consecutivas, o treino para. Isto evita:
- Desperdicio de tempo computacional
- Overfitting (o modelo comeca a "decorar" o treino em vez de generalizar)

O melhor modelo e guardado em `checkpoints/best_model.pt` cada vez que bate o recorde.

### 6.5 -- Checkpoint (guardar o modelo)

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "metrics": metrics,
}, "checkpoints/best_model.pt")
```

Guarda nao so os pesos do modelo mas tambem o estado do optimizer (momentos de Adam) e as metricas. Isto permite:
- **Retomar** o treino exactamente de onde parou (se o PC crashar)
- **Carregar** o melhor modelo para inferencia/teste
- **Comparar** metricas entre experiencias diferentes

---

## 7. Mapa de Dependencias entre Ficheiros

```
config.yaml
    |
    v
train.py  ----imports---->  fusion_model.py  ----imports---->  text_encoder.py
    |                            |                              audio_encoder.py
    |                            |
    +----imports---->  metrics.py
    |
    +----imports---->  load_meld.py (Fase 1)
```

- `config.yaml` nao tem dependencias -- e lido por `train.py`
- `text_encoder.py` e `audio_encoder.py` sao independentes entre si
- `fusion_model.py` importa ambos os encoders
- `metrics.py` importa `TARGET_LABELS` do `load_meld.py` (Fase 1)
- `train.py` importa tudo e orquestra o pipeline completo

---

## 8. Estimativa de VRAM para o Treino

| Componente | VRAM Estimada |
|---|---|
| Pesos RoBERTa-base (FP16) | ~240 MB |
| Pesos Wav2Vec2-base (FP16) | ~180 MB |
| Cabeca de fusao/classificacao | ~3 MB |
| Activacoes (batch=8, com gradientes) | ~2-3 GB |
| Estados do optimizer (AdamW, FP32) | ~1.6 GB |
| Gradient accumulation (4 passos) | ~0 extra (gradientes acumulam in-place) |
| **Total estimado** | **~4-5 GB** |

Sobram ~11 GB de headroom. Se quisermos aumentar o batch para 16, as activacoes dobram para ~5-6 GB, total ~8-9 GB -- ainda dentro do budget.

---

## 9. Notebook de Treino (notebooks/train.ipynb)

### O que faz

Em vez de correr o treino a partir do terminal (sem feedback visual), criamos um Jupyter Notebook que executa todo o pipeline com **graficos ao vivo** que se actualizam a cada epoch. O notebook substitui a execucao directa do `train.py` para uma experiencia interactiva.

### Porque e que e importante

1. **Visualizacao em tempo real** -- permite acompanhar se o modelo esta a aprender ou a divergir sem esperar que o treino acabe.
2. **Graficos persistentes** -- os graficos ficam guardados dentro do notebook (e como imagens em `data/processed/`), prontos para incluir em relatorios ou apresentacoes.
3. **Reprodutibilidade** -- o notebook mostra inputs, outputs e graficos tudo junto, servindo de "prova" do que aconteceu.
4. **Depuracao facil** -- se algo correr mal, pode-se inspeccionar variaveis a meio do treino em vez de reler logs.

### Como executar

```bash
jupyter notebook notebooks/train.ipynb
```

Depois, correr as celulas de cima para baixo (Shift+Enter em cada uma).

### Estrutura do notebook (celula a celula)

| Celula | O que faz |
|---|---|
| **1. Setup** | Imports de todas as bibliotecas e modulos do projecto |
| **2. Configuracao** | Le o `config.yaml` e mostra os hiperparametros |
| **3. Modelo** | Carrega RoBERTa + Wav2Vec2 para a GPU, congela encoders, mostra contagem de parametros e VRAM usada |
| **4. Dataset** | Carrega o MELD, cria DataLoaders com a collate function customizada |
| **5. Distribuicao** | Grafico de barras da distribuicao de classes + grafico dos pesos da weighted loss. Guardado em `data/processed/class_distribution.png` |
| **6. Optimizer** | Configura loss (CE ou focal), AdamW com parameter groups, scheduler de warmup, GradScaler |
| **7. TREINO** | O ciclo de treino completo. A cada epoch, actualiza 4 graficos ao vivo (explicados abaixo) |
| **8. Resultados** | Carrega o melhor modelo, avalia no conjunto de teste, mostra confusion matrix visual e grafico de F1/Recall por classe |

### Os 4 graficos ao vivo (actualizados a cada epoch)

```
+----------------------------+----------------------------+
|     Loss (Train vs Val)    |     Metricas Globais       |
|                            |  (Accuracy, W-F1, Macro-F1)|
|  Loss a descer = bom       |  Linhas a subir = bom      |
+----------------------------+----------------------------+
|   Frustration Recall       |     F1-Score por Classe    |
|   (METRICA CHAVE)          |  (5 linhas, uma por classe)|
|                            |                            |
|  Linha a subir = objectivo |  Ver qual classe melhora   |
+----------------------------+----------------------------+
```

**Grafico 1 -- Loss**: Mostra a loss de treino (vermelho) e validacao (azul) ao longo das epochs. Se a loss de treino desce mas a de validacao sobe, e sinal de overfitting.

**Grafico 2 -- Metricas Globais**: Accuracy, Weighted F1 e Macro F1. O Weighted F1 e enviesado para classes frequentes (neutral). O Macro F1 trata todas as classes igualmente -- e mais informativo para o nosso caso.

**Grafico 3 -- Frustration Recall**: A metrica pela qual fazemos early stopping. Inclui uma linha tracejada a verde a marcar o melhor valor ate ao momento. Este e o grafico mais importante de observar durante o treino.

**Grafico 4 -- F1 por Classe**: 5 linhas coloridas, uma por classe. Permite ver se o modelo esta a melhorar em todas as classes ou se esta a sacrificar uma classe para melhorar outra.

### Ficheiros gerados

| Ficheiro | Conteudo | Quando e gerado |
|---|---|---|
| `data/processed/class_distribution.png` | Distribuicao das classes + pesos | Celula 5 (antes do treino) |
| `data/processed/training_curves.png` | Os 4 graficos de treino | Actualizado a cada epoch |
| `data/processed/confusion_matrix.png` | Confusion matrix no teste | Celula 8 (apos treino) |
| `data/processed/final_results.png` | F1 e Recall por classe (barras) | Celula 8 (apos treino) |
| `checkpoints/best_model.pt` | Pesos do melhor modelo + metricas + historico | Sempre que bate recorde |

### O que observar enquanto o treino corre

- **Epochs 1-3** (encoders congelados): a loss deve descer rapidamente. O modelo aprende a usar features pre-treinadas sem as alterar.
- **Epoch 4** (unfreeze): aparece a mensagem "DESBLOQUEANDO top 2 camadas". A loss pode subir temporariamente -- e normal, os encoders estao a adaptar-se.
- **Epochs 5+**: a loss deve voltar a descer. O frustration recall deve começar a subir.
- **Early stopping**: se o frustration recall nao melhora em 5 epochs seguidas, o treino para automaticamente.

---

## 10. Como Executar (alternativa terminal)

Para quem preferir correr pelo terminal sem notebook:

```bash
python src/training/train.py
```

O script faz o mesmo que o notebook mas sem graficos -- mostra metricas em texto no terminal. O modelo e igualmente guardado em `checkpoints/best_model.pt`.
