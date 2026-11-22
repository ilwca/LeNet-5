# Implementação do Ambiente Experimental — LeNet-5 / CIFAR-10

## 1. Contexto do projeto

Você atuará como engenheiro de software e pesquisador especializado em Machine Learning, Deep Learning e experimentação científica.

O objetivo é implementar o ambiente experimental de um Trabalho de Conclusão de Curso (TCC) intitulado:

**"Impacto da quantização e poda em redes neurais pequenas: de MLP a CNN".**

O estudo investiga técnicas de aproximação de redes neurais, especialmente:

- Quantização de pesos.
- Poda estruturada (*structured pruning*).
- Combinação de poda e quantização.
- Impacto dessas técnicas sobre o desempenho preditivo e a eficiência computacional.

A pesquisa busca compreender o trade-off entre redução de recursos computacionais e preservação da capacidade de classificação das redes neurais.

O escopo experimental deste projeto é restrito à CNN (LeNet-5) sobre o CIFAR-10. Não há implementação de MLP nem experimentos com MNIST.

**CNN — LeNet-5 / CIFAR-10**
- Implementação de uma LeNet-5 adaptada.
- Treinamento do modelo original em precisão FP32.
- Coleta do baseline.
- Posterior aplicação das técnicas de quantização e poda.
- Comparação dos resultados entre as diferentes configurações.

Neste momento, o escopo de implementação está restrito à construção e avaliação do modelo baseline.

**IMPORTANTE: NÃO IMPLEMENTAR AINDA QUANTIZAÇÃO OU PODA.**

A arquitetura original, treinada em FP32, deverá constituir a referência experimental para todas as futuras comparações.

---

## 2. Objetivo da implementação

Desenvolver um ambiente experimental modular em Python utilizando PyTorch, capaz de:

1. Carregar e preparar o CIFAR-10.
2. Construir a arquitetura LeNet-5 adaptada.
3. Treinar a rede em precisão FP32.
4. Avaliar o desempenho no conjunto de validação e teste.
5. Registrar métricas de desempenho.
6. Medir o tamanho do modelo e a latência de inferência.
7. Salvar checkpoints e históricos de treinamento.
8. Permitir a reprodução dos experimentos por meio de configurações e sementes aleatórias controladas.

O projeto deverá ser preparado para receber posteriormente módulos independentes de quantização e poda, sem exigir uma reestruturação significativa do código.

---

## 3. Tecnologias

Utilizar:

- Linguagem: Python 3.11 ou versão compatível com as dependências.
- Framework de Deep Learning: PyTorch.
- Manipulação de dados: torchvision e NumPy.
- Visualização: Matplotlib.
- Configuração: YAML.
- Testes: pytest.
- Interface CLI: argparse.
- Registro de experimentos: arquivos CSV e JSON.

### Hardware

O ambiente deverá utilizar GPU NVIDIA com CUDA sempre que disponível.

Implementar seleção automática do dispositivo:

- CUDA disponível: utilizar GPU NVIDIA.
- CUDA indisponível: utilizar CPU.

A seleção deverá ser centralizada em um módulo de configuração do ambiente.

Registrar informações sobre:
- Nome do dispositivo.
- Modelo da GPU, quando disponível.
- Versão do CUDA.
- Versão do PyTorch.
- Versão do Python.
- Sistema operacional.

Não presumir que CUDA está disponível apenas porque existe uma GPU NVIDIA instalada.

---

## 4. Arquitetura da rede — LeNet-5 adaptada

Implementar uma versão adaptada da LeNet-5 para classificação das imagens RGB do CIFAR-10.

A arquitetura deverá preservar a estrutura geral da LeNet-5, substituindo as operações de ativação por ReLU e utilizando MaxPool2d.

### Especificação arquitetural

| Camada | Operação | Configuração | Saída |
|---|---|---|---|
| Input | Entrada | RGB, 32×32 | 3×32×32 |
| C1 | Conv2d | 6 filtros, kernel 5×5 | 6×28×28 |
| Ativação | ReLU | — | 6×28×28 |
| S1 | MaxPool2d | Kernel 2×2, stride 2 | 6×14×14 |
| C2 | Conv2d | 16 filtros, kernel 5×5 | 16×10×10 |
| Ativação | ReLU | — | 16×10×10 |
| S2 | MaxPool2d | Kernel 2×2, stride 2 | 16×5×5 |
| C3 | Conv2d | 120 filtros, kernel 5×5 | 120×1×1 |
| Ativação | ReLU | — | 120×1×1 |
| Flatten | Vetorização | — | 120 |
| FC1 | Linear | 120 → 84 | 84 |
| Ativação | ReLU | — | 84 |
| FC2 | Linear | 84 → 10 | 10 |

### Regras arquiteturais

- Entrada obrigatória: três canais RGB.
- Não utilizar padding nas convoluções.
- Não utilizar BatchNorm ou Dropout nesta versão inicial.
- Não adicionar camadas ou filtros além dos especificados.
- Utilizar `nn.ReLU`.
- Utilizar `nn.MaxPool2d`.
- A camada de saída deverá retornar logits.
- Não aplicar Softmax dentro do modelo, pois será utilizada `CrossEntropyLoss`, que já incorpora a operação necessária.

A implementação deverá permitir inspecionar as dimensões intermediárias e verificar automaticamente se correspondem às dimensões esperadas.

O modelo deverá possuir uma classe `LeNet5`, preferencialmente em `models/lenet.py`.

Documentar que esta implementação é uma adaptação da LeNet-5 original, e não uma reprodução integral de todas as suas operações históricas.

---

## 5. Dataset — CIFAR-10

Utilizar o CIFAR-10 disponibilizado por `torchvision.datasets.CIFAR10`.

Características:
- 50.000 imagens de treinamento.
- 10.000 imagens de teste.
- 10 classes.
- Imagens RGB de 32×32 pixels.

### Divisão dos dados

Dividir o conjunto original de treinamento em:

| Conjunto | Quantidade | Finalidade |
|---|---:|---|
| Treinamento | 45.000 | Atualização dos pesos |
| Validação | 5.000 | Seleção de hiperparâmetros |
| Teste | 10.000 | Avaliação final |

A divisão deverá ser estratificada, preservando a proporção das classes.

A separação deverá ser reproduzível por meio de uma semente aleatória.

O conjunto de teste não poderá ser utilizado durante o treinamento, seleção de hiperparâmetros ou seleção do melhor checkpoint.

### Pré-processamento

- Converter as imagens para tensores.
- Aplicar normalização por canal RGB.
- Calcular média e desvio padrão exclusivamente a partir dos dados de treinamento.
- Utilizar os mesmos parâmetros de normalização em validação e teste.

Não implementar data augmentation nesta primeira configuração baseline. A prioridade é estabelecer uma referência controlada e reproduzível.

Implementar o processamento em `src/dataset.py`.

---

## 6. Configuração inicial de treinamento

Criar um arquivo `configs/baseline.yaml` com os seguintes parâmetros iniciais:

```yaml
experiment:
  name: lenet5_cifar10_baseline
  seed: 42
  repetitions: 5

model:
  architecture: lenet5
  input_channels: 3
  num_classes: 10
  precision: float32

dataset:
  name: cifar10
  train_size: 45000
  validation_size: 5000
  test_size: 10000
  normalize: true
  augmentation: false

training:
  epochs: 50
  batch_size: 128
  optimizer: adam
  learning_rate: 0.001
  weight_decay: 0.0005
  loss: cross_entropy
  scheduler: reduce_on_plateau
  early_stopping: false

evaluation:
  batch_size: 128
  latency_batch_sizes: [1, 128]
  warmup_iterations: 50
  benchmark_iterations: 200

environment:
  device: auto
  precision: float32
```

Os parâmetros representam uma configuração inicial candidata, não resultados científicos previamente validados.

Permitir alterar esses parâmetros sem modificar o código-fonte.

O scheduler deverá possuir seus parâmetros de configuração explicitamente definidos e documentados.

---

## 7. Treinamento

Implementar o treinamento em `src/train.py`.

Requisitos:

1. Receber o modelo, os DataLoaders e a configuração.
2. Executar treinamento por épocas.
3. Calcular a perda de treinamento.
4. Calcular a acurácia de treinamento.
5. Avaliar a perda e acurácia de validação ao final de cada época.
6. Atualizar o scheduler de acordo com a perda de validação.
7. Registrar os resultados de cada época.
8. Salvar o melhor checkpoint com base na perda de validação.
9. Permitir retomar uma execução a partir de um checkpoint.

O treinamento deverá utilizar:

- `model.train()`.
- `torch.no_grad()` durante avaliação.
- `optimizer.zero_grad()`.
- `loss.backward()`.
- `optimizer.step()`.

Utilizar FP32 explicitamente nesta etapa.

Não implementar mixed precision, quantização automática ou outros mecanismos que alterem a precisão numérica do baseline.

### Reprodutibilidade

Implementar uma função centralizada para definir:

- Semente Python.
- Semente NumPy.
- Semente PyTorch.
- Semente CUDA.
- Geradores dos DataLoaders.

Utilizar configurações determinísticas quando suportadas, documentando possíveis impactos no desempenho e na reprodutibilidade entre diferentes dispositivos.

Cada repetição experimental deverá possuir um identificador próprio.

---

## 8. Avaliação e coleta de métricas

Implementar `src/evaluate.py`.

As avaliações deverão ocorrer separadamente para validação e teste.

Registrar as seguintes métricas:

### 8.1 Desempenho preditivo

- Acurácia.
- Loss.
- Precisão, recall e F1-score macro, como métricas complementares.
- Matriz de confusão.

A acurácia será a principal métrica de comparação.

### 8.2 Complexidade do modelo

Registrar:

- Quantidade total de parâmetros.
- Quantidade de parâmetros treináveis.
- Tamanho do checkpoint em bytes.
- Tamanho do checkpoint em MB.
- Precisão numérica utilizada.

Não confundir tamanho do arquivo serializado com o tamanho teórico dos parâmetros na memória.

### 8.3 Latência de inferência

Implementar `src/benchmark.py`.

Medir a latência em dois cenários:

- Batch size 1: inferência individual.
- Batch size 128: inferência em lote.

Procedimento:

1. Carregar o modelo treinado.
2. Colocá-lo em modo `eval()`.
3. Desabilitar gradientes.
4. Realizar aquecimento antes da medição.
5. Executar as inferências de benchmark.
6. Medir o tempo de execução.
7. Registrar média, mediana, desvio padrão e percentis 95 e 99.

Quando utilizar CUDA, sincronizar o dispositivo antes e depois da região cronometrada, para evitar medições incorretas devido à execução assíncrona.

Não incluir carregamento do modelo, transferência inicial dos dados ou pré-processamento externo na medição.

Registrar o dispositivo e o batch size utilizado.

Apresentar também a latência por imagem para os testes em lote, distinguindo-a da latência total do lote.

---

## 9. Arquitetura de diretórios

Criar a seguinte estrutura:

```text
lenet-experiment/
│
├── data/
│   └── .gitkeep
│
├── models/
│   ├── __init__.py
│   └── lenet.py
│
├── experiments/
│   └── baseline/
│
├── checkpoints/
│   └── .gitkeep
│
├── results/
│   ├── metrics/
│   ├── plots/
│   └── logs/
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── utils.py
│
├── configs/
│   └── baseline.yaml
│
├── tests/
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_training.py
│   └── test_metrics.py
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── pyproject.toml
```

### Responsabilidades

**models/**
- Implementação das arquiteturas.
- Definição dos módulos da rede neural.

**src/dataset.py**
- Download e carregamento do CIFAR-10.
- Divisão dos conjuntos.
- Normalização.
- DataLoaders.

**src/train.py**
- Loop de treinamento.
- Avaliação por época.
- Salvamento de checkpoints.

**src/evaluate.py**
- Métricas preditivas.
- Métricas de complexidade.
- Matriz de confusão.

**src/benchmark.py**
- Benchmark de inferência.
- Medição de latência.
- Estatísticas de execução.

**src/utils.py**
- Seleção de dispositivo.
- Sementes aleatórias.
- Leitura de configurações.
- Registro de ambiente.
- Funções auxiliares.

**main.py**
- Interface principal da aplicação.
- Orquestração das etapas experimentais.

**configs/**
- Configurações declarativas dos experimentos.

**tests/**
- Testes automatizados da arquitetura e do fluxo experimental.

---

## 10. Resultados e persistência

Cada execução deverá criar um diretório próprio dentro de `experiments/baseline/`, identificado por nome e timestamp.

Exemplo:

```text
experiments/
└── baseline/
    ├── run_2026-11-22_seed42/
    │   ├── config.yaml
    │   ├── environment.json
    │   ├── history.csv
    │   ├── metrics.json
    │   ├── benchmark.json
    │   ├── checkpoint_best.pth
    │   └── plots/
    │
    ├── run_2026-11-22_seed123/
    │   └── ...
    │
    └── summary.csv
```

Registrar os seguintes resultados:

- Histórico de perda de treinamento.
- Histórico de perda de validação.
- Histórico de acurácia de treinamento.
- Histórico de acurácia de validação.
- Métricas finais de teste.
- Quantidade de parâmetros.
- Tamanho do checkpoint.
- Resultados de benchmark.
- Configuração utilizada.
- Informações do ambiente computacional.

Criar gráficos:

1. Loss de treinamento e validação por época.
2. Acurácia de treinamento e validação por época.
3. Matriz de confusão.
4. Comparação das métricas entre as cinco repetições.

Ao final das cinco execuções, gerar uma tabela de resumo contendo média e desvio padrão das métricas principais.

Não calcular estatísticas agregadas enquanto as cinco execuções não estiverem concluídas.

---

## 11. Interface CLI

Implementar uma interface simples e documentada.

Exemplos desejados:

```bash
# Executar treinamento baseline
python main.py train --config configs/baseline.yaml

# Executar avaliação
python main.py evaluate --run-id RUN_ID

# Executar benchmark
python main.py benchmark --run-id RUN_ID

# Executar todo o pipeline
python main.py run-all --config configs/baseline.yaml

# Executar testes
pytest -v
```

O comando `run-all` deverá executar as repetições configuradas, treinar, avaliar, medir a latência e consolidar os resultados.

Evitar repetir treinamentos desnecessariamente quando já existirem execuções completas. Não sobrescrever resultados anteriores sem solicitação explícita.

---

## 12. Testes e validação

Implementar testes automatizados para verificar:

### Arquitetura
- Entrada com dimensões `[B, 3, 32, 32]`.
- Saída com dimensões `[B, 10]`.
- Dimensões intermediárias corretas.
- Quantidade de parâmetros.
- Compatibilidade com CPU.
- Compatibilidade com CUDA quando disponível.

### Dataset
- Quantidade de amostras.
- Divisão correta entre treinamento, validação e teste.
- Correspondência entre quantidade de classes e rótulos.
- Normalização.
- Reprodutibilidade da divisão.

### Treinamento
- Execução de uma época com pequeno subconjunto de dados.
- Atualização dos parâmetros.
- Cálculo da loss.
- Salvamento e carregamento do checkpoint.

### Métricas
- Cálculo correto da acurácia.
- Cálculo da quantidade de parâmetros.
- Registro do tamanho do checkpoint.
- Execução do benchmark.

Os testes não deverão exigir o download integral do CIFAR-10 nem uma GPU para serem executados.

---

## 13. Requisitos científicos e metodológicos

Esta implementação deverá respeitar os seguintes princípios:

**Reprodutibilidade**
- Todas as configurações devem ser explícitas.
- Registrar as sementes utilizadas.
- Preservar os resultados de cada execução.
- Registrar versões de bibliotecas e hardware.

**Comparabilidade**
- Manter a mesma arquitetura de referência.
- Utilizar o mesmo conjunto de teste.
- Manter condições de avaliação consistentes.
- Não alterar silenciosamente as configurações entre experimentos.

**Integridade experimental**
- Não utilizar o conjunto de teste para seleção de hiperparâmetros.
- Não apresentar resultados estimados como resultados efetivamente medidos.
- Não descartar execuções com resultados desfavoráveis sem justificativa metodológica.
- Diferenciar métricas medidas de métricas teóricas.
- Não afirmar que uma rede é mais eficiente apenas por possuir menos parâmetros.

**Separação de responsabilidades**
- O treinamento não deve conter lógica de quantização.
- O treinamento não deve conter lógica de poda.
- As métricas devem ser implementadas independentemente da arquitetura.
- O projeto deverá permitir adicionar futuramente módulos de compressão sem modificar o baseline original.

---

## 14. Referências e fundamentação

Considerar como referências de contextualização do projeto:

1. LeCun et al. (1998) — Gradient-Based Learning Applied to Document Recognition. Fundamentação histórica da LeNet.

2. Liang et al. (2021) — Pruning and Quantization for Deep Neural Network Acceleration: A Survey. Fundamentação das técnicas de poda, quantização e avaliação de redes comprimidas.

3. Mohd e Yousef (2025) — Energy-Efficient CNNs on FPGA via Convolutional Weight Quantization. Referência complementar sobre LeNet-5 e experimentação com quantização de pesos em CNNs.

As configurações deste projeto devem ser documentadas como decisões experimentais próprias. Não atribuir aos artigos hiperparâmetros ou procedimentos que não tenham sido efetivamente apresentados por seus autores.

---

## 15. Instruções de execução para o agente

Siga esta ordem de trabalho:

**Fase 1 — Inspeção**
- Inspecione o diretório atual.
- Identifique arquivos e configurações existentes.
- Verifique Python, PyTorch e disponibilidade CUDA.
- Não sobrescreva arquivos existentes sem antes entender sua finalidade.

**Fase 2 — Estruturação**
- Criar a estrutura de diretórios.
- Configurar dependências.
- Implementar a configuração YAML.
- Criar a seleção automática de dispositivo.

**Fase 3 — Implementação**
- Implementar a LeNet-5.
- Implementar o dataset.
- Implementar treinamento.
- Implementar avaliação.
- Implementar benchmark.
- Implementar interface CLI.

**Fase 4 — Validação**
- Executar os testes automatizados.
- Verificar dimensões e parâmetros da rede.
- Realizar um treinamento piloto.
- Validar o salvamento e carregamento dos resultados.
- Validar o benchmark.

**Fase 5 — Documentação**
- Documentar instalação.
- Documentar execução.
- Documentar configuração.
- Documentar estrutura do projeto.
- Explicar as decisões metodológicas.
- Apresentar exemplos de utilização.

### Restrições

- Não implementar poda.
- Não implementar quantização.
- Não implementar QAT.
- Não utilizar mixed precision.
- Não alterar a arquitetura definida.
- Não adicionar data augmentation.
- Não modificar o conjunto de teste.
- Não inventar resultados experimentais.
- Não executar cinco treinamentos completos sem antes validar o pipeline com um treinamento piloto.

### Entrega esperada

Ao finalizar, apresentar:

1. Estrutura de diretórios criada.
2. Arquivos implementados.
3. Dependências necessárias.
4. Testes executados e seus resultados.
5. Verificação da arquitetura.
6. Verificação da disponibilidade CUDA.
7. Resultado do treinamento piloto, caso executado.
8. Instruções para iniciar as cinco execuções definitivas.
9. Pendências ou limitações identificadas.

**Critério de conclusão:** o ambiente deverá ser capaz de executar integralmente o pipeline de treinamento, avaliação e coleta de métricas da LeNet-5 FP32 sobre CIFAR-10, com resultados persistidos e reproduzíveis.

Não avançar para a implementação das técnicas de aproximação. O objetivo desta entrega é estabelecer um baseline confiável para o desenvolvimento experimental subsequente.
