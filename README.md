# Baseline LeNet-5 FP32 / CIFAR-10

Ambiente experimental do TCC **"Impacto da quantização e poda em redes neurais pequenas: de MLP a CNN"**. Escopo: LeNet-5 (CNN) sobre CIFAR-10.

Este repositório estabelece o **baseline FP32**, a referência para todas as comparações futuras com quantização e poda. **Quantização, poda, QAT e mixed precision não estão implementadas**, de propósito.

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

O ambiente foi validado com Python 3.14.10 e PyTorch 2.15.0 (wheel `cu132`). O código requer Python ≥ 3.11. O wheel padrão do PyTorch para Linux já inclui CUDA. Para ele usar a GPU, a máquina precisa do **driver proprietário da NVIDIA**: com o driver livre `nouveau`, a GPU existe, mas `torch.cuda.is_available()` retorna `False`. Verifique com:

```bash
python main.py info
```

## Uso

```bash
python main.py info                                    # ambiente, dispositivo e verificação da arquitetura
python main.py run-all --config configs/pilot.yaml     # piloto: 3 épocas, 1 repetição (experiments/pilot/)
python main.py run-all --config configs/baseline.yaml  # 5 repetições definitivas (experiments/baseline/)

python main.py train     --config configs/baseline.yaml --seed 123   # uma repetição
python main.py train     --run-id RUN_ID                             # retoma um treinamento interrompido
python main.py evaluate  --run-id RUN_ID [--overwrite]
python main.py benchmark --run-id RUN_ID [--overwrite] [--device cpu]
python main.py summarize --config configs/baseline.yaml

pytest -v
```

`RUN_ID` é o nome do diretório da execução (por exemplo, `run_2026-11-22_132806_seed42`).

**Execuções existentes não são refeitas nem sobrescritas.** O `run-all` funciona assim:

- Pula as repetições já completas (mesma configuração e mesma semente).
- Retoma as incompletas a partir de `checkpoint_last.pth`.
- Executa apenas as etapas que faltam.

Pode ser interrompido (Ctrl+C) e reexecutado sem perda. Para forçar:

- `--new` cria novas execuções, sem apagar as antigas.
- `--overwrite` regrava métricas ou benchmark de uma execução.

`--device` sobrescreve `environment.device` (`auto | cpu | cuda | cuda:N`). Pedir `cuda` sem CUDA funcional gera **erro**, em vez de cair silenciosamente para CPU.

## Estrutura

```text
configs/baseline.yaml   Configuração do baseline (5 repetições, 50 épocas)
configs/pilot.yaml      Piloto de validação do pipeline (3 épocas, 1 repetição)
models/lenet.py         LeNet5 + verificação automática das dimensões intermediárias
src/dataset.py          Download, divisão estratificada, normalização, DataLoaders
src/train.py            Loop de treino, avaliação por época, checkpoints, retomada
src/evaluate.py         Métricas preditivas e de complexidade (independentes da arquitetura)
src/benchmark.py        Latência de inferência
src/plots.py            Gráficos
src/utils.py            Dispositivo, sementes, configuração, registro do ambiente
main.py                 CLI e orquestração
tests/                  pytest, com dados sintéticos (sem download e sem GPU)
data/                   CIFAR-10 (baixado automaticamente)
experiments/<exp>/      Uma pasta por execução + summary.csv/runs.csv/repetitions.png
results/metrics|plots   Cópias dos resumos e do gráfico entre repetições
results/logs/           Um log por execução
checkpoints/            Reservado para modelos futuros; os checkpoints do baseline ficam em cada execução
```

Conteúdo de cada execução (`experiments/baseline/run_<data>_<hora>_seed<S>/`):

| Arquivo | Conteúdo |
|---|---|
| `config.yaml` | Configuração efetiva (com a semente e o `run_id`) |
| `run_info.json` | Status, hash da configuração, impressão digital da divisão, normalização, melhor época |
| `environment.json` | Python, PyTorch, CUDA/cuDNN, GPU/CPU, SO, flags de determinismo e TF32 |
| `history.csv` | Loss/acurácia de treino e validação, LR e tempo por época |
| `checkpoint_best.pth` | Pesos da época com menor `val_loss` (artefato avaliado) |
| `checkpoint_last.pth` | Estado completo para retomada (modelo, otimizador, scheduler, RNGs) |
| `metrics.json` | Validação e teste: acurácia, loss, P/R/F1 macro e por classe, matriz de confusão; complexidade |
| `benchmark.json` | Latência para batch 1 e 128: média, mediana, desvio, p95, p99, por imagem, tempos brutos |
| `plots/` | `loss.png`, `accuracy.png`, `confusion_matrix_test.png`, `confusion_matrix_val.png` |

## Arquitetura

LeNet-5 **adaptada**, não uma reprodução integral da original (LeCun et al., 1998). As adaptações são:

- Entrada RGB.
- ReLU.
- MaxPool2d no lugar das camadas de subamostragem treináveis.
- C3 totalmente conectada.
- Saída linear (logits) no lugar das unidades RBF.
- Sem padding, BatchNorm, Dropout ou Softmax (o Softmax já está na `CrossEntropyLoss`).

| Camada | Saída | Parâmetros |
|---|---|---:|
| C1 Conv 5×5, 6 filtros + ReLU | 6×28×28 | 456 |
| S1 MaxPool 2×2/2 | 6×14×14 | 0 |
| C2 Conv 5×5, 16 filtros + ReLU | 16×10×10 | 2.416 |
| S2 MaxPool 2×2/2 | 16×5×5 | 0 |
| C3 Conv 5×5, 120 filtros + ReLU | 120×1×1 | 48.120 |
| FC1 120→84 + ReLU | 84 | 10.164 |
| FC2 84→10 (logits) | 10 | 850 |
| **Total** | | **62.006** |

## Decisões metodológicas

Os hiperparâmetros em `configs/baseline.yaml` são uma **configuração inicial candidata**, definida por este projeto. Eles não foram tirados dos artigos de referência.

- **Divisão.** Os 50.000 exemplos de treino são divididos de forma estratificada em 45.000 de treino e 5.000 de validação (500 por classe). A semente da divisão (`dataset.split_seed: 42`) é a mesma para todas as repetições, que portanto compartilham os mesmos conjuntos. As 10.000 imagens de teste não são alteradas.
- **Normalização.** Média e desvio padrão (populacional) são calculados por canal **somente nas 45.000 imagens de treino**. Os valores medidos foram mean = (0,4918; 0,4825; 0,4468) e std = (0,2471; 0,2435; 0,2617), e são aplicados também à validação e ao teste. Não há data augmentation.
- **Repetições.** As sementes são 42, 123, 456, 789 e 1024. Cada semente define a inicialização dos pesos e a ordem do embaralhamento do treino.
- **Uso do conjunto de teste.** O teste é avaliado **somente** depois do treinamento, uma única vez por execução, com `checkpoint_best.pth`. O scheduler e a escolha do melhor checkpoint usam apenas a perda de validação. A função de treino nem recebe o DataLoader de teste.
- **Otimização.** O otimizador é `torch.optim.Adam` (lr 1e-3, betas (0,9; 0,999), eps 1e-8) com `weight_decay` 5e-4. Esse weight decay é uma penalização L2 acoplada ao gradiente, não o AdamW. São 50 épocas, com batch 128 e sem early stopping.
- **Scheduler `ReduceLROnPlateau`.** Monitora a `val_loss` com:
  - `mode=min`, `factor=0.5`, `patience=5`;
  - `threshold=1e-4` (`rel`), `cooldown=0`, `min_lr=1e-6`.
- **FP32 estrito.** Não há AMP nem autocast. O **TF32 é desabilitado** explicitamente (`allow_tf32=False` e `float32_matmul_precision="highest"`). Em GPUs Ampere ou mais recentes, como a RTX 4050, o PyTorch usaria TF32 nas convoluções por padrão, o que reduziria a precisão sem aviso.
- **Determinismo.**
  - O que é fixado: sementes de Python, NumPy, PyTorch e CUDA; o gerador dedicado do DataLoader; `cudnn.deterministic=True` e `benchmark=False`; `use_deterministic_algorithms(True, warn_only=True)`; `CUBLAS_WORKSPACE_CONFIG`.
  - Custo: em GPU isso pode deixar o treino mais lento.
  - Garantia: resultados bit a bit idênticos só são esperados no **mesmo hardware e com as mesmas versões**. CPU e GPU produzem diferenças numéricas pequenas.
  - Retomada: retomar um treinamento interrompido produz os mesmos pesos que o treino contínuo (há um teste para isso).
- **Complexidade.** Três medidas de tamanho são registradas separadamente:
  - **teórico**: 62.006 parâmetros × 4 bytes = 248.024 bytes;
  - **state_dict serializado**: medido;
  - **arquivo `checkpoint_best.pth`**: medido, inclui os metadados do checkpoint.

  MB = 10⁶ bytes; também se registra MiB. Ter menos parâmetros **não** implica, por si só, maior eficiência. Para isso, use as latências medidas.
- **Latência.**
  - Preparação: o modelo é colocado em `eval()`, com `no_grad`, e a entrada sintética FP32 é alocada no dispositivo antes da medição. Carregamento do modelo, transferência de dados e pré-processamento ficam fora do tempo medido.
  - Medição: 50 iterações de aquecimento, depois 200 iterações cronometradas uma a uma com `perf_counter`. Em CUDA, há `synchronize` antes e depois de cada iteração.
  - Para o batch 128, a latência do lote e a latência por imagem (lote ÷ 128) são registradas em campos separados.
  - Em CPU, a latência sofre interferência do sistema operacional. Compare medianas e percentis, e mantenha as mesmas condições de máquina entre os experimentos.
- **Resumo.** Média e desvio padrão **amostral** (ddof = 1) só são calculados quando todas as repetições configuradas estão completas. Nenhuma execução é descartada.

## Extensão futura

As métricas (`src/evaluate.py`) e o benchmark (`src/benchmark.py`) recebem qualquer `nn.Module`. Os módulos da rede têm nomes (`features.c1`, …, `classifier.fc2`). Módulos de poda e quantização podem, portanto, ser adicionados (por exemplo em `src/compression/`) carregando `checkpoint_best.pth`, sem modificar o baseline.

## Referências

1. LeCun et al. (1998). *Gradient-Based Learning Applied to Document Recognition*.
2. Liang et al. (2021). *Pruning and Quantization for Deep Neural Network Acceleration: A Survey*.
3. Mohd e Yousef (2025). *Energy-Efficient CNNs on FPGA via Convolutional Weight Quantization*.
