"""LeNet-5 adaptada para CIFAR-10 (entrada RGB 32x32).

Esta implementação é uma ADAPTAÇÃO da LeNet-5 de LeCun et al. (1998), e não
uma reprodução integral de todas as suas operações históricas. Diferenças em
relação ao artigo original:

- Entrada RGB (3 canais) em vez de imagens em tons de cinza (1 canal).
- Ativações ReLU em vez da tangente hiperbólica escalada.
- Subamostragem por MaxPool2d em vez das camadas de subamostragem treináveis
  (média ponderada + bias + sigmoide).
- C3 totalmente conectada aos 16 mapas de S2 (sem a tabela de conexões
  esparsas do artigo original).
- Camada de saída linear (logits) em vez das unidades RBF euclidianas.
  O Softmax NÃO é aplicado no modelo: `nn.CrossEntropyLoss` já o incorpora.

Não há padding, BatchNorm nem Dropout.

Os módulos são nomeados (c1, s1, c2, ...) para que futuros módulos de poda e
quantização possam referenciá-los sem alterar esta classe.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

# Dimensões esperadas (sem a dimensão do batch) após cada módulo, conforme a
# especificação arquitetural do projeto.
EXPECTED_SHAPES: "OrderedDict[str, tuple[int, ...]]" = OrderedDict(
    [
        ("input", (3, 32, 32)),
        ("c1", (6, 28, 28)),
        ("relu1", (6, 28, 28)),
        ("s1", (6, 14, 14)),
        ("c2", (16, 10, 10)),
        ("relu2", (16, 10, 10)),
        ("s2", (16, 5, 5)),
        ("c3", (120, 1, 1)),
        ("relu3", (120, 1, 1)),
        ("flatten", (120,)),
        ("fc1", (84,)),
        ("relu4", (84,)),
        ("fc2", (10,)),
    ]
)

# Parâmetros por camada (pesos + bias):
#   C1: 3*6*5*5 + 6       =    456
#   C2: 6*16*5*5 + 16     =  2.416
#   C3: 16*120*5*5 + 120  = 48.120
#   FC1: 120*84 + 84      = 10.164
#   FC2: 84*10 + 10       =    850
#   Total                 = 62.006
EXPECTED_PARAMETERS = 62006


class LeNet5(nn.Module):
    """LeNet-5 adaptada: Conv(6) -> Pool -> Conv(16) -> Pool -> Conv(120) -> FC(84) -> FC(10)."""

    def __init__(self, input_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        if input_channels != 3:
            raise ValueError("A LeNet-5 adaptada exige entrada RGB (input_channels=3).")
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("c1", nn.Conv2d(input_channels, 6, kernel_size=5, padding=0)),
                    ("relu1", nn.ReLU()),
                    ("s1", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("c2", nn.Conv2d(6, 16, kernel_size=5, padding=0)),
                    ("relu2", nn.ReLU()),
                    ("s2", nn.MaxPool2d(kernel_size=2, stride=2)),
                    ("c3", nn.Conv2d(16, 120, kernel_size=5, padding=0)),
                    ("relu3", nn.ReLU()),
                ]
            )
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(120, 84)),
                    ("relu4", nn.ReLU()),
                    ("fc2", nn.Linear(84, num_classes)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna logits com shape [B, num_classes]."""
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)

    def named_layers(self) -> "list[tuple[str, nn.Module]]":
        """Sequência ordenada (nome, módulo) de todas as operações da rede."""
        layers = list(self.features.named_children())
        layers.append(("flatten", self.flatten))
        layers.extend(self.classifier.named_children())
        return layers

    @torch.no_grad()
    def trace_shapes(self, batch_size: int = 1, device: "torch.device | str | None" = None) -> "OrderedDict[str, tuple[int, ...]]":
        """Executa uma entrada fictícia e devolve a shape (sem batch) após cada operação."""
        if device is None:
            device = next(self.parameters()).device
        x = torch.zeros(batch_size, self.input_channels, 32, 32, device=device, dtype=torch.float32)
        shapes: "OrderedDict[str, tuple[int, ...]]" = OrderedDict([("input", tuple(x.shape[1:]))])
        for name, module in self.named_layers():
            x = module(x)
            shapes[name] = tuple(x.shape[1:])
        return shapes

    def expected_shapes(self) -> "OrderedDict[str, tuple[int, ...]]":
        shapes = OrderedDict(EXPECTED_SHAPES)
        shapes["fc2"] = (self.num_classes,)
        return shapes

    def verify_shapes(self) -> "OrderedDict[str, tuple[int, ...]]":
        """Verifica as dimensoes intermediarias; levanta AssertionError se divergirem."""
        observed = self.trace_shapes()
        expected = self.expected_shapes()
        mismatches = {
            name: (observed.get(name), shape)
            for name, shape in expected.items()
            if observed.get(name) != shape
        }
        if mismatches or list(observed) != list(expected):
            raise AssertionError(f"Dimensoes divergentes (observado, esperado): {mismatches}")
        return observed


def build_model(config: dict) -> LeNet5:
    """Constrói o modelo a partir da secao `model` da configuracao, em FP32."""
    model_cfg = config["model"]
    architecture = model_cfg.get("architecture", "lenet5")
    if architecture != "lenet5":
        raise ValueError(f"Arquitetura não suportada: {architecture!r}")
    if model_cfg.get("precision", "float32") != "float32":
        raise ValueError("O baseline deve ser construído em float32.")
    model = LeNet5(
        input_channels=int(model_cfg.get("input_channels", 3)),
        num_classes=int(model_cfg.get("num_classes", 10)),
    )
    return model.to(dtype=torch.float32)
