import torch.nn as nn
import torch
import torch.nn.functional as F


class ProjectionFCNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        norm: bool = True,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        layers = []
        if norm:
            layers.append(nn.LayerNorm(in_features=in_features))

        layers.append(nn.Linear(in_features, out_features))
        layers.append(activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
