import torch.nn as nn
import torch
import einops
from src.model.Perciever.Layers.Attention import (
    MultiHeadCrossAttention,
    MultiHeadSelfAttention,
)
from src.model.Perciever.Layers.FCNN import ProjectionFCNN


class PercieverProcessor(nn.Module):

    def __init__(
        self,
        latent_size: int = 128,
        input_size: int = 64,
        num_perceptions: int = 128,
        num_attn_heads: int = 8,
        attention_hidden_size: int = 32,
    ):
        super().__init__()

        self.latent_size = latent_size
        self.num_perceptions = num_perceptions
        self.perception_array = nn.Parameter(
            self.num_perceptions, self.latent_size, requires_grad=True
        )
        self.cross_attention1 = MultiHeadCrossAttention(
            num_heads=num_attn_heads,
            query_sequence_size=self.latent_size,
            key_value_sequence_size=input_size,
            hidden_size=attention_hidden_size,
        )
        self.self_attention1 = MultiHeadSelfAttention(
            num_heads=num_attn_heads, input_size=latent_size, hidden_size=num_attn_heads
        )
        self.cross_attention2 = MultiHeadCrossAttention(
            num_heads=num_attn_heads,
            query_sequence_size=self.latent_size,
            key_value_sequence_size=input_size,
            hidden_size=attention_hidden_size,
        )
        self.self_attention2 = MultiHeadSelfAttention(
            num_heads=num_attn_heads, input_size=latent_size, hidden_size=num_attn_heads
        )
        self.projection1 = ProjectionFCNN(self.latent_size, self.latent_size)
        self.projection2 = ProjectionFCNN(self.latent_size, self.latent_size)
        self.projection3 = ProjectionFCNN(self.latent_size, self.latent_size)
        self.projection4 = ProjectionFCNN(self.latent_size, self.latent_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.cross_attention1(self.perception_array, input)
        output = self.projection1(output) + output
        output = self.self_attention1(output, output)
        output = self.projection2(output) + output
        output = self.cross_attention2(output, input)
        output = self.projection3(output) + output
        output = self.self_attention2(output, output)
        output = self.projection4(output) + output
        return output
