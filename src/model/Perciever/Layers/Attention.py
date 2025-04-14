import torch.nn as nn
import torch
import torch.nn.functional as F
import einops
from src.model.Perciever.Layers.FCNN import ProjectionFCNN
from src.model.Perciever.Layers.Attention import MultiHeadSelfAttention


class SelfAttentionLayer(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, skip_connection: bool):
        super().__init__()
        self.key_projection = ProjectionFCNN(input_size, hidden_size, norm=True)
        self.query_projection = ProjectionFCNN(input_size, hidden_size, norm=True)
        self.value_projection = ProjectionFCNN(input_size, input_size, norm=True)
        self.skip_connection = skip_connection

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        input_sequence: batch length channel
        """
        if self.skip_connection:
            skip = input_sequence
        values = self.value_projection(input_sequence)
        key = self.key_projection(input_sequence)
        query = self.query_projection(input_sequence)
        attn_score = einops.einsum(key, query, "b l c, b L c -> b l L")
        attn_score = F.softmax(attn_score, dim=-1)
        values = einops.einsum(attn_score, values, "b l L, b L c -> b l c")
        if self.skip_connection:
            values += skip
        return values


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_sequence_size: int,
        key_value_sequence_size: int,
        hidden_size: int,
        skip_connection: bool = True,
    ):
        super().__init__()
        self.key_projection = ProjectionFCNN(
            key_value_sequence_size, hidden_size, norm=True
        )
        self.query_projection = ProjectionFCNN(
            query_sequence_size, hidden_size, norm=True
        )
        self.value_projection = ProjectionFCNN(
            key_value_sequence_size, query_sequence_size, norm=True
        )
        self.skip_connection = skip_connection

    def forward(
        self, query_sequence: torch.Tensor, key_value_sequence: torch.Tensor
    ) -> torch.Tensor:
        query = self.query_projection(query_sequence)
        key = self.key_projection(key_value_sequence)
        value = self.value_projection(key_value_sequence)
        attn_score = einops.einsum(key, query, "b l c, b L c -> b l L")
        attn_score = F.softmax(attn_score, dim=-1)
        value = einops.einsum(attn_score, value, "b l L, b L c -> b l c")
        if self.skip_connection:
            value += query_sequence
        return value


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_heads: int,
        hidden_size: int,
        skip_connection: bool = True,
    ):
        super().__init__()
        assert (
            input_size % num_heads == 0
        ), f"Input size of {input_size} is not divisible among {num_heads} heads!"

        self.head_input_size = input_size // num_heads
        self.attention_heads = nn.ModuleList(
            [
                SelfAttentionLayer(
                    input_size=self.head_input_size,
                    hidden_size=hidden_size,
                    skip_connection=skip_connection,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, input_sequentce: torch.Tensor) -> torch.Tensor:
        head_outputs = []
        input_chunks = torch.split(input_sequentce, dim=-1)
        for attn_head, chunk in zip(self.attention_heads, input_chunks):
            head_outputs.append(attn_head(chunk))
        return torch.cat(head_outputs, dim=-1)


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        query_sequence_size: int,
        key_value_sequence_size: int,
        hidden_size: int,
        skip_connection: bool = True,
    ):
        super().__init__()
        assert (
            query_sequence_size % num_heads == 0
        ), f"Query Input size of {query_sequence_size} is not divisible among {num_heads} heads!"

        assert (
            key_value_sequence_size % num_heads == 0
        ), f"Key and Value Input size of {query_sequence_size} is not divisible among {num_heads} heads!"

        self.query_head_input_size = query_sequence_size // num_heads
        self.key_value_head_input_size = key_value_sequence_size // num_heads
        self.attention_heads = nn.ModuleList(
            [
                CrossAttention(
                    query_sequence_size=self.query_head_input_size,
                    key_value_sequence_size=self.key_value_head_input_size,
                    hidden_size=hidden_size,
                    skip_connection=skip_connection,
                )
            ]
        )

    def forward(
        self, query_sequence: torch.Tensor, key_value_sequence: torch.Tensor
    ) -> torch.Tensor:
        query_chunks = torch.split(query_sequence, dim=-1)
        key_value_chunks = torch.split(key_value_sequence, dim=-1)
        output_chunks = []
        for head, query, key_value in zip(
            self.attention_heads, query_chunks, key_value_chunks
        ):
            output_chunks.append(head(query, key_value))
        return torch.cat(output_chunks, dim=-1)
