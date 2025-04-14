import numpy as np
import torch
import torch.nn as nn


def get_2d_sincos_pos_encoding(
    height, width, embed_dim, temperature=10000.0, dtype=torch.float32
):
    """
    Create 2D sinusoidal positional embeddings for transformers.

    Args:
        height (int): Height of the 2D grid
        width (int): Width of the 2D grid
        embed_dim (int): Embedding dimension (must be divisible by 4)
        temperature (float): Temperature parameter for the encoding
        dtype (torch.dtype): Data type of the output tensor

    Returns:
        torch.Tensor: Position encodings of shape (height*width, embed_dim)
    """
    if embed_dim % 4 != 0:
        raise ValueError(f"Embedding dimension {embed_dim} must be divisible by 4")

    # Create position coordinates
    y_pos = torch.arange(height, dtype=dtype).reshape(-1, 1)
    x_pos = torch.arange(width, dtype=dtype).reshape(1, -1)

    # Compute the dimension indices for the encodings
    # Half the dimensions for height, half for width
    half_dim = embed_dim // 2
    dim_t = torch.arange(0, half_dim, 2, dtype=dtype)

    # Apply the temperature scaling
    dim_t = temperature ** (2 * dim_t / half_dim)

    # Create position encodings for height and width dimensions separately
    pos_y = y_pos / dim_t.reshape(1, -1)
    pos_x = x_pos / dim_t.reshape(1, -1)

    # Apply sine and cosine
    pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=2).flatten(2)
    pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=2).flatten(2)

    # Combine height and width encodings
    pos = torch.cat(
        (
            pos_y.repeat(1, width, 1).reshape(height * width, half_dim),
            pos_x.repeat(height, 1, 1).reshape(height * width, half_dim),
        ),
        dim=1,
    )

    return pos


class PositionalEncoding2D(nn.Module):
    """
    2D Positional Encoding module for Vision Transformers and similar architectures.
    """

    def __init__(self, height, width, embed_dim, dropout=0.1, temperature=10000.0):
        """
        Initialize the 2D positional encoding.

        Args:
            height (int): Height of the input grid
            width (int): Width of the input grid
            embed_dim (int): Embedding dimension (must be divisible by 4)
            dropout (float): Dropout probability
            temperature (float): Temperature parameter for the encoding
        """
        super().__init__()
        self.height = height
        self.width = width
        self.embed_dim = embed_dim

        # Create fixed positional encodings
        pos_embed = get_2d_sincos_pos_encoding(
            height=height, width=width, embed_dim=embed_dim, temperature=temperature
        )
        self.register_buffer("pos_embed", pos_embed)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Add positional encoding to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, height*width, embed_dim)

        Returns:
            torch.Tensor: Output with positional encoding added
        """
        # Add positional encoding
        x = x + self.pos_embed
        return self.dropout(x)


class ViTPositionalEncoding(nn.Module):
    """
    Learnable positional encoding for Vision Transformers.
    This implementation follows the original ViT approach with learnable parameters.
    """

    def __init__(self, height, width, embed_dim, dropout=0.1):
        """
        Initialize the learnable positional encoding.

        Args:
            height (int): Height of the input grid
            width (int): Width of the input grid
            embed_dim (int): Embedding dimension
            dropout (float): Dropout probability
        """
        super().__init__()
        self.height = height
        self.width = width

        # Create learnable positional embeddings
        # Add 1 for the [CLS] token if needed
        self.pos_embed = nn.Parameter(torch.zeros(1, height * width + 1, embed_dim))

        # Initialize using truncated normal distribution
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Add positional encoding to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embed_dim)
                              where sequence_length = height*width + 1 (for [CLS] token)

        Returns:
            torch.Tensor: Output with positional encoding added
        """
        # Add positional encoding
        x = x + self.pos_embed
        return self.dropout(x)


# Example usage with a patch embedding from an image
class PatchEmbedding(nn.Module):
    """
    Convert an image into patches and embed them.
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2

        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (batch_size, channels, height, width)

        Returns:
            torch.Tensor: Patch embeddings of shape (batch_size, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert (
            H == W == self.img_size
        ), f"Input image size ({H}*{W}) doesn't match expected size ({self.img_size}*{self.img_size})"

        # Extract patches using convolution and flatten
        x = self.projection(x)  # (B, embed_dim, grid_size, grid_size)
        x = x.flatten(2)  # (B, embed_dim, grid_size*grid_size)
        x = x.transpose(1, 2)  # (B, grid_size*grid_size, embed_dim)

        return x
