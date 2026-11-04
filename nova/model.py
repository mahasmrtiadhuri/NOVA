"""A small decoder-only Transformer trained to predict the next token."""

import torch
from torch import nn


class TinyTransformer(nn.Module):
    def __init__(
        self, vocab_size: int, *, d_model: int = 96, layers: int = 2,
        heads: int = 4, max_context: int = 8,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.max_context = max_context
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_context, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, *, return_all: bool = False) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("Expected [batch, sequence] token IDs")
        _, length = input_ids.shape
        if length < 1 or length > self.max_context:
            raise ValueError("Sequence length outside supported context")
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        mask = torch.triu(
            torch.ones(length, length, device=input_ids.device, dtype=torch.bool), diagonal=1
        )
        output = self.blocks(hidden, mask=mask, is_causal=True)
        logits = self.head(self.norm(output))
        return logits if return_all else logits[:, -1, :]

    @property
    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())
