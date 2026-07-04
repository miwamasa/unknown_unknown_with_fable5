"""条件付き双方向 Transformer デノイザ。

自己回帰デコーダと違い **causal mask を使わない**: 各位置は系列全体
（マスク含む）と条件メモリの両方を常に参照できる（SPEC.md §2.2 の動機）。
入力はノイズ系列 x_t（[MASK] を含む ID 列）と拡散時刻 t、出力は各位置の
語彙 logits。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diffsr.model.embedding import TimeEmbedding


class Denoiser(nn.Module):
    """x_t (B, L) と条件メモリ (B, m, d) から全位置の logits (B, L, V) を返す。

    Args:
        vocab_size: 語彙サイズ V。
        seq_len: 系列長 L（学習可能な位置埋め込みの数）。
        d_model: 表現次元。
        n_heads: ヘッド数。
        n_layers: TransformerDecoder 層数（self-attn + cross-attn）。
    """

    def __init__(
        self, vocab_size: int, seq_len: int, d_model: int, n_heads: int, n_layers: int
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        self.time_emb = TimeEmbedding(d_model)
        layer = nn.TransformerDecoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.blocks = nn.TransformerDecoder(layer, n_layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """x_t: (B, L) int64 / t: (B,) float / memory: (B, m, d) → (B, L, V)。"""
        h = self.token_emb(x_t) + self.pos_emb[None] + self.time_emb(t)[:, None, :]
        h = self.blocks(h, memory)  # causal mask なし＝全位置双方向
        return self.head(self.out_norm(h))
