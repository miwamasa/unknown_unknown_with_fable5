"""観測データ (X, y) のエンコーダ（順序不変な条件ベクトル列を作る）。

- v1 ``"points"``: 点ごとの埋め込みに self-attention を掛け、全点の表現を
  そのままメモリとして返す（デコーダが cross-attention で参照）。
  位置埋め込みを使わないため点の順序に依存しない。
- v2 ``"pma"``: Set Transformer の PMA（Pooling by Multihead Attention）風に、
  学習可能な m 個のクエリで全点をプーリングし、固定長 m のメモリを返す。
  点数 n が大きいときにデコーダ側の計算を抑える。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from diffsr.model.embedding import PointEmbedding


class DataEncoder(nn.Module):
    """観測点集合 → 条件メモリ (B, m, d_model)。

    Args:
        d_in: 1点あたりのスカラー数（n_vars + 1）。
        d_model: 表現次元。
        n_heads: アテンションヘッド数。
        n_layers: self-attention 層数。
        encoder_type: "points" または "pma"。
        n_pooled: "pma" のときのプーリング後トークン数 m。
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        encoder_type: str = "points",
        n_pooled: int = 16,
    ) -> None:
        super().__init__()
        if encoder_type not in ("points", "pma"):
            raise ValueError(f"未知の encoder_type: {encoder_type}")
        self.encoder_type = encoder_type
        self.embed = PointEmbedding(d_in, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.attn = nn.TransformerEncoder(layer, n_layers)
        if encoder_type == "pma":
            self.queries = nn.Parameter(torch.randn(n_pooled, d_model) * 0.02)
            self.pool = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.pool_norm = nn.LayerNorm(d_model)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points: (B, n, d_in) → メモリ (B, n or m, d_model)。

        位置埋め込みを加えないため、点の並べ替えに対して出力集合は不変。
        """
        h = self.attn(self.embed(points))
        if self.encoder_type == "points":
            return h
        q = self.queries[None].expand(h.shape[0], -1, -1)
        pooled, _ = self.pool(q, h, h)
        return self.pool_norm(pooled)
