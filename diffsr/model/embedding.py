"""埋め込み層: 数値（観測点）・トークン位置・拡散時刻。

数値の埋め込み（SPEC.md §2.3 の簡易版）: 各スカラー v を
``[clip(v/5, ±10), sign(v), log1p(|v|)]`` の3特徴に展開してから線形層に
通す。生の値だけよりスケールの大きい y（例 |y|~1e4）でも勾配が壊れない。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def scalar_features(v: torch.Tensor) -> torch.Tensor:
    """スカラーテンソル (..., d) → 特徴 (..., 3d)。"""
    a = torch.clamp(v / 5.0, -10.0, 10.0)
    b = torch.sign(v)
    c = torch.log1p(v.abs())
    return torch.cat([a, b, c], dim=-1)


class PointEmbedding(nn.Module):
    """観測点 (x_1..x_d, y) → d_model ベクトル。

    Args:
        d_in: 1点あたりの生スカラー数（変数数 + 1）。
        d_model: 出力次元。
    """

    def __init__(self, d_in: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d_in, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points: (B, n, d_in) → (B, n, d_model)"""
        return self.net(scalar_features(points))


class TimeEmbedding(nn.Module):
    """拡散時刻 t ∈ [0,1] の正弦波埋め込み＋MLP。

    Args:
        d_model: 出力次元。
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, dtype=torch.float32) / max(half - 1, 1)
        )
        self.register_buffer("freqs", freqs)
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) → (B, d_model)"""
        ang = t[:, None] * self.freqs[None, :] * 1000.0
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < self.d_model:  # d_model が奇数の場合のパディング
            emb = torch.nn.functional.pad(emb, (0, self.d_model - emb.shape[-1]))
        return self.net(emb)
