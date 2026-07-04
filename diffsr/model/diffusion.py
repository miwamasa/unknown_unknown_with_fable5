"""吸収状態（マスク型）離散拡散の学習目的とサンプラ。

前向き過程（SPEC.md §2.2）: 時刻 t ∈ (0,1] で各トークンを独立に確率 t で
``[MASK]`` に置換する（線形スケジュール）。t=1 で全マスク。

学習目的: マスクされた位置の元トークンに対するクロスエントロピー。
MDLM 系の重み（1/t）は使わず一様重みとする簡略化を採る
（根拠となる原典が本環境から閲覧不能のため。EXPERIMENTS.md に記録）。

逆過程: 全 ``[MASK]`` から開始し、S ステップで「残マスク数の目標」を
線形に減らしながら、確信度（予測確率の最大値）の高い位置から順に
アンマスクする（confidence-based unmasking）。トークン値は温度付き
softmax からサンプリングする。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffsr.config import Config
from diffsr.expressions.grammar import Vocabulary
from diffsr.model.denoiser import Denoiser
from diffsr.model.encoder import DataEncoder


class DiffSRModel(nn.Module):
    """データエンコーダ＋デノイザを束ねた本体。

    ``points=None`` のとき（無条件モデル、M4 の検証用）は学習可能な
    ヌルメモリ1トークンを条件として使う。

    Args:
        cfg: 実験 config。
        vocab: 語彙。
    """

    def __init__(self, cfg: Config, vocab: Vocabulary) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = DataEncoder(
            d_in=cfg.n_vars + 1,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers_enc,
            encoder_type=cfg.encoder_type,
        )
        self.denoiser = Denoiser(
            vocab_size=len(vocab),
            seq_len=cfg.seq_len,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers_dec,
        )
        self.null_memory = nn.Parameter(torch.zeros(1, cfg.d_model))

    def encode(self, points: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        """観測点 → 条件メモリ。points=None なら無条件（ヌルメモリ）。"""
        if points is None:
            return self.null_memory[None].expand(batch_size, -1, -1)
        return self.encoder(points)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, points: torch.Tensor | None
    ) -> torch.Tensor:
        memory = self.encode(points, x_t.shape[0])
        return self.denoiser(x_t, t, memory)


class MaskedDiffusion:
    """学習損失と逆過程サンプリングの実装（モデル本体は持たない）。

    Args:
        vocab: 語彙（mask_id / pad_id を参照）。
        cfg: 実験 config（seq_len, diffusion_steps, temperature）。
    """

    def __init__(self, vocab: Vocabulary, cfg: Config) -> None:
        self.vocab = vocab
        self.cfg = cfg

    def corrupt(
        self, x0: torch.Tensor, t: torch.Tensor, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """前向き過程: x0 (B,L) を時刻 t (B,) でマスクする。

        Returns:
            (x_t, mask): mask==True の位置が [MASK] に置換されている。
        """
        u = torch.rand(x0.shape, generator=generator, device=x0.device)
        mask = u < t[:, None]
        x_t = torch.where(mask, torch.full_like(x0, self.vocab.mask_id), x0)
        return x_t, mask

    def loss(
        self,
        model: DiffSRModel,
        x0: torch.Tensor,
        points: torch.Tensor | None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """マスク位置のみのクロスエントロピー損失。"""
        B = x0.shape[0]
        # t=0 付近はマスクが空になり損失が定義できないため下限を切る
        t = torch.rand(B, generator=generator, device=x0.device) * 0.98 + 0.02
        x_t, mask = self.corrupt(x0, t, generator)
        if not mask.any():  # 極端に低い t ばかり引いた場合の保険
            mask[:, 0] = True
            x_t[:, 0] = self.vocab.mask_id
        logits = model(x_t, t, points)
        return F.cross_entropy(logits[mask], x0[mask])

    @torch.no_grad()
    def sample(
        self,
        model: DiffSRModel,
        memory: torch.Tensor,
        n_samples: int,
        generator: torch.Generator | None = None,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """逆過程: 全 [MASK] から n_samples 本の系列を生成する。

        Args:
            model: 学習済みモデル。
            memory: 条件メモリ (1, m, d) または (n_samples, m, d)。
            n_samples: 生成本数（memory が1件なら複製する）。
            generator: torch 乱数生成器（再現性用）。
            temperature: softmax 温度。None なら cfg.temperature。

        Returns:
            (n_samples, L) の ID 列。[MASK] は残らない。
        """
        cfg = self.cfg
        temp = cfg.temperature if temperature is None else temperature
        L, S = cfg.seq_len, cfg.diffusion_steps
        device = memory.device
        if memory.shape[0] == 1:
            memory = memory.expand(n_samples, -1, -1)
        x = torch.full((n_samples, L), self.vocab.mask_id, dtype=torch.long, device=device)
        masked = torch.ones(n_samples, L, dtype=torch.bool, device=device)

        for s in range(S, 0, -1):
            t = torch.full((n_samples,), s / S, device=device)
            logits = model.denoiser(x, t, memory)
            logits[..., self.vocab.mask_id] = -torch.inf  # MASK を生成しない
            probs = F.softmax(logits / temp, dim=-1)
            # 温度付きサンプリング（全位置分を一括で引く）
            flat = probs.view(-1, probs.shape[-1])
            drawn = torch.multinomial(flat, 1, generator=generator).view(n_samples, L)
            conf = probs.gather(-1, drawn[..., None])[..., 0]  # 引いたトークンの確率

            # 残マスク数の目標（線形スケジュール）。最終ステップで必ず 0。
            target_remaining = int(round(L * (s - 1) / S))
            for i in range(n_samples):
                idx = torch.nonzero(masked[i], as_tuple=True)[0]
                n_unmask = max(len(idx) - target_remaining, 0)
                if n_unmask == 0:
                    continue
                order = torch.argsort(conf[i, idx], descending=True)
                chosen = idx[order[:n_unmask]]
                x[i, chosen] = drawn[i, chosen]
                masked[i, chosen] = False
        return x
