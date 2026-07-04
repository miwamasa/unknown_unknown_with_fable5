"""推論パイプライン: best-of-k サンプリング → 棄却 → 定数フィット → 選択。

SPEC.md §2.4 の手順そのまま:
1. 条件メモリを1回計算し、k 本のスケルトンをサンプリング
2. パース不能な候補を棄却（無効率を記録）
3. 各有効候補の C を BFGS でフィッティング
4. ``score = MSE + λ * complexity`` 最小の候補を返す
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from diffsr.config import Config
from diffsr.expressions.grammar import Vocabulary
from diffsr.expressions.sympy_bridge import complexity
from diffsr.expressions.tokenizer import Tokenizer
from diffsr.expressions.tree import Node, ParseError
from diffsr.fit.constants import fit_constants
from diffsr.model.diffusion import DiffSRModel, MaskedDiffusion


@dataclass
class Candidate:
    """フィッティング済みの候補式。"""

    skeleton: Node
    constants: np.ndarray
    mse: float
    complexity: int
    score: float


@dataclass
class PredictionResult:
    """1問題に対する推論の結果。

    Attributes:
        best: 最良候補（有効候補が1つも無ければ None）。
        candidates: 全有効候補（score 昇順）。
        n_sampled: サンプリングした本数 k。
        n_invalid: パース不能で棄却された本数（無効生成率 = n_invalid / k）。
    """

    best: Candidate | None
    candidates: list[Candidate] = field(default_factory=list)
    n_sampled: int = 0
    n_invalid: int = 0

    @property
    def invalid_rate(self) -> float:
        return self.n_invalid / self.n_sampled if self.n_sampled else 0.0


def predict_expression(
    model: DiffSRModel,
    diffusion: MaskedDiffusion,
    cfg: Config,
    vocab: Vocabulary,
    tokenizer: Tokenizer,
    X: np.ndarray,
    y: np.ndarray,
    k: int | None = None,
    seed: int = 0,
    fit_time_budget_s: float = 2.0,
) -> PredictionResult:
    """観測データ (X, y) から式を推論する。

    Args:
        X: (n, d)。d < cfg.n_vars なら 0 列でパディングする。
        y: (n,)。
        k: best-of-k の k（None なら cfg.k_samples）。
        seed: サンプリングと定数フィットの乱数シード。
        fit_time_budget_s: 候補1本あたりの BFGS 時間予算。
    """
    k = k or cfg.k_samples
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[1] < cfg.n_vars:  # 使わない変数は 0 で埋める（学習分布と同様に周辺化はしない簡易策）
        X = np.concatenate([X, np.zeros((X.shape[0], cfg.n_vars - X.shape[1]))], axis=1)
    y = np.asarray(y, dtype=float)

    model.eval()
    points = torch.from_numpy(np.concatenate([X, y[:, None]], axis=1)).float()[None]
    with torch.no_grad():
        memory = model.encode(points, 1)
    gen = torch.Generator().manual_seed(seed)
    ids = diffusion.sample(model, memory, n_samples=k, generator=gen)

    # 同一スケルトンの重複フィットを避ける
    seen: set[tuple[str, ...]] = set()
    candidates: list[Candidate] = []
    n_invalid = 0
    for row in ids:
        try:
            tree = tokenizer.decode_to_tree(row.numpy())
        except ParseError:
            n_invalid += 1
            continue
        key = tuple(tree.serialize())
        if key in seen:
            continue
        seen.add(key)
        fit = fit_constants(tree, X, y, time_budget_s=fit_time_budget_s, seed=seed)
        comp = complexity(tree)
        candidates.append(
            Candidate(tree, fit.constants, fit.mse, comp,
                      score=fit.mse + cfg.lambda_complexity * comp)
        )
    candidates.sort(key=lambda c: c.score)
    return PredictionResult(
        best=candidates[0] if candidates else None,
        candidates=candidates,
        n_sampled=k,
        n_invalid=n_invalid,
    )
