"""スケルトン中の定数 ``C`` の BFGS マルチスタートフィッティング。

損失は MSE。定義域外（nan/inf）の予測点は大きな有限ペナルティに置換して
最適化を継続する（SPEC.md §2.5）。時間予算を超えたらリスタートを打ち切る。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from diffsr.expressions.grammar import CONST
from diffsr.expressions.sympy_bridge import make_numpy_fn
from diffsr.expressions.tree import Node

#: 非有限予測1点あたりのペナルティ（y_max_abs=1e4 の二乗を上回る大きさ）
_PENALTY = 1e9

#: 乱択に先立って必ず試す決定的な初期値（SPEC.md §2.5）
_FIXED_STARTS = [1.0, -1.0, 0.5, 2.0]


@dataclass
class FitResult:
    """定数フィッティングの結果。

    Attributes:
        constants: 最良の定数値（前置記法順）。C が無い式では空配列。
        mse: 最良定数での MSE（全予測が非有限なら _PENALTY 級の値）。
        n_restarts_used: 実際に実行したリスタート数。
    """

    constants: np.ndarray
    mse: float
    n_restarts_used: int


def _mse(y_pred: np.ndarray, y: np.ndarray) -> float:
    bad = ~np.isfinite(y_pred)
    if bad.all():
        return _PENALTY
    err = np.where(bad, 0.0, y_pred - y) ** 2
    return float(np.mean(err) + bad.mean() * _PENALTY)


def fit_constants(
    tree: Node,
    X: np.ndarray,
    y: np.ndarray,
    n_restarts: int = 8,
    time_budget_s: float = 2.0,
    seed: int = 0,
) -> FitResult:
    """スケルトンの ``C`` を BFGS マルチスタートでフィッティングする。

    Args:
        tree: ``C`` を含む（含まなくてもよい）式木。
        X: 観測点 (n, d)。
        y: 目的値 (n,)。
        n_restarts: リスタート回数の上限（固定初期値＋乱数初期値）。
        time_budget_s: この関数全体の時間予算（超過でリスタート打ち切り）。
        seed: 乱数初期値のシード。

    Returns:
        :class:`FitResult`。
    """
    n_c = tree.count(CONST)
    f = make_numpy_fn(tree)
    if n_c == 0:
        return FitResult(np.zeros(0), _mse(f(X, []), y), 0)

    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)

    def loss(c: np.ndarray) -> float:
        return _mse(f(X, c), y)

    starts = [np.full(n_c, v) for v in _FIXED_STARTS]
    while len(starts) < n_restarts:
        starts.append(rng.normal(0.0, 2.0, size=n_c))
    starts = starts[:n_restarts]

    best_c = starts[0]
    best_mse = loss(best_c)
    t0 = time.monotonic()
    used = 0
    for c0 in starts:
        if time.monotonic() - t0 > time_budget_s:
            break
        used += 1
        try:
            res = minimize(loss, c0, method="BFGS", options={"maxiter": 200})
        except Exception:
            continue
        if np.all(np.isfinite(res.x)) and res.fun < best_mse:
            best_mse = float(res.fun)
            best_c = np.asarray(res.x, dtype=float)
    return FitResult(best_c, best_mse, used)
