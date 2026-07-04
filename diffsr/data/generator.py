"""学習データ（式＋観測点）の合成生成。

生成手順（SPEC.md §2.6）:
1. 事前分布からスケルトン（``C`` 入り式木）をサンプリング
2. ``C`` に実数値を割当（|c| が小さすぎる値は避ける）
3. X を一様サンプリングし y を評価
4. 棄却フィルタ: 長すぎる／変数を含まない／非有限値／|y| 過大／退化（y がほぼ定数）

すべて ``numpy.random.Generator`` 経由で、シードから決定的に再現できる。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from diffsr.config import Config
from diffsr.expressions.grammar import CONST, Vocabulary
from diffsr.expressions.sympy_bridge import make_numpy_fn
from diffsr.expressions.tokenizer import Tokenizer
from diffsr.expressions.tree import Node, sample_tree


@dataclass
class Problem:
    """1つの回帰問題（正解式つき）。

    Attributes:
        skeleton: 定数を ``C`` に抽象化した正解式木。
        const_values: 生成時に ``C`` へ割り当てた実数値（前置記法順）。
        X: 観測点 (n_points, n_vars)。
        y: 目的値 (n_points,)。
        token_ids: skeleton の固定長 ID 列（モデルの学習ターゲット）。
    """

    skeleton: Node
    const_values: np.ndarray
    X: np.ndarray
    y: np.ndarray
    token_ids: np.ndarray


def _sample_constants(rng: np.random.Generator, n: int) -> np.ndarray:
    """|c| ∈ [0.2, 5] の実数定数を符号付きでサンプリングする。"""
    mag = rng.uniform(0.2, 5.0, size=n)
    sign = rng.choice([-1.0, 1.0], size=n)
    return mag * sign


def generate_problem(
    rng: np.random.Generator,
    cfg: Config,
    vocab: Vocabulary,
    tokenizer: Tokenizer,
    max_attempts: int = 200,
) -> Problem:
    """棄却サンプリングで有効な問題を1つ生成する。

    Raises:
        RuntimeError: max_attempts 回の試行で有効な問題が得られない場合
            （事前分布の設定ミスを疑うべき状況）。
    """
    lo, hi = cfg.x_range
    for _ in range(max_attempts):
        tree = sample_tree(rng, vocab, cfg.max_ops)
        tokens = tree.serialize()
        if len(tokens) > cfg.seq_len:
            continue
        if not tree.variables_used():
            continue  # 変数を含まない定数関数は回帰問題として無意味
        n_c = tree.count(CONST)
        consts = _sample_constants(rng, n_c)
        X = rng.uniform(lo, hi, size=(cfg.n_points, cfg.n_vars))
        y = make_numpy_fn(tree)(X, consts)
        if not np.all(np.isfinite(y)):
            continue
        if np.max(np.abs(y)) > cfg.y_max_abs:
            continue
        if np.std(y) < 1e-9:
            continue  # 数値的に定数（退化）
        return Problem(
            skeleton=tree,
            const_values=consts,
            X=X,
            y=y,
            token_ids=tokenizer.encode(tokens),
        )
    raise RuntimeError(f"{max_attempts} 回の試行で有効な式を生成できませんでした")


def generate_dataset(
    cfg: Config,
    vocab: Vocabulary,
    tokenizer: Tokenizer,
    n: int,
    seed: int,
) -> list[Problem]:
    """n 個の問題を決定的に生成する（seed で完全再現）。"""
    rng = np.random.default_rng(seed)
    return [generate_problem(rng, cfg, vocab, tokenizer) for _ in range(n)]
