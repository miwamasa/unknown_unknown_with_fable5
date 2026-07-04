"""比較ベースライン（SPEC.md §4.3）。

- :func:`lasso_baseline` — 特徴量辞書（多項式＋初等関数）上の Lasso。
  「線形手法で十分な問題か」の下限確認。
- :func:`gplearn_baseline` — 遺伝的プログラミング（純 Python）。
  gplearn の div/log/sqrt は protected 演算（ゼロ割り等を握り潰す）なので、
  SymPy への変換は通常の演算として近似する（記号的一致judgeには影響しうる。
  EXPERIMENTS.md に記載）。
- :func:`random_prior_baseline` — 学習なしで事前分布から k 本サンプルして
  BFGS フィットするアブレーション。提案手法との差が「条件付き拡散の寄与」。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import sympy as sp

from diffsr.config import Config
from diffsr.data.generator import _sample_constants  # noqa: F401  (再利用予定)
from diffsr.expressions.grammar import Vocabulary
from diffsr.expressions.sympy_bridge import make_numpy_fn
from diffsr.expressions.tokenizer import Tokenizer
from diffsr.expressions.tree import sample_tree
from diffsr.fit.constants import fit_constants


@dataclass
class BaselineResult:
    """ベースライン1手法の出力。

    Attributes:
        name: 手法名。
        expr: 予測式（SymPy）。得られなかった場合は None。
        y_pred_fn: ``f(X) -> y`` の予測関数（数値評価用）。
        elapsed_s: 実行時間。
    """

    name: str
    expr: sp.Expr | None
    y_pred_fn: object
    elapsed_s: float


def lasso_baseline(X: np.ndarray, y: np.ndarray, seed: int = 0) -> BaselineResult:
    """特徴量辞書 {x, x², x³, sin x, cos x, sqrt|x|, 交差項 xi*xj} 上の LassoCV。"""
    from sklearn.linear_model import LassoCV

    t0 = time.monotonic()
    n, d = X.shape
    feats: list[np.ndarray] = []
    names: list[sp.Expr] = []
    syms = [sp.Symbol(f"x{i}") for i in range(d)]
    for i in range(d):
        xi = X[:, i]
        feats += [xi, xi**2, xi**3, np.sin(xi), np.cos(xi), np.sqrt(np.abs(xi))]
        names += [syms[i], syms[i] ** 2, syms[i] ** 3,
                  sp.sin(syms[i]), sp.cos(syms[i]), sp.sqrt(sp.Abs(syms[i]))]
    for i in range(d):
        for j in range(i + 1, d):
            feats.append(X[:, i] * X[:, j])
            names.append(syms[i] * syms[j])
    F = np.stack(feats, axis=1)
    scale = F.std(axis=0) + 1e-12
    model = LassoCV(cv=5, random_state=seed, max_iter=50000).fit(F / scale, y)
    coefs = model.coef_ / scale
    expr = sp.Float(model.intercept_)
    for c, nm in zip(coefs, names):
        if abs(c) > 1e-6:
            expr = expr + sp.Float(c) * nm

    def y_pred_fn(Xq: np.ndarray) -> np.ndarray:
        Fq = []
        for i in range(d):
            xi = Xq[:, i]
            Fq += [xi, xi**2, xi**3, np.sin(xi), np.cos(xi), np.sqrt(np.abs(xi))]
        for i in range(d):
            for j in range(i + 1, d):
                Fq.append(Xq[:, i] * Xq[:, j])
        return np.stack(Fq, axis=1) @ coefs + model.intercept_

    return BaselineResult("Lasso", expr, y_pred_fn, time.monotonic() - t0)


def gplearn_baseline(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    population_size: int = 1000,
    generations: int = 20,
) -> BaselineResult:
    """gplearn SymbolicRegressor（演算子集合は提案手法に揃える）。"""
    from gplearn.genetic import SymbolicRegressor

    t0 = time.monotonic()
    est = SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        function_set=("add", "sub", "mul", "div", "sin", "cos", "sqrt", "log"),
        parsimony_coefficient=0.001,
        random_state=seed,
        verbose=0,
    )
    est.fit(X, y)
    expr = _gplearn_to_sympy(str(est._program), X.shape[1])
    return BaselineResult("gplearn", expr, lambda Xq: est.predict(Xq),
                          time.monotonic() - t0)


def _gplearn_to_sympy(program: str, d: int) -> sp.Expr | None:
    """gplearn のプログラム文字列を SymPy 式に変換する（protected 演算は通常演算として近似）。"""
    local = {
        "add": lambda a, b: a + b,
        "sub": lambda a, b: a - b,
        "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b,
        "sin": sp.sin,
        "cos": sp.cos,
        "sqrt": lambda a: sp.sqrt(sp.Abs(a)),
        "log": lambda a: sp.log(sp.Abs(a)),
        "neg": lambda a: -a,
    }
    for i in range(d):
        local[f"X{i}"] = sp.Symbol(f"x{i}")
    try:
        return sp.sympify(program, locals=local)
    except Exception:
        return None


def random_prior_baseline(
    cfg: Config,
    vocab: Vocabulary,
    tokenizer: Tokenizer,
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    seed: int = 0,
    fit_time_budget_s: float = 2.0,
) -> BaselineResult:
    """事前分布から k 本サンプル→BFGS→最良選択（学習なしアブレーション）。"""
    from diffsr.expressions.sympy_bridge import complexity

    t0 = time.monotonic()
    rng = np.random.default_rng(seed)
    best = None
    best_score = float("inf")
    for _ in range(k):
        tree = sample_tree(rng, vocab, cfg.max_ops)
        if len(tree.serialize()) > cfg.seq_len or not tree.variables_used():
            continue
        fit = fit_constants(tree, X, y, time_budget_s=fit_time_budget_s, seed=seed)
        score = fit.mse + cfg.lambda_complexity * complexity(tree)
        if score < best_score:
            best_score = score
            best = (tree, fit.constants)
    if best is None:
        return BaselineResult("RandomPrior", None, lambda Xq: np.zeros(len(Xq)),
                              time.monotonic() - t0)
    tree, consts = best
    f = make_numpy_fn(tree)
    from diffsr.eval.metrics import snap_constants
    from diffsr.expressions.sympy_bridge import to_sympy

    expr = to_sympy(tree, const_values=None).subs(
        {sp.Symbol(f"c{i}"): v for i, v in enumerate(snap_constants(consts))}
    )
    return BaselineResult("RandomPrior", expr, lambda Xq: f(Xq, consts),
                          time.monotonic() - t0)
