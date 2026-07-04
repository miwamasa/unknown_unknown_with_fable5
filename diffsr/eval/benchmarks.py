"""ベンチマーク問題の定義（SPEC.md §4.1）。

- Nguyen 1〜10（Uy et al. 2011）: 定義とサンプリング区間は文献で標準的に
  用いられるもの。**注意**: 本環境から原典を確認できないため、式・区間は
  複数のサーベイに再掲される慣用値に基づく（EXPERIMENTS.md 参照）。
- Feynman easy サブセット: SRSD/AI-Feynman 由来の式のうち、変数 ≤3 で
  本プロトタイプの演算子集合に収まるものを手で選定。**サンプリング区間は
  SRSD のものではなく本プロジェクト独自の正区間**（データセット本体を
  取得できない場合のフォールバック。取得できた場合の差し替え手順は
  EXPERIMENTS.md）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass
class BenchmarkProblem:
    """1つのベンチマーク問題。

    Attributes:
        name: 問題名（例 "Nguyen-5", "Feynman I.12.1"）。
        expr_str: 正解式（SymPy 構文、変数 x0, x1, ...）。
        n_vars: 変数の数。
        x_low / x_high: 各変数のサンプリング区間。
    """

    name: str
    expr_str: str
    n_vars: int
    x_low: float
    x_high: float

    @property
    def expr(self) -> sp.Expr:
        return sp.sympify(self.expr_str)

    def make_data(self, n_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """観測データ (X, y) を生成する。"""
        rng = np.random.default_rng(seed)
        X = rng.uniform(self.x_low, self.x_high, size=(n_points, self.n_vars))
        syms = [sp.Symbol(f"x{i}") for i in range(self.n_vars)]
        f = sp.lambdify(syms, self.expr, "numpy")
        y = np.asarray(f(*[X[:, i] for i in range(self.n_vars)]), dtype=float)
        return X, np.broadcast_to(y, (n_points,)).copy()


NGUYEN: list[BenchmarkProblem] = [
    BenchmarkProblem("Nguyen-1", "x0**3 + x0**2 + x0", 1, -1.0, 1.0),
    BenchmarkProblem("Nguyen-2", "x0**4 + x0**3 + x0**2 + x0", 1, -1.0, 1.0),
    BenchmarkProblem("Nguyen-3", "x0**5 + x0**4 + x0**3 + x0**2 + x0", 1, -1.0, 1.0),
    BenchmarkProblem("Nguyen-4", "x0**6 + x0**5 + x0**4 + x0**3 + x0**2 + x0", 1, -1.0, 1.0),
    BenchmarkProblem("Nguyen-5", "sin(x0**2)*cos(x0) - 1", 1, -1.0, 1.0),
    BenchmarkProblem("Nguyen-6", "sin(x0) + sin(x0 + x0**2)", 1, -1.0, 1.0),
    BenchmarkProblem("Nguyen-7", "log(x0 + 1) + log(x0**2 + 1)", 1, 0.0, 2.0),
    BenchmarkProblem("Nguyen-8", "sqrt(x0)", 1, 0.0, 4.0),
    BenchmarkProblem("Nguyen-9", "sin(x0) + sin(x1**2)", 2, 0.0, 1.0),
    BenchmarkProblem("Nguyen-10", "2*sin(x0)*cos(x1)", 2, 0.0, 1.0),
]

#: 物理式サブセット（変数 x0.. に置き換え済み。区間は独自の正区間 [0.5, 2]）
FEYNMAN_EASY: list[BenchmarkProblem] = [
    BenchmarkProblem("Feynman I.12.1 (F=mu*N)", "x0*x1", 2, 0.5, 2.0),
    BenchmarkProblem("Feynman I.14.3 (U=m*g*z)", "x0*x1*x2", 3, 0.5, 2.0),
    BenchmarkProblem("Feynman I.14.4 (U=k*x^2/2)", "x0*x1**2/2", 2, 0.5, 2.0),
    BenchmarkProblem("Feynman I.25.13 (V=q/C)", "x0/x1", 2, 0.5, 2.0),
    BenchmarkProblem("Feynman I.29.4 (k=omega/c)", "x0/x1", 2, 0.5, 2.0),
    BenchmarkProblem("Feynman I.39.1 (E=3*p*V/2)", "3*x0*x1/2", 2, 0.5, 2.0),
    BenchmarkProblem("Feynman I.43.31 (D=mu*k*T)", "x0*x1*x2", 3, 0.5, 2.0),
    BenchmarkProblem("Feynman II.3.24 (h=P/(4*pi*r^2))", "x0/(4*pi*x1**2)", 2, 0.5, 2.0),
    BenchmarkProblem("Feynman I.27.6 (f=1/(1/d1+n/d2))", "1/(1/x0 + x1/x2)", 3, 0.5, 2.0),
    BenchmarkProblem("Feynman I.6.20a (gauss)", "exp(-x0**2/2)/sqrt(2*pi)", 1, 0.5, 2.0),
]

SUITES: dict[str, list[BenchmarkProblem]] = {
    "nguyen": NGUYEN,
    "feynman": FEYNMAN_EASY,
}
