"""M3 検証: 既知定数の復元・定義域エラー処理・時間予算（PLAN.md §4.1）。"""

import time

import numpy as np

from diffsr.expressions.tree import Node
from diffsr.fit.constants import fit_constants


def _linear_skeleton():
    # C*x0 + C  （前置記法: add mul C x0 C）
    return Node("add", (Node("mul", (Node("C"), Node("x0"))), Node("C")))


def test_recover_2x_plus_1():
    """y = 2x + 1 から定数 (2, 1) を 1e-3 で復元する（PLAN.md M3）。"""
    X = np.linspace(-4, 4, 50).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0
    res = fit_constants(_linear_skeleton(), X, y, seed=0)
    assert res.mse < 1e-8
    np.testing.assert_allclose(res.constants, [2.0, 1.0], atol=1e-3)


def test_recover_amplitude_of_sin():
    """y = 3.5*sin(x) から振幅を復元する。"""
    skel = Node("mul", (Node("C"), Node("sin", (Node("x0"),))))
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = 3.5 * np.sin(X[:, 0])
    res = fit_constants(skel, X, y, seed=0)
    assert res.mse < 1e-8
    np.testing.assert_allclose(res.constants, [3.5], atol=1e-3)


def test_log_domain_error_is_penalized_not_fatal():
    """log(C*x) で C<0 初期値でも発散せず、正しい C=2 に到達する。"""
    skel = Node("log", (Node("mul", (Node("C"), Node("x0"))),))
    X = np.linspace(0.5, 5, 60).reshape(-1, 1)
    y = np.log(2.0 * X[:, 0])
    res = fit_constants(skel, X, y, n_restarts=8, seed=0)
    assert np.isfinite(res.mse)
    assert res.mse < 1e-6
    np.testing.assert_allclose(res.constants, [2.0], atol=1e-2)


def test_no_constants_returns_plain_mse():
    """C を含まない式はフィッティングせず MSE を直接返す。"""
    skel = Node("mul", (Node("x0"), Node("x0")))
    X = np.linspace(-2, 2, 30).reshape(-1, 1)
    y = X[:, 0] ** 2
    res = fit_constants(skel, X, y)
    assert res.mse < 1e-12
    assert res.constants.size == 0
    assert res.n_restarts_used == 0


def test_time_budget_respected():
    """時間予算を大幅超過しない（緩い上限: budget の5倍以内）。"""
    # 定数8個の重い式を作る: sum of C*sin(C*x) を3段
    inner = Node("x0")
    skel = inner
    for _ in range(4):
        skel = Node(
            "add",
            (Node("mul", (Node("C"), Node("sin", (Node("mul", (Node("C"), Node("x0"))),)))), skel),
        )
    X = np.linspace(-3, 3, 200).reshape(-1, 1)
    y = np.sin(X[:, 0])
    t0 = time.monotonic()
    fit_constants(skel, X, y, n_restarts=50, time_budget_s=1.0, seed=0)
    assert time.monotonic() - t0 < 5.0
