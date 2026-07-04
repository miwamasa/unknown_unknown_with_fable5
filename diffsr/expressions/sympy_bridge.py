"""式木 ⇔ SymPy / numpy の橋渡しと、タイムアウト付き等価判定。

- :func:`to_sympy`: 式木 → SymPy 式。定数 ``C`` は出現順に ``c0, c1, ...``
  のシンボル、または与えられた数値に置換。
- :func:`make_numpy_fn`: 式木 → ``f(X, consts) -> y`` の numpy 関数。
  定義域外は nan/inf のまま返す（呼び出し側でペナルティ処理）。
- :func:`symbolic_equivalent`: 別プロセスで simplify/equals を実行し、
  タイムアウトは「判定不能（False 扱い）」とする（SPEC.md §4.2, リスク A5）。
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Sequence

import numpy as np
import sympy as sp

from diffsr.expressions.grammar import CONST, OPERATORS
from diffsr.expressions.tree import Node


def to_sympy(tree: Node, const_values: Sequence[float] | None = None) -> sp.Expr:
    """式木を SymPy 式に変換する。

    Args:
        tree: 式木。
        const_values: 指定すると ``C`` を出現順（前置記法順）にこの数値で置換。
            None なら ``c0, c1, ...`` のシンボルにする。

    Returns:
        SymPy 式。

    Raises:
        ValueError: const_values の個数が ``C`` の出現数と合わない場合。
    """
    counter = [0]

    def build(node: Node) -> sp.Expr:
        if node.token in OPERATORS:
            args = [build(c) for c in node.children]
            return OPERATORS[node.token].sympy_fn(*args)
        if node.token == CONST:
            i = counter[0]
            counter[0] += 1
            if const_values is not None:
                return sp.Float(const_values[i])
            return sp.Symbol(f"c{i}")
        if node.token.isdigit():
            return sp.Integer(int(node.token))
        return sp.Symbol(node.token)

    n_consts = tree.count(CONST)
    if const_values is not None and len(const_values) != n_consts:
        raise ValueError(f"定数の個数が不一致: 期待 {n_consts}, 受領 {len(const_values)}")
    return build(tree)


def make_numpy_fn(tree: Node):
    """式木から ``f(X, consts) -> y`` の numpy 関数を作る。

    Args:
        tree: 式木（``C`` を ``consts`` の出現順の要素に対応させる）。

    Returns:
        ``f(X, consts)``。``X`` は shape (n, d)、``consts`` は 1 次元列。
        返り値は shape (n,) で、定義域外の点は nan/inf を含みうる。
    """

    def f(X: np.ndarray, consts: Sequence[float]) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        counter = [0]

        def ev(node: Node) -> np.ndarray:
            if node.token in OPERATORS:
                args = [ev(c) for c in node.children]
                return OPERATORS[node.token].numpy_fn(*args)
            if node.token == CONST:
                i = counter[0]
                counter[0] += 1
                return np.full(X.shape[0], float(consts[i]))
            if node.token.isdigit():
                return np.full(X.shape[0], float(node.token))
            idx = int(node.token[1:])
            return X[:, idx]

        with np.errstate(all="ignore"):
            y = ev(tree)
        return np.asarray(y, dtype=float).reshape(X.shape[0])

    return f


def complexity(tree: Node) -> int:
    """式の複雑度（＝式木のノード総数）。SPEC.md §4.2 指標3。"""
    return tree.size()


# --- タイムアウト付き等価判定 ----------------------------------------------


def _equiv_worker(s1: str, s2: str, q: mp.Queue) -> None:
    """子プロセス本体。simplify → equals の順で判定する。"""
    try:
        e1 = sp.sympify(s1)
        e2 = sp.sympify(s2)
        d = sp.expand(e1 - e2)
        if d == 0:
            q.put(True)
            return
        d = sp.simplify(d)
        if d == 0:
            q.put(True)
            return
        r = sp.Expr.equals(e1, e2)  # ランダム数値評価による判定（True/False/None）
        q.put(bool(r) if r is not None else False)
    except Exception:
        q.put(False)


def symbolic_equivalent(
    expr1: sp.Expr, expr2: sp.Expr, timeout: float = 5.0
) -> bool:
    """2つの SymPy 式が数学的に等価かをタイムアウト付きで判定する。

    別プロセスで ``expand → simplify → equals`` を試す。タイムアウト・例外は
    False（不一致扱い）を返す。これは記号的一致率を**過小評価する方向**の
    安全側の設計（SPEC.md リスク A5）。

    Args:
        expr1: 比較する式（定数は数値化済みであること）。
        expr2: 比較する式。
        timeout: 判定の上限秒数。

    Returns:
        等価と判定できれば True。判定不能・タイムアウトは False。
    """
    ctx = mp.get_context("fork")
    q: mp.Queue = ctx.Queue()
    p = ctx.Process(target=_equiv_worker, args=(sp.srepr(expr1), sp.srepr(expr2), q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return False
    try:
        return bool(q.get_nowait())
    except Exception:
        return False
