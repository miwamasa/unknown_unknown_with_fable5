"""評価指標（SPEC.md §4.2）。

- :func:`expression_match` — 記号的一致（主指標）。フィッティング済み定数を
  「スナップ」（近い整数・簡単な有理数に丸め）してから SymPy 等価判定する。
  スナップなしでは BFGS の数値誤差（2.0000001 等）で一致判定が不可能なため。
- :func:`skeleton_match` — スケルトン一致（定数値を無視した構造の一致）。
  前置記法トークン列の完全一致で判定する**厳格な**指標（等価な別構造は
  不一致扱い）。expression_match の補助として併記する。
- :func:`r2_score` — 数値精度（副指標）。
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from diffsr.expressions.sympy_bridge import symbolic_equivalent, to_sympy
from diffsr.expressions.tree import Node


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """決定係数 R²。非有限な予測を含む場合は -inf を返す。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.all(np.isfinite(y_pred)):
        return float("-inf")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else float("-inf")
    return 1.0 - ss_res / ss_tot


def snap_constants(values: np.ndarray, tolerance: float = 1e-3) -> list[sp.Expr]:
    """フィット済み定数を SymPy の簡単な数に丸める。

    各値について、tolerance 以内の整数があれば整数に、なければ
    ``nsimplify``（有理数近似）を試し、失敗時は 6 桁 Float にする。
    """
    out: list[sp.Expr] = []
    for v in np.asarray(values, dtype=float):
        r = round(v)
        if abs(v - r) < tolerance:
            out.append(sp.Integer(int(r)))
            continue
        try:
            s = sp.nsimplify(round(v, 6), tolerance=tolerance, rational=True)
            # 分母が巨大な有理数は「簡単な数」ではないので Float に落とす
            if isinstance(s, sp.Rational) and abs(s.q) <= 100:
                out.append(s)
                continue
        except Exception:
            pass
        out.append(sp.Float(round(v, 6)))
    return out


def expression_match(
    pred_tree: Node,
    pred_constants: np.ndarray,
    true_expr: sp.Expr,
    timeout: float = 5.0,
) -> bool:
    """記号的一致の判定（定数スナップ → タイムアウト付き等価判定）。

    Args:
        pred_tree: 予測スケルトン。
        pred_constants: BFGS フィット済み定数。
        true_expr: 正解の SymPy 式（数値定数込み）。
        timeout: SymPy 等価判定の上限秒数（超過は不一致扱い）。
    """
    try:
        expr = to_sympy(pred_tree, const_values=None)
        subs = {sp.Symbol(f"c{i}"): v for i, v in enumerate(snap_constants(pred_constants))}
        expr = expr.subs(subs)
    except Exception:
        return False
    return symbolic_equivalent(expr, true_expr, timeout=timeout)


def skeleton_match(pred_tree: Node, true_tree: Node) -> bool:
    """スケルトンの厳格一致（前置記法トークン列の完全一致）。"""
    return pred_tree.serialize() == true_tree.serialize()
