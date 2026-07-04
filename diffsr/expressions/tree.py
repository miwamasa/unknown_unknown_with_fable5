"""式木の表現、前置記法の直列化／パース、ランダムサンプリング。

式木は不変な :class:`Node` の再帰構造。前置記法トークン列との対応は

    tree = Node("add", (Node("mul", (Node("C"), Node("x0"))), Node("C")))
    serialize(tree) == ["add", "mul", "C", "x0", "C"]   # y = c1*x0 + c2

ランダムサンプリングは Lample & Charton (2019) の unary-binary 木の考え方に
準拠した簡易版: 内部ノード数の予算を決め、演算子を事前確率で選びながら
再帰的に木を構築する（厳密な一様木サンプリングではない。EXPERIMENTS.md 参照）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

from diffsr.expressions.grammar import CONST, OPERATORS, Vocabulary


class ParseError(ValueError):
    """前置記法トークン列が構文的に無効なときに送出される。"""


@dataclass(frozen=True)
class Node:
    """式木のノード。

    Args:
        token: トークン文字列（演算子名・変数名・``C``・整数リテラル）。
        children: 子ノードのタプル。アリティと一致していなければならない。
    """

    token: str
    children: tuple["Node", ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        expected = OPERATORS[self.token].arity if self.token in OPERATORS else 0
        if len(self.children) != expected:
            raise ParseError(
                f"トークン {self.token} はアリティ {expected} ですが"
                f"子が {len(self.children)} 個あります"
            )

    def serialize(self) -> list[str]:
        """前置記法トークン列に直列化する。"""
        out = [self.token]
        for c in self.children:
            out.extend(c.serialize())
        return out

    def size(self) -> int:
        """ノード総数（＝複雑度の基本指標）。"""
        return 1 + sum(c.size() for c in self.children)

    def count(self, token: str) -> int:
        """指定トークンの出現数（定数プレースホルダの数え上げ等に使用）。"""
        return int(self.token == token) + sum(c.count(token) for c in self.children)

    def variables_used(self) -> set[str]:
        used: set[str] = set()
        if self.token.startswith("x") and self.token[1:].isdigit():
            used.add(self.token)
        for c in self.children:
            used |= c.variables_used()
        return used


def parse_prefix(tokens: Sequence[str], vocab: Vocabulary) -> Node:
    """前置記法トークン列を式木にパースする。

    Args:
        tokens: 前置記法のトークン列（PAD を含まない）。
        vocab: 語彙（アリティ判定に使用）。

    Returns:
        パースされた式木のルート。

    Raises:
        ParseError: トークンが語彙外、列が途中で尽きる、または余りが出る場合。
    """
    it = iter(tokens)

    def build() -> Node:
        try:
            token = next(it)
        except StopIteration:
            raise ParseError("トークン列が途中で尽きました（アリティ不足）") from None
        if token not in vocab.token_to_id:
            raise ParseError(f"語彙外のトークン: {token}")
        arity = vocab.arity(token)  # 特殊トークンはここで ValueError → ParseError に変換
        children = tuple(build() for _ in range(arity))
        return Node(token, children)

    try:
        root = build()
    except ValueError as e:  # vocab.arity の特殊トークンエラーを包む
        raise ParseError(str(e)) from e
    leftover = list(it)
    if leftover:
        raise ParseError(f"パース後に {len(leftover)} 個のトークンが余りました: {leftover}")
    return root


# --- ランダムサンプリング -------------------------------------------------

#: 演算子の事前重み（SPEC.md §2.6: add/mul 高頻度、div/pow 低頻度、単項は中頻度）
DEFAULT_OP_WEIGHTS: dict[str, float] = {
    "add": 10.0,
    "sub": 5.0,
    "mul": 10.0,
    "div": 2.0,
    "pow": 2.0,
    "sin": 3.0,
    "cos": 3.0,
    "exp": 1.0,
    "log": 1.0,
    "sqrt": 1.0,
    "neg": 1.0,
}

#: pow の右側（指数）に許すトークン。定義域の暴れを防ぐため小整数に限定する
POW_EXPONENTS = ["2", "3"]


def sample_tree(
    rng: np.random.Generator,
    vocab: Vocabulary,
    max_ops: int,
    p_const_leaf: float = 0.25,
    p_int_leaf: float = 0.15,
    op_weights: dict[str, float] | None = None,
) -> Node:
    """事前分布から式木を1本サンプリングする。

    内部ノード（演算子）数を ``1..max_ops`` から一様に選び、再帰的に構築する。
    pow の指数は :data:`POW_EXPONENTS` の整数に固定する。

    Args:
        rng: numpy 乱数生成器（再現性のため必ず外から渡す）。
        vocab: 語彙。演算子と変数の候補を決める。
        max_ops: 内部ノード数の上限。
        p_const_leaf: 葉が定数 ``C`` になる確率。
        p_int_leaf: 葉が整数リテラルになる確率（残りは変数）。
        op_weights: 演算子の相対頻度。None なら :data:`DEFAULT_OP_WEIGHTS`。

    Returns:
        サンプリングされた式木。変数を含む保証はない（呼び出し側で棄却する）。
    """
    weights = op_weights or DEFAULT_OP_WEIGHTS
    ops = [o for o in vocab.operators if o in weights]
    w = np.array([weights[o] for o in ops], dtype=float)
    w /= w.sum()
    budget = int(rng.integers(1, max_ops + 1))
    int_tokens = [t for t in vocab.tokens if t.isdigit()]

    def leaf() -> Node:
        u = rng.random()
        if u < p_const_leaf:
            return Node(CONST)
        if u < p_const_leaf + p_int_leaf and int_tokens:
            return Node(str(rng.choice(int_tokens)))
        return Node(f"x{rng.integers(0, vocab.n_vars)}")

    def build(b: int) -> Node:
        if b <= 0:
            return leaf()
        op = str(rng.choice(ops, p=w))
        if OPERATORS[op].arity == 1:
            return Node(op, (build(b - 1),))
        if op == "pow":
            exps = [e for e in POW_EXPONENTS if e in vocab.token_to_id]
            exp_tok = str(rng.choice(exps)) if exps else "2"
            return Node(op, (build(b - 1), Node(exp_tok)))
        left = int(rng.integers(0, b))
        return Node(op, (build(left), build(b - 1 - left)))

    return build(budget)
