"""演算子定義と語彙。

トークンの種類:
- 特殊: ``[PAD]``（パディング）, ``[MASK]``（拡散の吸収状態）
- 演算子: 二項 ``add, sub, mul, div, pow`` / 単項 ``sin, cos, exp, log, sqrt, neg``
- 葉: 変数 ``x0..x{n-1}``、定数プレースホルダ ``C``、小整数リテラル ``1..5``

数式との対応: トークン列は式木の前置記法（ポーランド記法）。
例: ``y = 2*x0 + 1`` のスケルトンは ``add mul C x0 C``（定数は C に抽象化）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import sympy as sp

PAD = "[PAD]"
MASK = "[MASK]"
CONST = "C"


@dataclass(frozen=True)
class Operator:
    """演算子1個の定義。

    Args:
        name: トークン文字列（例 ``"add"``）。
        arity: 引数の数（1 or 2）。
        sympy_fn: SymPy 式を受けて SymPy 式を返す関数。
        numpy_fn: numpy 配列を受けて numpy 配列を返す関数。
    """

    name: str
    arity: int
    sympy_fn: Callable
    numpy_fn: Callable


def _np_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(a, b)


def _np_pow(a, b):
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.power(a, b)


def _np_log(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(a)


def _np_sqrt(a):
    with np.errstate(invalid="ignore"):
        return np.sqrt(a)


def _np_exp(a):
    with np.errstate(over="ignore"):
        return np.exp(a)


OPERATORS: dict[str, Operator] = {
    op.name: op
    for op in [
        Operator("add", 2, lambda a, b: a + b, np.add),
        Operator("sub", 2, lambda a, b: a - b, np.subtract),
        Operator("mul", 2, lambda a, b: a * b, np.multiply),
        Operator("div", 2, lambda a, b: a / b, _np_div),
        Operator("pow", 2, lambda a, b: a**b, _np_pow),
        Operator("sin", 1, sp.sin, np.sin),
        Operator("cos", 1, sp.cos, np.cos),
        Operator("exp", 1, sp.exp, _np_exp),
        Operator("log", 1, sp.log, _np_log),
        Operator("sqrt", 1, sp.sqrt, _np_sqrt),
        Operator("neg", 1, lambda a: -a, np.negative),
    ]
}


@dataclass
class Vocabulary:
    """トークン ⇔ ID の対応と、各トークンのアリティを管理する。

    ID 割当は決定的: ``[PAD]=0, [MASK]=1``、以降は operators, variables,
    ``C``, integers の順。

    Args:
        operators: 使用する演算子名のリスト（``OPERATORS`` のキー）。
        n_vars: 変数の数（``x0..x{n_vars-1}`` を語彙に含める）。
        integers: 語彙に含める整数リテラル。
    """

    operators: Sequence[str]
    n_vars: int
    integers: Sequence[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])

    def __post_init__(self) -> None:
        unknown = [o for o in self.operators if o not in OPERATORS]
        if unknown:
            raise ValueError(f"未定義の演算子: {unknown}")
        self.tokens: list[str] = (
            [PAD, MASK]
            + list(self.operators)
            + [f"x{i}" for i in range(self.n_vars)]
            + [CONST]
            + [str(i) for i in self.integers]
        )
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        if len(self.token_to_id) != len(self.tokens):
            raise ValueError("トークンが重複しています")

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def mask_id(self) -> int:
        return self.token_to_id[MASK]

    def id_to_token(self, i: int) -> str:
        return self.tokens[i]

    def arity(self, token: str) -> int:
        """トークンのアリティ。葉（変数・C・整数）は 0。特殊トークンはエラー。"""
        if token in OPERATORS:
            return OPERATORS[token].arity
        if token in (PAD, MASK):
            raise ValueError(f"特殊トークン {token} にアリティはありません")
        return 0

    def is_operator(self, token: str) -> bool:
        return token in OPERATORS

    def variables(self) -> list[str]:
        return [f"x{i}" for i in range(self.n_vars)]
