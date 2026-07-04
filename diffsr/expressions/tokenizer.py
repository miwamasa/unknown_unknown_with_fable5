"""トークン列 ⇔ ID 列の変換（固定長パディング付き）。

モデルが扱うのは長さ ``seq_len`` の ID 列。式の前置記法トークン列を右側
``[PAD]`` 埋めで固定長にする。デコードは PAD を剥がして式木にパースする。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from diffsr.expressions.grammar import PAD, Vocabulary
from diffsr.expressions.tree import Node, ParseError, parse_prefix


class Tokenizer:
    """語彙と系列長を束ねた変換器。

    Args:
        vocab: 語彙。
        seq_len: 固定系列長。これより長い式はエンコード時にエラー。
    """

    def __init__(self, vocab: Vocabulary, seq_len: int) -> None:
        self.vocab = vocab
        self.seq_len = seq_len

    def encode(self, tokens: Sequence[str]) -> np.ndarray:
        """トークン列 → 固定長 ID 配列（int64, shape=(seq_len,)）。

        Raises:
            ValueError: 式が seq_len を超える、または語彙外トークンを含む場合。
        """
        if len(tokens) > self.seq_len:
            raise ValueError(f"式の長さ {len(tokens)} が seq_len={self.seq_len} を超えています")
        ids = np.full(self.seq_len, self.vocab.pad_id, dtype=np.int64)
        for i, t in enumerate(tokens):
            if t not in self.vocab.token_to_id:
                raise ValueError(f"語彙外のトークン: {t}")
            ids[i] = self.vocab.token_to_id[t]
        return ids

    def decode(self, ids: Sequence[int]) -> list[str]:
        """ID 列 → トークン列。末尾に限らず PAD は全て除去する。

        MASK が残っている列（未完成のサンプル）はそのまま ``[MASK]``
        トークンとして返す（パースは失敗する）。
        """
        return [
            self.vocab.id_to_token(int(i))
            for i in ids
            if self.vocab.id_to_token(int(i)) != PAD
        ]

    def encode_tree(self, tree: Node) -> np.ndarray:
        return self.encode(tree.serialize())

    def decode_to_tree(self, ids: Sequence[int]) -> Node:
        """ID 列 → 式木。構文的に無効なら :class:`ParseError`。"""
        tokens = self.decode(ids)
        if not tokens:
            raise ParseError("空の系列です")
        return parse_prefix(tokens, self.vocab)
