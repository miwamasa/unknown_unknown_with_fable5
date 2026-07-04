"""M1 検証: 往復可逆性・不正系列・語彙の評価可能性（PLAN.md §4.1）。"""

import numpy as np
import pytest
import sympy as sp

from diffsr.config import get_config
from diffsr.expressions.grammar import MASK, PAD, OPERATORS, Vocabulary
from diffsr.expressions.sympy_bridge import make_numpy_fn, to_sympy
from diffsr.expressions.tokenizer import Tokenizer
from diffsr.expressions.tree import Node, ParseError, parse_prefix, sample_tree


@pytest.fixture(params=["tiny1var", "proto3var"])
def setup(request):
    cfg = get_config(request.param)
    vocab = Vocabulary(list(cfg.operators), cfg.n_vars, list(cfg.integers))
    tok = Tokenizer(vocab, cfg.seq_len)
    return cfg, vocab, tok


def test_roundtrip_1000_random_trees(setup):
    """ランダム式1000本で 式木→トークン→ID→トークン→式木 の完全一致。"""
    cfg, vocab, tok = setup
    rng = np.random.default_rng(0)
    n_ok = 0
    for _ in range(1000):
        tree = sample_tree(rng, vocab, cfg.max_ops)
        tokens = tree.serialize()
        if len(tokens) > cfg.seq_len:
            continue  # 長すぎる式はエンコード対象外（生成側で棄却される）
        ids = tok.encode(tokens)
        assert tok.decode(ids) == tokens
        assert tok.decode_to_tree(ids) == tree
        n_ok += 1
    assert n_ok > 500  # 大半が seq_len に収まっていること


def test_parse_rejects_truncated(setup):
    """アリティ不足（途中で尽きる）は ParseError。"""
    _, vocab, _ = setup
    with pytest.raises(ParseError):
        parse_prefix(["add", "x0"], vocab)


def test_parse_rejects_leftover(setup):
    """余分なトークンが残る列は ParseError。"""
    _, vocab, _ = setup
    with pytest.raises(ParseError):
        parse_prefix(["x0", "x0"], vocab)


def test_parse_rejects_special_and_unknown(setup):
    """[PAD]/[MASK]/語彙外トークンを含む列は ParseError。"""
    _, vocab, _ = setup
    for bad in ([MASK], [PAD], ["nosuchop"]):
        with pytest.raises(ParseError):
            parse_prefix(bad, vocab)


def test_encode_rejects_too_long(setup):
    cfg, vocab, tok = setup
    too_long = ["sin"] * cfg.seq_len + ["x0"]
    with pytest.raises(ValueError):
        tok.encode(too_long)


def test_every_vocab_token_is_evaluable(setup):
    """全語彙トークン（特殊除く）が SymPy / numpy 評価に対応している。"""
    _, vocab, _ = setup
    x = np.full((1, vocab.n_vars), 0.5)
    for t in vocab.tokens:
        if t in (PAD, MASK):
            continue
        if t in OPERATORS:
            arity = OPERATORS[t].arity
            children = tuple(Node("x0") for _ in range(arity))
            tree = Node(t, children)
        else:
            tree = Node(t)
        expr = to_sympy(tree, const_values=[1.0] * tree.count("C"))
        assert isinstance(expr, sp.Expr)
        y = make_numpy_fn(tree)(x, [1.0] * tree.count("C"))
        assert y.shape == (1,)


def test_sample_tree_reproducible(setup):
    """同一シードで同一の式列が得られる（再現性の基礎）。"""
    cfg, vocab, _ = setup
    trees1 = _sample_n(vocab, cfg, seed=7, n=50)
    trees2 = _sample_n(vocab, cfg, seed=7, n=50)
    assert trees1 == trees2


def _sample_n(vocab, cfg, seed, n):
    rng = np.random.default_rng(seed)
    return [sample_tree(rng, vocab, cfg.max_ops).serialize() for _ in range(n)]
