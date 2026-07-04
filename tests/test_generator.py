"""M2 検証: 生成データの再現性・妥当性・統計（PLAN.md §4.1）。"""

import numpy as np
import pytest

from diffsr.config import get_config
from diffsr.data.generator import generate_dataset
from diffsr.expressions.grammar import Vocabulary
from diffsr.expressions.tokenizer import Tokenizer


@pytest.fixture(params=["tiny1var", "proto3var"])
def setup(request):
    cfg = get_config(request.param, n_points=32)
    vocab = Vocabulary(list(cfg.operators), cfg.n_vars, list(cfg.integers))
    tok = Tokenizer(vocab, cfg.seq_len)
    return cfg, vocab, tok


def test_fixed_seed_full_reproducibility(setup):
    """同一シードで token_ids・X・y が bit 単位で一致する。"""
    cfg, vocab, tok = setup
    d1 = generate_dataset(cfg, vocab, tok, n=30, seed=123)
    d2 = generate_dataset(cfg, vocab, tok, n=30, seed=123)
    for p1, p2 in zip(d1, d2):
        np.testing.assert_array_equal(p1.token_ids, p2.token_ids)
        np.testing.assert_array_equal(p1.X, p2.X)
        np.testing.assert_array_equal(p1.y, p2.y)
        np.testing.assert_array_equal(p1.const_values, p2.const_values)


def test_generated_problems_are_valid(setup):
    """生成された全問題: パース可能・変数使用・有限値・非退化。"""
    cfg, vocab, tok = setup
    data = generate_dataset(cfg, vocab, tok, n=200, seed=0)
    for p in data:
        assert tok.decode_to_tree(p.token_ids) == p.skeleton  # 往復一致
        assert p.skeleton.variables_used()
        assert np.all(np.isfinite(p.y))
        assert np.max(np.abs(p.y)) <= cfg.y_max_abs
        assert np.std(p.y) > 1e-9
        assert p.X.shape == (cfg.n_points, cfg.n_vars)


def test_distribution_statistics(setup):
    """長さ分布・演算子頻度が想定レンジ内（緩い健全性チェック）。"""
    cfg, vocab, tok = setup
    data = generate_dataset(cfg, vocab, tok, n=300, seed=1)
    lengths = [len(p.skeleton.serialize()) for p in data]
    assert 1 < np.mean(lengths) < cfg.seq_len
    # add/mul が div より高頻度という事前分布が反映されていること
    counts = {op: 0 for op in cfg.operators}
    for p in data:
        for t in p.skeleton.serialize():
            if t in counts:
                counts[t] += 1
    assert counts["add"] > counts.get("div", 0)
