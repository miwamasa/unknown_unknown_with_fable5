"""M4 検証: 前向き過程・1バッチ過学習・無条件サンプルの妥当性（PLAN.md §4.1）。"""

import numpy as np
import pytest
import torch

from diffsr.config import get_config
from diffsr.expressions.tree import ParseError
from diffsr.train import build_components


@pytest.fixture
def tiny():
    cfg = get_config("tiny1var")
    vocab, tok, model, diff = build_components(cfg)
    return cfg, vocab, tok, model, diff


def test_forward_process_boundaries(tiny):
    """t=1 で全トークンが [MASK]、t=0 で原系列と一致する。"""
    cfg, vocab, tok, model, diff = tiny
    x0 = torch.randint(2, len(vocab), (4, cfg.seq_len))
    x_t, mask = diff.corrupt(x0, torch.ones(4))
    assert (x_t == vocab.mask_id).all() and mask.all()
    x_t, mask = diff.corrupt(x0, torch.zeros(4))
    assert (x_t == x0).all() and (~mask).all()


def test_forward_process_mask_fraction(tiny):
    """t=0.5 でマスク率がほぼ 1/2（大数の法則の範囲で）。"""
    cfg, vocab, tok, model, diff = tiny
    x0 = torch.randint(2, len(vocab), (64, cfg.seq_len))
    g = torch.Generator().manual_seed(0)
    _, mask = diff.corrupt(x0, torch.full((64,), 0.5), generator=g)
    assert abs(mask.float().mean().item() - 0.5) < 0.05


def test_sample_contains_no_mask_or_oov(tiny):
    """未学習モデルでも、サンプル結果に [MASK] が残らない。"""
    cfg, vocab, tok, model, diff = tiny
    g = torch.Generator().manual_seed(0)
    memory = model.encode(None, 1)
    ids = diff.sample(model, memory, n_samples=8, generator=g)
    assert ids.shape == (8, cfg.seq_len)
    assert (ids != vocab.mask_id).all()


@pytest.mark.slow
def test_overfit_single_sequence(tiny):
    """単一系列への過学習で損失が十分下がる（最適化の sanity check）。"""
    cfg, vocab, tok, model, diff = tiny
    x0 = torch.from_numpy(tok.encode(["add", "mul", "C", "x0", "C"]))[None].repeat(32, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    last = None
    for step in range(300):
        loss = diff.loss(model, x0, points=None, generator=g)
        opt.zero_grad(); loss.backward(); opt.step()
        last = loss.item()
    assert last < 0.05, f"過学習で損失が下がらない: {last}"


@pytest.mark.slow
def test_unconditional_samples_mostly_parse(tiny):
    """小コーパスで無条件学習 → サンプル500本のパース成功率 ≥80%（M4 ゲート）。"""
    from diffsr.data.generator import generate_dataset
    from diffsr.train import problems_to_tensors

    cfg, vocab, tok, model, diff = tiny
    problems = generate_dataset(cfg, vocab, tok, n=8000, seed=0)
    ids, _ = problems_to_tensors(problems)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    for ep in range(20):
        perm = torch.randperm(len(ids), generator=g)
        for i in range(0, len(ids), 128):
            loss = diff.loss(model, ids[perm[i : i + 128]], points=None, generator=g)
            opt.zero_grad(); loss.backward(); opt.step()

    memory = model.encode(None, 1)
    samples = diff.sample(model, memory, n_samples=500, generator=g)
    ok = 0
    for row in samples:
        try:
            tok.decode_to_tree(row.numpy())
            ok += 1
        except ParseError:
            pass
    rate = ok / 500
    assert rate >= 0.8, f"無条件サンプルのパース成功率 {rate:.2f} < 0.8"
