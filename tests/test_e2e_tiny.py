"""M5 合格ゲート: tiny1var の end-to-end 復元テスト（PLAN.md §3）。

held-out 10 式のうち 7 式以上を記号的一致（定数スナップ込み）で復元できれば
合格。学習済みチェックポイント ``runs/tiny1var`` が無い場合は skip する
（学習は ``python scripts/train.py --config tiny1var --out runs/tiny1var``）。
"""

from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from diffsr.eval.metrics import expression_match
from diffsr.pipeline import predict_expression
from diffsr.train import load_model

RUN_DIR = Path(__file__).resolve().parent.parent / "runs" / "tiny1var"

#: held-out の簡単な式（学習コーパスから独立に手で選定。PLAN.md M5）
GATE_EXPRESSIONS = [
    "2*x0 + 1",
    "x0**2 + x0",
    "3*sin(x0)",
    "sin(x0) + x0",
    "x0**3",
    "x0**2 - 2",
    "2*sin(x0) + 1",
    "sin(2*x0)",
    "5*x0 - 3",
    "x0*sin(x0)",
]


@pytest.mark.slow
def test_tiny1var_gate():
    if not (RUN_DIR / "model.pt").exists():
        pytest.skip("runs/tiny1var が未学習（scripts/train.py を先に実行）")
    cfg, vocab, tokenizer, model, diffusion = load_model(RUN_DIR)
    rng = np.random.default_rng(42)
    lo, hi = cfg.x_range
    x0 = sp.Symbol("x0")

    results = {}
    for s in GATE_EXPRESSIONS:
        true_expr = sp.sympify(s)
        X = rng.uniform(lo, hi, size=(cfg.n_points, 1))
        y = np.asarray(sp.lambdify(x0, true_expr, "numpy")(X[:, 0]), dtype=float)
        res = predict_expression(model, diffusion, cfg, vocab, tokenizer, X, y,
                                 k=16, seed=0)
        ok = res.best is not None and expression_match(
            res.best.skeleton, res.best.constants, true_expr
        )
        results[s] = ok

    n_ok = sum(results.values())
    detail = "\n".join(f"  {'OK' if v else 'NG'}  {k}" for k, v in results.items())
    assert n_ok >= 7, f"復元 {n_ok}/10 < 7（M5 ゲート不合格）:\n{detail}"
