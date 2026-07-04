"""1問題の推論 CLI: 式を指定してデータを作り、モデルに復元させる。

使い方:
    python scripts/evaluate.py --model runs/tiny1var --expr "2*x0 + 1" --k 16
"""

import argparse

import numpy as np
import sympy as sp

from diffsr.eval.metrics import expression_match, r2_score, snap_constants
from diffsr.expressions.sympy_bridge import make_numpy_fn, to_sympy
from diffsr.pipeline import predict_expression
from diffsr.train import load_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="チェックポイントディレクトリ")
    ap.add_argument("--expr", required=True, help="正解式（SymPy 構文、変数は x0, x1, ...）")
    ap.add_argument("--k", type=int, default=None, help="best-of-k の k")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg, vocab, tokenizer, model, diffusion = load_model(args.model)
    true_expr = sp.sympify(args.expr)

    rng = np.random.default_rng(args.seed)
    lo, hi = cfg.x_range
    X = rng.uniform(lo, hi, size=(cfg.n_points, cfg.n_vars))
    lam = sp.lambdify([sp.Symbol(f"x{i}") for i in range(cfg.n_vars)], true_expr, "numpy")
    y = np.broadcast_to(np.asarray(lam(*[X[:, i] for i in range(cfg.n_vars)]), dtype=float), (X.shape[0],))

    res = predict_expression(model, diffusion, cfg, vocab, tokenizer, X, y,
                             k=args.k, seed=args.seed)
    print(f"サンプル {res.n_sampled} 本 / 無効 {res.n_invalid} 本 (無効率 {res.invalid_rate:.2f})")
    if res.best is None:
        print("有効な候補が得られませんでした")
        return
    best = res.best
    pred = to_sympy(best.skeleton, const_values=None).subs(
        {sp.Symbol(f"c{i}"): v for i, v in enumerate(snap_constants(best.constants))}
    )
    y_pred = make_numpy_fn(best.skeleton)(X, best.constants)
    print(f"予測式: {sp.simplify(pred)}")
    print(f"MSE={best.mse:.3e}  R²={r2_score(y, y_pred):.6f}  複雑度={best.complexity}")
    print(f"記号的一致: {expression_match(best.skeleton, best.constants, true_expr)}")


if __name__ == "__main__":
    main()
