"""in-distribution 評価 CLI（M7 検証）。

学習と同じ事前分布から held-out の式を生成し（シードは学習と別）、
記号的一致率・スケルトン一致率・R²>0.999 率・無効生成率を測る。

使い方:
    python scripts/eval_indist.py --model runs/proto3var --n 200 --k 32
"""

import argparse
import json

import numpy as np

from diffsr.data.generator import generate_dataset
from diffsr.eval.metrics import expression_match, r2_score, skeleton_match
from diffsr.expressions.sympy_bridge import make_numpy_fn, to_sympy
from diffsr.pipeline import predict_expression
from diffsr.train import load_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=200, help="held-out 問題数")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=999, help="held-out 生成シード（学習と別に）")
    ap.add_argument("--out", default=None, help="集計 JSON の保存先")
    args = ap.parse_args()

    cfg, vocab, tokenizer, model, diffusion = load_model(args.model)
    k = args.k or cfg.k_samples
    problems = generate_dataset(cfg, vocab, tokenizer, n=args.n, seed=args.seed)

    stats = {"expr_match": 0, "skel_match": 0, "r2_999": 0, "no_candidate": 0}
    invalid_rates = []
    for i, p in enumerate(problems):
        true_expr = to_sympy(p.skeleton, const_values=list(p.const_values))
        res = predict_expression(model, diffusion, cfg, vocab, tokenizer, p.X, p.y,
                                 k=k, seed=args.seed + i)
        invalid_rates.append(res.invalid_rate)
        if res.best is None:
            stats["no_candidate"] += 1
            continue
        b = res.best
        stats["skel_match"] += skeleton_match(b.skeleton, p.skeleton)
        stats["expr_match"] += expression_match(b.skeleton, b.constants, true_expr)
        y_pred = make_numpy_fn(b.skeleton)(p.X, b.constants)
        stats["r2_999"] += r2_score(p.y, y_pred) > 0.999
        if (i + 1) % 20 == 0:
            print(f"{i+1}/{args.n}  一致率 {stats['expr_match']/(i+1):.2f}")

    n = args.n
    summary = {
        "model": args.model, "n": n, "k": k, "seed": args.seed,
        "expression_match_rate": stats["expr_match"] / n,
        "skeleton_match_rate": stats["skel_match"] / n,
        "r2_gt_0.999_rate": stats["r2_999"] / n,
        "mean_invalid_rate": float(np.mean(invalid_rates)),
        "no_candidate_rate": stats["no_candidate"] / n,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
