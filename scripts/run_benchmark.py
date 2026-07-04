"""ベンチマーク一括実行 CLI: 提案手法＋ベースラインを同一問題で比較する。

使い方:
    python scripts/run_benchmark.py --model runs/proto3var --suite nguyen \
        --out results/nguyen --k 32 --seed 0
    python scripts/run_benchmark.py --model runs/proto3var --suite feynman \
        --out results/feynman --skip gplearn

出力: <out>.json（生データ）と <out>.md（結果表）。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sympy as sp

from diffsr.eval.baselines import gplearn_baseline, lasso_baseline, random_prior_baseline
from diffsr.eval.benchmarks import SUITES
from diffsr.eval.metrics import expression_match, r2_score, snap_constants
from diffsr.expressions.sympy_bridge import make_numpy_fn, symbolic_equivalent, to_sympy
from diffsr.pipeline import predict_expression
from diffsr.train import load_model

N_TEST = 200  # R² 計算用のテスト点数（条件付けデータとは独立に同一区間から生成）


def evaluate_sympy_expr(expr, true_expr, y_pred, y_test):
    """(記号的一致, R²) を計算する共通処理。"""
    match = expr is not None and symbolic_equivalent(expr, true_expr, timeout=5.0)
    r2 = r2_score(y_test, y_pred) if y_pred is not None else float("-inf")
    return match, r2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", required=True, choices=sorted(SUITES))
    ap.add_argument("--out", required=True, help="出力パスの接頭辞（.json/.md を付けて保存）")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[], choices=["gplearn", "lasso", "random"],
                    help="スキップするベースライン")
    args = ap.parse_args()

    cfg, vocab, tokenizer, model, diffusion = load_model(args.model)
    k = args.k or cfg.k_samples
    problems = [p for p in SUITES[args.suite] if p.n_vars <= cfg.n_vars]
    skipped = [p.name for p in SUITES[args.suite] if p.n_vars > cfg.n_vars]
    if skipped:
        print(f"変数数がモデル上限 {cfg.n_vars} を超えるためスキップ: {skipped}")

    rows = []
    for prob in problems:
        true_expr = prob.expr
        X, y = prob.make_data(cfg.n_points, seed=args.seed)
        X_test, y_test = prob.make_data(N_TEST, seed=args.seed + 1)
        row = {"problem": prob.name, "true": str(true_expr), "methods": {}}

        # --- 提案手法 ---
        t0 = time.monotonic()
        res = predict_expression(model, diffusion, cfg, vocab, tokenizer, X, y,
                                 k=k, seed=args.seed)
        elapsed = time.monotonic() - t0
        if res.best is not None:
            b = res.best
            Xt = X_test
            if Xt.shape[1] < cfg.n_vars:
                pad_rng = np.random.default_rng(args.seed + 10_000)
                lo, hi = cfg.x_range
                Xt = np.concatenate(
                    [Xt, pad_rng.uniform(lo, hi, (Xt.shape[0], cfg.n_vars - Xt.shape[1]))], axis=1)
            y_pred = make_numpy_fn(b.skeleton)(Xt, b.constants)
            match = expression_match(b.skeleton, b.constants, true_expr)
            pred_str = str(to_sympy(b.skeleton, const_values=None).subs(
                {sp.Symbol(f"c{i}"): v for i, v in enumerate(snap_constants(b.constants))}))
            row["methods"]["DiffSR"] = {
                "match": bool(match), "r2": r2_score(y_test, y_pred),
                "expr": pred_str, "complexity": b.complexity,
                "invalid_rate": res.invalid_rate, "time_s": elapsed,
            }
        else:
            row["methods"]["DiffSR"] = {"match": False, "r2": float("-inf"),
                                        "expr": None, "invalid_rate": res.invalid_rate,
                                        "time_s": elapsed}

        # --- ベースライン（1手法の失敗でスイート全体を止めない） ---
        def run_baseline(name: str, fn) -> None:
            try:
                r = fn()
                y_pred = r.y_pred_fn(X_test[:, : prob.n_vars]) if r.expr is not None else None
                m, r2 = evaluate_sympy_expr(r.expr, true_expr, y_pred, y_test)
                row["methods"][name] = {
                    "match": bool(m), "r2": r2, "expr": str(r.expr), "time_s": r.elapsed_s}
            except Exception as e:
                row["methods"][name] = {"match": False, "r2": float("-inf"),
                                        "expr": None, "error": repr(e)}

        if "random" not in args.skip:
            run_baseline("RandomPrior", lambda: random_prior_baseline(
                cfg, vocab, tokenizer, X, y, k=k, seed=args.seed))
        if "lasso" not in args.skip:
            run_baseline("Lasso", lambda: lasso_baseline(
                X[:, : prob.n_vars], y, seed=args.seed))
        if "gplearn" not in args.skip:
            run_baseline("gplearn", lambda: gplearn_baseline(
                X[:, : prob.n_vars], y, seed=args.seed))

        rows.append(row)
        summary = "  ".join(
            f"{name}: {'一致' if v['match'] else '不一致'} R²={v['r2']:.3f}"
            for name, v in row["methods"].items())
        print(f"{prob.name}: {summary}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"suite": args.suite, "model": args.model, "k": k, "seed": args.seed,
            "n_points": cfg.n_points, "n_test": N_TEST}
    out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2, default=str))
    out.with_suffix(".md").write_text(render_markdown(meta, rows))
    print(f"保存: {out}.json / {out}.md")


def render_markdown(meta: dict, rows: list[dict]) -> str:
    methods = list(rows[0]["methods"]) if rows else []
    lines = [
        f"# ベンチマーク結果: {meta['suite']}",
        "",
        f"モデル: `{meta['model']}` / k={meta['k']} / seed={meta['seed']} / "
        f"条件付け {meta['n_points']} 点 / テスト {meta['n_test']} 点",
        "",
        "| 問題 | 正解 | " + " | ".join(f"{m} 一致 | {m} R²" for m in methods) + " |",
        "|---|---|" + "---|" * (2 * len(methods)),
    ]
    for row in rows:
        cells = []
        for m in methods:
            v = row["methods"].get(m, {})
            cells.append("✔" if v.get("match") else "✘")
            r2 = v.get("r2", float("-inf"))
            cells.append(f"{r2:.3f}" if np.isfinite(r2) else "—")
        lines.append(f"| {row['problem']} | `{row['true']}` | " + " | ".join(cells) + " |")
    # 集計
    lines += ["", "## 集計（記号的一致数 / 問題数）", ""]
    for m in methods:
        n_match = sum(1 for r in rows if r["methods"].get(m, {}).get("match"))
        lines.append(f"- {m}: {n_match}/{len(rows)}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
