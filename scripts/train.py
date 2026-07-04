"""学習 CLI。

使い方:
    python scripts/train.py --config tiny1var --out runs/tiny1var
    python scripts/train.py --config proto3var --out runs/proto3var --seed 0
"""

import argparse

from diffsr.config import get_config
from diffsr.train import train_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="プリセット名 (tiny1var / proto3var)")
    ap.add_argument("--out", required=True, help="チェックポイント出力ディレクトリ")
    ap.add_argument("--seed", type=int, default=None, help="乱数シード（省略時はプリセット値）")
    ap.add_argument("--n-train", type=int, default=None, help="学習式数の上書き")
    ap.add_argument("--epochs", type=int, default=None, help="エポック数の上書き")
    args = ap.parse_args()

    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    cfg = get_config(args.config, **overrides)
    train_model(cfg, out_dir=args.out, n_train=args.n_train, epochs=args.epochs)
    print(f"保存先: {args.out}")


if __name__ == "__main__":
    main()
