"""実験 config（dataclass プリセット）。

プリセット:
- ``tiny1var``: PLAN.md §3 の最小 end-to-end 構成（M5 合格ゲート用）。
- ``proto3var``: プロトタイプ規模（M7、1〜3変数・全演算子）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    """1回の実験を完全に規定する設定。

    Attributes:
        name: プリセット名。
        n_vars: 変数の最大数（語彙は x0..x{n_vars-1}）。
        operators: 使用演算子。
        integers: 語彙に含める整数リテラル。
        max_ops: 式サンプリング時の内部ノード数上限。
        seq_len: トークン系列の固定長。
        n_points: 1問題あたりの観測点数。
        x_range: 変数のサンプリング区間 (low, high)。
        y_max_abs: |y| がこれを超えるサンプルは生成時に棄却。
        n_train: 学習用の式の本数。
        d_model / n_heads / n_layers_dec / n_layers_enc: モデル寸法。
        encoder_type: "points"（v1: 全観測点の埋め込みをそのままメモリにする）
            / "pma"（v2: learned query による m 個への Set Transformer 風プーリング）。
        diffusion_steps: 逆過程のステップ数。
        batch_size / lr / epochs: 学習ハイパーパラメータ。
        k_samples: 推論時の best-of-k の k。
        temperature: サンプリング温度。
        lambda_complexity: 候補選択スコア MSE + λ*complexity の λ。
        seed: 乱数シード（データ生成・学習・推論すべての起点）。
    """

    name: str = "tiny1var"
    # --- 問題クラス ---
    n_vars: int = 1
    operators: tuple[str, ...] = ("add", "sub", "mul", "sin")
    integers: tuple[int, ...] = (1, 2, 3)
    max_ops: int = 4
    seq_len: int = 16
    n_points: int = 100
    x_range: tuple[float, float] = (-4.0, 4.0)
    y_max_abs: float = 1e4
    # --- データ量 ---
    n_train: int = 20000
    # --- モデル ---
    d_model: int = 64
    n_heads: int = 2
    n_layers_dec: int = 2
    n_layers_enc: int = 2
    encoder_type: str = "points"
    # --- 拡散・学習 ---
    diffusion_steps: int = 20
    batch_size: int = 128
    lr: float = 3e-4
    epochs: int = 8
    # --- 推論 ---
    k_samples: int = 16
    temperature: float = 1.0
    lambda_complexity: float = 1e-4
    # --- 再現性 ---
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, Config] = {
    "tiny1var": Config(),
    "proto3var": Config(
        name="proto3var",
        n_vars=3,
        operators=("add", "sub", "mul", "div", "pow", "sin", "cos", "exp", "log", "sqrt"),
        integers=(1, 2, 3, 4, 5),
        max_ops=7,
        seq_len=32,
        n_points=64,  # 128から縮小: CPU 60分予算に収めるため（EXPERIMENTS.md §5）
        n_train=60000,
        d_model=128,
        n_heads=4,
        n_layers_dec=4,
        n_layers_enc=3,
        encoder_type="pma",
        diffusion_steps=40,
        batch_size=128,
        lr=3e-4,
        epochs=4,
        k_samples=32,
    ),
}


def get_config(name: str, **overrides) -> Config:
    """プリセット名から Config を取得し、キーワードで上書きする。"""
    import dataclasses

    if name not in PRESETS:
        raise KeyError(f"未知のプリセット: {name}（候補: {list(PRESETS)}）")
    return dataclasses.replace(PRESETS[name], **overrides)
