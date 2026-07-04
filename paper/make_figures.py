"""論文用の図を EXPERIMENTS.md の実測値から生成する。

すべての数値は EXPERIMENTS.md / results/*.json に記録済みの実測値。
配色は検証済みカテゴリカルパレット（blue/aqua/yellow/green、CVD ΔE 24.2）。
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path

# --- スタイル ---------------------------------------------------------------
for f in ["/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf"]:
    font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "IPAPGothic",
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#d8d7d2",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e8e7e2",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 11,
    "figure.dpi": 150,
})
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"
GRAY = "#9b9a94"

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def strip(ax, x=False):
    ax.spines[["top", "right"]].set_visible(False)
    if x:
        ax.grid(axis="x", visible=False)
    else:
        ax.grid(axis="y", visible=True)
        ax.grid(axis="x", visible=False)


# --- 図1: 学習曲線 -----------------------------------------------------------
tiny = [1.2666, 0.7639, 0.6983, 0.6655, 0.6463, 0.6271, 0.6122, 0.6049]
proto = [0.8727, 0.6588, 0.6294, 0.6156]
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(range(1, 9), tiny, color=BLUE, lw=2, marker="o", ms=5, label="tiny1var（1変数）")
ax.plot(range(1, 5), proto, color=AQUA, lw=2, marker="o", ms=5, label="proto3var（3変数）")
ax.text(8, tiny[-1] - 0.04, "0.605", color=BLUE, fontsize=10, ha="center", va="top")
ax.text(4, proto[-1] - 0.04, "0.616", color=AQUA, fontsize=10, ha="center", va="top")
ax.set_xlabel("エポック")
ax.set_ylabel("学習損失（マスク位置CE）")
ax.set_title("学習曲線（CPU 学習: tiny 約3分 / proto 12.5分）", fontsize=12)
ax.legend(frameon=False)
strip(ax)
fig.tight_layout()
fig.savefig(OUT / "fig_loss.png")

# --- 図2: M4 パース成功率 vs 学習ステップ ------------------------------------
steps = [252, 504, 756, 1008, 1260]
rates = [0.74, 0.90, 0.76, 0.93, 0.90]
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(steps, rates, color=BLUE, lw=2, marker="o", ms=5, label="無条件モデル（8,000式）")
ax.scatter([96], [0.37], color=YELLOW, s=45, zorder=5)
ax.annotate("初期試行（2,000式・96step）\n0.37 で不合格", (96, 0.37), textcoords="offset points",
            xytext=(10, -6), fontsize=9, color="#52514e")
ax.axhline(0.8, color=GRAY, lw=1.2, ls="--")
ax.text(1265, 0.805, "合格ゲート 0.8", color="#52514e", fontsize=9, ha="right", va="bottom")
ax.set_xlabel("学習ステップ")
ax.set_ylabel("構文的に妥当なサンプルの割合")
ax.set_ylim(0.3, 1.0)
ax.set_title("無条件生成の構文妥当率（M4 検証）", fontsize=12)
ax.legend(frameon=False, loc="lower right")
strip(ax)
fig.tight_layout()
fig.savefig(OUT / "fig_parse.png")

# --- 図3: best-of-k 曲線（tiny） ---------------------------------------------
ks = [1, 4, 16, 64]
rec = [1, 4, 7, 9]
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(range(len(ks)), rec, color=BLUE, lw=2, marker="o", ms=6)
for i, v in enumerate(rec):
    ax.annotate(f"{v}/10", (i, v), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=10, color="#0b0b0b")
ax.set_xticks(range(len(ks)), [str(k) for k in ks])
ax.set_xlabel("サンプル数 k（best-of-k）")
ax.set_ylabel("記号的一致で復元できた式数（/10）")
ax.set_ylim(0, 10.8)
ax.set_title("tiny1var: k と復元数（ゲート10式・k=16 が M5 合格基準）", fontsize=12)
strip(ax)
fig.tight_layout()
fig.savefig(OUT / "fig_kcurve.png")

# --- 図4: in-distribution（proto、k=8 vs 32） --------------------------------
metrics = ["記号的一致", "スケルトン一致", "R²>0.999", "無効生成率"]
k8 = [0.14, 0.10, 0.22, 0.345]
k32 = [0.16, 0.10, 0.27, 0.316]
x = range(len(metrics))
w = 0.34
fig, ax = plt.subplots(figsize=(6.4, 3.4))
b1 = ax.bar([i - w / 2 for i in x], k8, width=w - 0.02, color=BLUE, label="k=8")
b2 = ax.bar([i + w / 2 for i in x], k32, width=w - 0.02, color=AQUA, label="k=32")
for bars in (b1, b2):
    ax.bar_label(bars, fmt="%.2f", fontsize=9, color="#52514e", padding=2)
ax.set_xticks(list(x), metrics)
ax.set_ylabel("割合")
ax.set_ylim(0, 0.45)
ax.set_title("proto3var の in-distribution 評価（held-out 100問）", fontsize=12)
ax.legend(frameon=False)
strip(ax)
fig.tight_layout()
fig.savefig(OUT / "fig_indist.png")

# --- 図5: ベンチマーク集計 ----------------------------------------------------
methods = ["DiffSR\n(提案)", "乱択事前分布", "Lasso", "gplearn"]
nguyen = [0, 0, 0, 0]
feynman = [0, 2, 0, 5]
x = range(len(methods))
fig, ax = plt.subplots(figsize=(6.4, 3.4))
b1 = ax.bar([i - w / 2 for i in x], nguyen, width=w - 0.02, color=BLUE, label="Nguyen（10問）")
b2 = ax.bar([i + w / 2 for i in x], feynman, width=w - 0.02, color=YELLOW, label="Feynman 易問（10問）")
for bars in (b1, b2):
    ax.bar_label(bars, fontsize=10, color="#0b0b0b", padding=2)
ax.set_xticks(list(x), methods)
ax.set_ylabel("記号的一致数（/10）")
ax.set_ylim(0, 6.5)
ax.set_title("ベンチマーク: 記号的一致数の比較（k=32）", fontsize=12)
ax.legend(frameon=False)
strip(ax)
fig.tight_layout()
fig.savefig(OUT / "fig_bench.png")

# --- 図6: 分布ギャップ決定実験 -------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 3.2))
conds = ["ベンチマーク区間\n[0.5, 2]", "学習区間\n[−4, 4]"]
vals = [0, 3]
bars = ax.bar(conds, vals, width=0.45, color=[GRAY, BLUE])
ax.bar_label(bars, labels=["0/5", "3/5"], fontsize=12, color="#0b0b0b", padding=3)
ax.set_ylabel("復元できた式数（/5）")
ax.set_ylim(0, 5)
ax.set_title("同一の5式・条件付けデータの区間だけを変えた場合（k=32）", fontsize=12)
strip(ax)
fig.tight_layout()
fig.savefig(OUT / "fig_gap.png")

# --- 図7: ステップ時間の要因分解 -----------------------------------------------
configs = [
    ("d64 + points + n=32", 0.079),
    ("d_model=64", 0.103),
    ("n_points=32", 0.179),
    ("現行 (PMA, n=64, d=128)", 0.247),
    ("PMA → points", 0.255),
    ("n_points=128", 0.421),
]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
names = [c[0] for c in configs]
vals = [c[1] for c in configs]
colors = [BLUE if "現行" not in n else GREEN for n in names]
bars = ax.barh(names, vals, height=0.55, color=colors)
ax.bar_label(bars, fmt="%.3f", fontsize=9, color="#52514e", padding=3)
ax.set_xlabel("1ステップの学習時間（秒、batch=128、20ステップ平均）")
ax.set_xlim(0, 0.47)
ax.set_title("学習速度の要因分解（支配要因は観測点数 n_points）", fontsize=12)
ax.invert_yaxis()
strip(ax, x=True)
ax.grid(axis="x", visible=True)
fig.tight_layout()
fig.savefig(OUT / "fig_speed.png")

print("生成完了:", sorted(p.name for p in OUT.glob("*.png")))
