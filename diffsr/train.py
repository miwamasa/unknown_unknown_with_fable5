"""学習ループ（CPU 前提の素朴な実装）。

`build_components` で config から語彙・トークナイザ・モデル・拡散ラッパを
一括構築し、`train_model` でデータ生成→学習→チェックポイント保存まで行う。
すべての乱数は cfg.seed から決定的に導出する（再現性テスト対象）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from diffsr.config import Config
from diffsr.data.generator import Problem, generate_dataset
from diffsr.expressions.grammar import Vocabulary
from diffsr.expressions.tokenizer import Tokenizer
from diffsr.model.diffusion import DiffSRModel, MaskedDiffusion


def build_components(cfg: Config) -> tuple[Vocabulary, Tokenizer, DiffSRModel, MaskedDiffusion]:
    """config から (vocab, tokenizer, model, diffusion) を構築する。"""
    vocab = Vocabulary(list(cfg.operators), cfg.n_vars, list(cfg.integers))
    tokenizer = Tokenizer(vocab, cfg.seq_len)
    torch.manual_seed(cfg.seed)  # モデル初期化を決定的に
    model = DiffSRModel(cfg, vocab)
    return vocab, tokenizer, model, MaskedDiffusion(vocab, cfg)


def problems_to_tensors(problems: list[Problem]) -> tuple[torch.Tensor, torch.Tensor]:
    """Problem 列 → (token_ids (N,L), points (N,n,d+1)) のテンソル。"""
    ids = torch.from_numpy(np.stack([p.token_ids for p in problems]))
    pts = torch.from_numpy(
        np.stack([np.concatenate([p.X, p.y[:, None]], axis=1) for p in problems])
    ).float()
    return ids, pts


def _print_flush(*args) -> None:
    print(*args, flush=True)  # nohup リダイレクト先でもリアルタイムに見えるように


def train_model(
    cfg: Config,
    unconditional: bool = False,
    out_dir: str | Path | None = None,
    n_train: int | None = None,
    epochs: int | None = None,
    log_fn=_print_flush,
    log_every_steps: int = 100,
) -> tuple[DiffSRModel, list[float]]:
    """データ生成＋学習を実行し、(model, エポック平均損失の列) を返す。

    Args:
        cfg: 実験 config。
        unconditional: True なら条件付けなし（M4 の分布学習検証用）。
        out_dir: 指定すると model.pt / config.json / losses.json を保存。
        n_train / epochs: cfg の値の上書き（テストの短縮用）。
        log_fn: 進捗ロガー。
        log_every_steps: ステップ単位の計時ログ（data/train 秒数）の間隔。
    """
    n_train = n_train or cfg.n_train
    epochs = epochs or cfg.epochs
    vocab, tokenizer, model, diffusion = build_components(cfg)

    t0 = time.monotonic()
    problems = generate_dataset(cfg, vocab, tokenizer, n=n_train, seed=cfg.seed)
    log_fn(f"データ生成: {n_train} 問題 {time.monotonic()-t0:.1f}s")
    ids, pts = problems_to_tensors(problems)

    gen = torch.Generator().manual_seed(cfg.seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    losses: list[float] = []
    model.train()
    step = 0
    for ep in range(epochs):
        perm = torch.randperm(len(ids), generator=gen)
        ep_losses = []
        for i in range(0, len(ids), cfg.batch_size):
            ts = time.monotonic()
            b = perm[i : i + cfg.batch_size]
            x0 = ids[b]
            points = None if unconditional else pts[b]
            t_data = time.monotonic()
            loss = diffusion.loss(model, x0, points, generator=gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            t_train = time.monotonic()
            ep_losses.append(loss.item())
            if step % log_every_steps == 0:
                log_fn(
                    f"step={step} data={t_data-ts:.3f}s train={t_train-t_data:.3f}s "
                    f"loss={ep_losses[-1]:.4f}"
                )
            step += 1
        losses.append(float(np.mean(ep_losses)))
        log_fn(f"epoch {ep+1}/{epochs}  loss={losses[-1]:.4f}  ({time.monotonic()-t0:.0f}s)")

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        (out / "losses.json").write_text(json.dumps(losses))
    return model, losses


def load_model(out_dir: str | Path) -> tuple[Config, Vocabulary, Tokenizer, DiffSRModel, MaskedDiffusion]:
    """保存済みディレクトリから (cfg, vocab, tokenizer, model, diffusion) を復元。"""
    out = Path(out_dir)
    cfg = Config(**json.loads((out / "config.json").read_text()))
    # JSON 経由で tuple が list になるため戻す
    cfg.operators = tuple(cfg.operators)
    cfg.integers = tuple(cfg.integers)
    cfg.x_range = tuple(cfg.x_range)
    vocab, tokenizer, model, diffusion = build_components(cfg)
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True))
    model.eval()
    return cfg, vocab, tokenizer, model, diffusion
