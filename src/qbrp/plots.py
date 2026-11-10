"""Figures for the experiment report (requires the optional ``plots`` extra)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .experiments import STRATEGIES

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.hashsalt": "qbrp",
    }
)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def make_figures(out: Path, runs, sens, over, ex_equity, ex_cut, ex_grid) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    label = {k: v[0] for k, v in STRATEGIES.items()}
    short = {
        "buy_and_hold": "Buy &\nhold",
        "ma_5_20": "MA 5/20\nL/S",
        "ma_20_50": "MA 20/50\nL/S",
        "ma_20_50_long": "MA 20/50\nlong-only",
        "momentum_126": "Mom. 126d\nL/S",
    }
    names: list[str] = []

    regimes = list(runs["regime"].unique())
    fig, axes = plt.subplots(1, len(regimes), figsize=(5 * len(regimes), 3.6), sharey=True)
    for ax, regime in zip(axes, regimes, strict=True):
        sub = runs[runs["regime"] == regime]
        keys = list(sub["strategy"].unique())
        ax.boxplot(
            [sub.loc[sub["strategy"] == k, "sharpe"] for k in keys],
            showfliers=False,
            whis=(5, 95),
        )
        ax.set_xticks(range(1, len(keys) + 1), [short.get(k, k) for k in keys])
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"{regime}: Sharpe across paths (5-95% whiskers)")
    axes[0].set_ylabel("Annualised Sharpe")
    _save(fig, out / "compare_sharpe.png")
    names.append("compare_sharpe.png")

    strategies = list(sens["strategy"].unique())
    fig, axes = plt.subplots(1, len(strategies), figsize=(5 * len(strategies), 3.4), sharey=True)
    for ax, strat in zip(axes, strategies, strict=True):
        s = sens[sens["strategy"] == strat]
        for lag, grp in s.groupby("execution_lag"):
            ax.plot(grp["fee_bps"], grp["mean_sharpe"], marker="o", ms=3, label=f"lag {lag}")
        ax.axhline(s["bh_mean_sharpe"].iloc[0], color="black", ls="--", lw=1, label="buy & hold")
        ax.set_title(f"{label[strat]} (weak_trend)")
        ax.set_xlabel("Fee (bp per unit turnover)")
    axes[0].set_ylabel("Mean Sharpe")
    axes[-1].legend(fontsize=7, frameon=False, loc="lower left", ncol=2)
    _save(fig, out / "sensitivity.png")
    names.append("sensitivity.png")

    best = ex_grid.iloc[0]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(ex_equity.index, ex_equity.to_numpy(), lw=1)
    ax.axvspan(ex_equity.index[0], ex_equity.index[ex_cut], color="tab:blue", alpha=0.07)
    ax.axvline(ex_equity.index[ex_cut], color="red", ls="--", lw=1)
    top = ax.get_ylim()[1]
    ax.text(ex_equity.index[ex_cut // 2], top, "in-sample: rule chosen here", ha="center", va="top")
    ax.text(ex_equity.index[(ex_cut + len(ex_equity)) // 2], top, "out-of-sample", ha="center",
            va="top")
    ax.set_title(
        f"Best of {len(ex_grid)} MA pairs on a zero-edge path: MA {int(best['fast'])}/"
        f"{int(best['slow'])}, in-sample Sharpe {best['is_sharpe']:.2f}, "
        f"out-of-sample {best['oos_sharpe']:.2f}"
    )
    ax.set_ylabel("Equity")
    _save(fig, out / "overfit_example.png")
    names.append("overfit_example.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, sharey=True)
    for ax, (regime, grp) in zip(axes, over.groupby("regime", sort=False), strict=False):
        ax.scatter(grp["best_is_sharpe"], grp["best_oos_sharpe"], s=8, alpha=0.6)
        lo = min(grp["best_is_sharpe"].min(), grp["best_oos_sharpe"].min())
        hi = max(grp["best_is_sharpe"].max(), grp["best_oos_sharpe"].max())
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls="--", label="OOS = IS")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(f"{regime}: selected rule per path")
        ax.set_xlabel("In-sample Sharpe (selected)")
    axes[0].set_ylabel("Out-of-sample Sharpe")
    axes[0].legend(fontsize=7, frameon=False)
    _save(fig, out / "overfit_scatter.png")
    names.append("overfit_scatter.png")
    return names
