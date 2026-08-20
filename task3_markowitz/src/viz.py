"""
viz.py

Plotting utilities for Task 3:
  - efficient frontier (risk vs return), with individual assets overlaid
  - portfolio weight allocation across the frontier (stacked area)
  - single portfolio weight bar chart
"""

import numpy as np
import matplotlib.pyplot as plt

NAVY = "#1E2761"
BLUE = "#378ADD"
TEAL = "#1D9E75"
GOLD = "#C9922E"
CORAL = "#D85A30"
PALETTE = [NAVY, BLUE, TEAL, GOLD, CORAL]


def plot_efficient_frontier(frontier, min_var_result, asset_returns=None, asset_risks=None,
                              asset_names=None, save_path=None):
    risks = [p["risk"] for p in frontier]
    returns = [p["achieved_return"] for p in frontier]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(risks, returns, color=NAVY, linewidth=2, marker="o", markersize=3, label="Efficient frontier")
    ax.scatter([min_var_result["portfolio_risk"]], [min_var_result["portfolio_return"]],
               color=GOLD, s=120, zorder=5, edgecolor="black", label="Global min-variance portfolio")

    if asset_risks is not None and asset_returns is not None:
        ax.scatter(asset_risks, asset_returns, color=CORAL, s=60, zorder=4, label="Individual assets")
        if asset_names is not None:
            for name, r, ret in zip(asset_names, asset_risks, asset_returns):
                ax.annotate(name, (r, ret), fontsize=8, xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Risk (portfolio std dev)")
    ax.set_ylabel("Expected return")
    ax.set_title("Efficient Frontier (long-only)")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130)
    return fig


def plot_weight_allocation(frontier, asset_names, save_path=None):
    returns = [p["achieved_return"] for p in frontier]
    weights_matrix = np.array([p["weights"] for p in frontier])  # shape (n_points, n_assets)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stackplot(returns, weights_matrix.T, labels=asset_names, colors=PALETTE[:len(asset_names)], alpha=0.85)
    ax.set_xlabel("Target expected return")
    ax.set_ylabel("Portfolio weight")
    ax.set_title("Portfolio Composition Across the Efficient Frontier")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130)
    return fig


def plot_portfolio_weights(w, asset_names, title="Portfolio Weights", save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(asset_names, w, color=PALETTE[:len(asset_names)])
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.set_ylim(0, max(w.max() * 1.15, 0.1))
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130)
    return fig