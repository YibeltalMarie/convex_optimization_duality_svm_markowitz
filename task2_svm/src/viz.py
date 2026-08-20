"""
viz.py

Decision boundary plotting utilities:
  - single model: data + decision boundary + margin lines + support vectors
  - side-by-side CVXPY vs LinearSVC comparison
  - hyperparameter (C) sweep: boundary grid + metric curves
"""

import numpy as np
import matplotlib.pyplot as plt

NAVY = "#1E2761"
TEAL = "#1D9E75"
BLUE = "#378ADD"
CORAL = "#D85A30"


def _bounds(X, pad=1.0):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    return x_min, x_max, y_min, y_max


def plot_decision_boundary(ax, X, y, w, b, sv_mask=None, title=""):
    """
    Plot data points, decision boundary, margin lines, and (optionally)
    highlighted support vectors, on a given matplotlib axis.
    """
    x_min, x_max, y_min, y_max = _bounds(X)

    ax.scatter(X[y == 1, 0], X[y == 1, 1], c=BLUE, edgecolor="white", s=45, label="Class +1", zorder=3)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c=CORAL, edgecolor="white", s=45, label="Class -1", zorder=3)

    if sv_mask is not None and sv_mask.any():
        ax.scatter(X[sv_mask, 0], X[sv_mask, 1], facecolors="none", edgecolors="black",
                   s=140, linewidths=1.5, label="Support vectors", zorder=4)

    # Decision boundary: w1*x1 + w2*x2 + b = 0  ->  x2 = -(w1*x1+b)/w2
    xx = np.linspace(x_min, x_max, 200)
    if abs(w[1]) > 1e-8:
        yy = -(w[0] * xx + b) / w[1]
        yy_margin_pos = -(w[0] * xx + b - 1) / w[1]
        yy_margin_neg = -(w[0] * xx + b + 1) / w[1]
        ax.plot(xx, yy, color=NAVY, linewidth=2, zorder=2, label="Decision boundary")
        ax.plot(xx, yy_margin_pos, color=TEAL, linewidth=1.2, linestyle="--", zorder=2)
        ax.plot(xx, yy_margin_neg, color=TEAL, linewidth=1.2, linestyle="--", zorder=2, label="Margin")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def plot_cvxpy_vs_sklearn(X, y, cvxpy_result, sklearn_result, sv_mask, C, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_decision_boundary(axes[0], X, y, cvxpy_result["w"], cvxpy_result["b"], sv_mask,
                            title=f"CVXPY primal SVM (C={C})")
    plot_decision_boundary(axes[1], X, y, sklearn_result["w"], sklearn_result["b"], None,
                            title=f"scikit-learn LinearSVC (C={C})")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130)
    return fig


def plot_c_sweep_boundaries(X, y, results_by_C, save_path=None):
    """results_by_C: dict {C: (w, b, sv_mask)}, in the order to display."""
    n = len(results_by_C)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2))
    if n == 1:
        axes = [axes]
    for ax, (C, (w, b, sv_mask)) in zip(axes, results_by_C.items()):
        plot_decision_boundary(ax, X, y, w, b, sv_mask, title=f"C = {C}")
    axes[0].legend(loc="upper left", fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130)
    return fig


def plot_c_sweep_metrics(C_values, accuracies, margins, n_violations, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(C_values, accuracies, marker="o", color=NAVY)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("C")
    axes[0].set_ylabel("Training accuracy")
    axes[0].set_title("Accuracy vs C")

    axes[1].plot(C_values, margins, marker="o", color=TEAL)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("C")
    axes[1].set_ylabel("Margin width (2/||w||)")
    axes[1].set_title("Margin width vs C")

    axes[2].plot(C_values, n_violations, marker="o", color=CORAL)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("C")
    axes[2].set_ylabel("Points with xi > 0")
    axes[2].set_title("Margin violations vs C")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130)
    return fig
