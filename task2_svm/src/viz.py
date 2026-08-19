"""
viz.py

Visualization for Task 2:
  - decision boundary + margins + support-vector classification, for a
    single fitted model
  - side-by-side CVXPY vs. LinearSVC comparison (visual counterpart to the
    numerical agreement already shown in svm_sklearn.py)
  - C-sweep: margin width / support-vector counts / convergence status as
    a function of C, saved as both a figure and a printed table

All figures are written to task2_svm/results/figures/.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from dataset import generate_dataset
from svm_primal import fit_svm_primal
from svm_sklearn import fit_svm_sklearn
from verification import get_duals, classify_support_vectors

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "results/figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def _plot_boundary_on_ax(ax, X, y, w, b, sv_mask=None, title=""):
    """
    Draw points, decision boundary, and the two margin lines on a given
    matplotlib axis.

    sv_mask : optional dict of {label: boolean_mask}, e.g.
        {"margin SV": margin_sv_mask, "bound SV": bound_sv_mask}
    so different flavors of support vector can be marked distinctly rather
    than lumped into one generic "circled point."
    """
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:blue", s=25, label="+1", zorder=2)
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c="tab:red", s=25, label="-1", zorder=2)

    # decision boundary + margins: w^T x + b = 0, +1, -1
    xx1 = np.linspace(x1_min, x1_max, 200)
    if abs(w[1]) > 1e-12:
        boundary = (-w[0] * xx1 - b) / w[1]
        margin_pos = (-w[0] * xx1 - b + 1) / w[1]
        margin_neg = (-w[0] * xx1 - b - 1) / w[1]
        ax.plot(xx1, boundary, "k-", linewidth=1.5, label="decision boundary", zorder=1)
        ax.plot(xx1, margin_pos, "k--", linewidth=1, label="margin (+1)", zorder=1)
        ax.plot(xx1, margin_neg, "k--", linewidth=1, label="margin (-1)", zorder=1)

    if sv_mask is not None:
        markers = {"margin SV": ("o", "gold", 140), "bound SV": ("s", "darkorange", 110)}
        for label, mask in sv_mask.items():
            marker, color, size = markers.get(label, ("o", "gray", 120))
            ax.scatter(
                X[mask, 0], X[mask, 1],
                s=size, facecolors="none", edgecolors=color, linewidths=2,
                marker=marker, label=label, zorder=3,
            )

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)


def plot_single_model(X, y, result, C, save_name="decision_boundary_cvxpy.png"):
    """
    Decision boundary for the CVXPY primal solution, with the three-way
    KKT support-vector classification overlaid (margin SV vs. bound SV --
    these mean geometrically different things, so they're marked
    differently rather than as one generic 'support vector' category).
    """
    alpha, _ = get_duals(result)
    sv = classify_support_vectors(alpha, C)

    fig, ax = plt.subplots(figsize=(7, 6))
    _plot_boundary_on_ax(
        ax, X, y, result["w"], result["b"],
        sv_mask={"margin SV": sv["margin_sv_mask"], "bound SV": sv["bound_sv_mask"]},
        title="CVXPY primal SVM decision boundary (C={})".format(C),
    )
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_cvxpy_vs_sklearn(X, y, cvxpy_result, sklearn_result, C, save_name="cvxpy_vs_sklearn.png"):
    """
    Side-by-side boundaries: visual counterpart to the numerical
    ||w_cvxpy - w_sklearn|| comparison already computed in svm_sklearn.py.
    Support vectors are shown for the CVXPY panel only, since alpha_i (and
    therefore the margin/bound classification) comes from OUR dual
    variables -- LinearSVC's internal solve doesn't expose the same
    quantities in a directly comparable form.
    """
    alpha, _ = get_duals(cvxpy_result)
    sv = classify_support_vectors(alpha, C)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

    _plot_boundary_on_ax(
        axes[0], X, y, cvxpy_result["w"], cvxpy_result["b"],
        sv_mask={"margin SV": sv["margin_sv_mask"], "bound SV": sv["bound_sv_mask"]},
        title="CVXPY primal (QP)",
    )
    _plot_boundary_on_ax(
        axes[1], X, y, sklearn_result["w"], sklearn_result["b"],
        sv_mask=None,
        title="sklearn LinearSVC (coordinate descent)",
    )

    w_diff = np.linalg.norm(cvxpy_result["w"] - sklearn_result["w"])
    b_diff = abs(cvxpy_result["b"] - sklearn_result["b"])
    fig.suptitle(
        "C={}  |  ||w_cvxpy - w_sklearn|| = {:.4f}   |b_cvxpy - b_sklearn| = {:.4f}\n"
        "(boundaries agree visually; small numerical gap expected -- see svm_sklearn.py docstring)".format(
            C, w_diff, b_diff
        )
    )
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_c_sweep(X, y, C_values=None):
    """
    Fit the CVXPY primal SVM across a range of C and record, per C:
        - solver status (checked individually -- NOT assumed from one run)
        - margin width 2/||w||
        - number of margin SVs / bound SVs / non-SVs
        - objective value

    This is the piece we'd stopped at: an OSQP "solution may be inaccurate"
    warning appeared during an earlier interactive sweep, and the plan was
    to check convergence status per C value before trusting any of those
    numbers. Doing that explicitly here, per C, rather than trusting the
    sweep as a whole.
    """
    if C_values is None:
        # Wide log-spaced range so both the under-regularized (small C,
        # margin dominates, many bound SVs allowed) and over-regularized
        # (large C, hard-margin-like, ill-conditioning risk) regimes show up.
        C_values = np.logspace(-2, 3, 12)  # 0.01 ... 1000

    records = []
    for C in C_values:
        result = fit_svm_primal(X, y, C=C)
        status = result["status"]
        fix_applied = "none"

        # Graduated retry rather than silently discarding non-optimal points.
        # Two genuinely different failure modes were found by hand for this
        # dataset (see report.md): (1) default max_iter too small for some
        # mid-range C -- fixed by raising max_iter at the SAME tolerance;
        # (2) large C pushes the soft-margin QP toward its hard-margin limit,
        # which is a known hard regime for OSQP's first-order (ADMM) method --
        # fixed only by relaxing eps_abs/eps_rel, and flagged as such since
        # that's a real precision trade-off, not a free fix.
        if status != "optimal":
            import cvxpy as cp
            n, d = X.shape
            w_v = cp.Variable(d); b_v = cp.Variable(); xi_v = cp.Variable(n)
            obj = cp.Minimize(0.5 * cp.sum_squares(w_v) + C * cp.sum(xi_v))
            margin_c = cp.multiply(y, X @ w_v + b_v) >= 1 - xi_v
            nonneg_c = xi_v >= 0
            prob = cp.Problem(obj, [margin_c, nonneg_c])

            prob.solve(solver=cp.OSQP, max_iter=100000)
            if prob.status == "optimal":
                fix_applied = "raised max_iter (100000)"
            else:
                prob.solve(solver=cp.OSQP, max_iter=100000, eps_abs=1e-3, eps_rel=1e-3)
                if prob.status == "optimal":
                    fix_applied = "raised max_iter + relaxed tolerance (eps=1e-3)"

            if prob.status == "optimal":
                status = prob.status
                result = {
                    "w": w_v.value, "b": b_v.value, "xi": xi_v.value,
                    "status": status, "objective_value": prob.value,
                    "margin_constraint": margin_c, "nonneg_constraint": nonneg_c,
                }

        is_optimal = status == "optimal"

        record = {
            "C": C,
            "status": status,
            "is_optimal": is_optimal,
            "fix_applied": fix_applied,
            "objective_value": result["objective_value"],
        }

        if is_optimal and result["w"] is not None:
            w, b = result["w"], result["b"]
            alpha, _ = get_duals(result)
            sv = classify_support_vectors(alpha, C)
            record.update({
                "margin_width": 2.0 / np.linalg.norm(w),
                "w_norm": np.linalg.norm(w),
                "n_margin_sv": sv["margin_support_vectors"],
                "n_bound_sv": sv["bound_support_vectors"],
                "n_non_sv": sv["non_support_vectors"],
            })
        else:
            # Solver did not report "optimal" for this C -- do not silently
            # compute derived quantities from a possibly-invalid solution.
            record.update({
                "margin_width": np.nan, "w_norm": np.nan,
                "n_margin_sv": np.nan, "n_bound_sv": np.nan, "n_non_sv": np.nan,
            })

        records.append(record)

    return records


def plot_c_sweep(records, save_name="c_sweep.png"):
    """Two-panel figure: margin width vs C, and SV counts vs C (log-x)."""
    C_vals = [r["C"] for r in records]
    margin_widths = [r["margin_width"] for r in records]
    n_margin = [r["n_margin_sv"] for r in records]
    n_bound = [r["n_bound_sv"] for r in records]
    not_optimal = [r["C"] for r in records if not r["is_optimal"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(C_vals, margin_widths, "o-", color="tab:purple")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("C (log scale)")
    axes[0].set_ylabel("margin width  2/||w||")
    axes[0].set_title("Margin width shrinks as C grows\n(less tolerance for slack -> tighter fit)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(C_vals, n_margin, "o-", label="margin SVs (0<alpha<C)", color="gold")
    axes[1].plot(C_vals, n_bound, "s-", label="bound SVs (alpha=C)", color="darkorange")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("C (log scale)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Support vector composition vs. C")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    if not_optimal:
        for ax in axes:
            for C in not_optimal:
                ax.axvline(C, color="red", linestyle=":", alpha=0.5)
        fig.suptitle(
            "Red dotted lines mark C values where solver status != 'optimal': {}".format(not_optimal),
            color="red",
        )

    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, save_name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    X, y = generate_dataset()
    C = 1.0

    cvxpy_result = fit_svm_primal(X, y, C=C)
    sklearn_result = fit_svm_sklearn(X, y, C=C)

    p1 = plot_single_model(X, y, cvxpy_result, C)
    print("Saved: {}".format(p1))

    p2 = plot_cvxpy_vs_sklearn(X, y, cvxpy_result, sklearn_result, C)
    print("Saved: {}".format(p2))

    print("\n=== C-sweep: per-C convergence check ===")
    records = run_c_sweep(X, y)
    print("{:>10} {:>10} {:>13} {:>10} {:>9}  {}".format(
        "C", "status", "margin_width", "margin_sv", "bound_sv", "fix_applied"))
    for r in records:
        mw = "{:.4f}".format(r["margin_width"]) if r["is_optimal"] else "  --  "
        msv = r["n_margin_sv"] if r["is_optimal"] else "--"
        bsv = r["n_bound_sv"] if r["is_optimal"] else "--"
        print("{:>10.4f} {:>10} {:>13} {:>10} {:>9}  {}".format(
            r["C"], r["status"], mw, str(msv), str(bsv), r["fix_applied"]))

    still_failing = [r["C"] for r in records if not r["is_optimal"]]
    retried = [r["C"] for r in records if r["fix_applied"] != "none"]
    if still_failing:
        print("\nWARNING: even after retry, solver did not reach 'optimal' for C = {}".format(still_failing))
        print("These points are excluded from the sweep plot's trend lines.")
    if retried:
        print("\nC values that needed a retry: {}".format(retried))
        print("See per-row 'fix_applied' column above for which fix worked.")
    if not still_failing and not retried:
        print("\nAll C values converged with status 'optimal' on the first solve.")

    p3 = plot_c_sweep(records)
    print("\nSaved: {}".format(p3))