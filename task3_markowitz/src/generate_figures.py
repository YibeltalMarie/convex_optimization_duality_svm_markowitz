"""
generate_figures.py

Driver script that ties together data.py, markowitz.py, frontier.py, and
viz.py to produce every figure required for Task 3, saved into
../results/figures/.

Run from inside task3_markowitz/src/:
    python generate_figures.py
"""

import os
import numpy as np

from data import get_covariance_matrix, EXPECTED_RETURNS, VOLATILITIES, ASSET_NAMES
from markowitz import fit_min_variance_portfolio
from frontier import compute_efficient_frontier
from viz import plot_efficient_frontier, plot_weight_allocation, plot_portfolio_weights

FIGURES_DIR = "../results/figures"


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    Sigma, eigenvalues, is_psd = get_covariance_matrix()
    assert is_psd, "Covariance matrix is not PSD -- fix data.py before proceeding."
    print(f"Covariance matrix verified PSD. Eigenvalues: {np.round(eigenvalues, 6)}")

    # --- Global minimum-variance portfolio ---
    min_var_result = fit_min_variance_portfolio(EXPECTED_RETURNS, Sigma)
    print(f"\nGlobal min-variance portfolio (status={min_var_result['status']}):")
    for name, w in zip(ASSET_NAMES, min_var_result["w"]):
        print(f"  {name:20s}: {w:.4f}")
    print(f"  return: {min_var_result['portfolio_return']:.4f}, risk: {min_var_result['portfolio_risk']:.4f}")

    plot_portfolio_weights(
        min_var_result["w"], ASSET_NAMES,
        title="Global Minimum-Variance Portfolio",
        save_path=f"{FIGURES_DIR}/01_min_variance_weights.png",
    )
    print("saved 01_min_variance_weights.png")

    # --- Efficient frontier ---
    frontier, _ = compute_efficient_frontier(EXPECTED_RETURNS, Sigma, n_points=25)
    n_failed = 25 - len(frontier)
    print(f"\nFrontier points solved: {len(frontier)} / 25 (failed: {n_failed})")

    plot_efficient_frontier(
        frontier, min_var_result,
        asset_returns=EXPECTED_RETURNS, asset_risks=VOLATILITIES, asset_names=ASSET_NAMES,
        save_path=f"{FIGURES_DIR}/02_efficient_frontier.png",
    )
    print("saved 02_efficient_frontier.png")

    plot_weight_allocation(
        frontier, ASSET_NAMES,
        save_path=f"{FIGURES_DIR}/03_weight_allocation.png",
    )
    print("saved 03_weight_allocation.png")

    # --- A higher-return sample portfolio for contrast ---
    high_return_target = EXPECTED_RETURNS.max() * 0.85
    high_return_result = fit_min_variance_portfolio(EXPECTED_RETURNS, Sigma, target_return=high_return_target)
    plot_portfolio_weights(
        high_return_result["w"], ASSET_NAMES,
        title=f"Higher-Return Portfolio (target return={high_return_target:.2f})",
        save_path=f"{FIGURES_DIR}/04_high_return_weights.png",
    )
    print(f"saved 04_high_return_weights.png (status={high_return_result['status']})")

    print("\nAll figures written to", FIGURES_DIR)


if __name__ == "__main__":
    main()