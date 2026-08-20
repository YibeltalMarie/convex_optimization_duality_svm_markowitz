"""
frontier.py

Efficient frontier: sweep a range of target returns and solve the long-only
minimum-variance portfolio at each, tracing out the risk-return trade-off.
"""

import numpy as np
from markowitz import fit_min_variance_portfolio


def compute_efficient_frontier(mu, Sigma, n_points=25, solver=None):
    """
    Sweep target returns from the global minimum-variance portfolio's return
    up to the highest single-asset return (the long-only ceiling), solving
    the minimum-variance problem at each target.

    Returns
    -------
    frontier : list of dict, one per successfully solved point:
        target_return, achieved_return, risk, weights, status
    min_var_result : dict
        The unconstrained-return (global) minimum-variance portfolio result,
        used as the frontier's starting anchor point.
    """
    min_var_result = fit_min_variance_portfolio(mu, Sigma)
    min_return = min_var_result["portfolio_return"]
    max_return = mu.max()  # cannot exceed 100% allocation to the best single asset (long-only)

    target_returns = np.linspace(min_return, max_return, n_points)

    frontier = []
    for target in target_returns:
        kwargs = {"solver": solver} if solver is not None else {}
        result = fit_min_variance_portfolio(mu, Sigma, target_return=target, **kwargs)
        if result["status"] == "optimal":
            frontier.append({
                "target_return": float(target),
                "achieved_return": result["portfolio_return"],
                "risk": result["portfolio_risk"],
                "weights": result["w"],
                "status": result["status"],
            })
    return frontier, min_var_result


if __name__ == "__main__":
    from data import get_covariance_matrix, EXPECTED_RETURNS, ASSET_NAMES

    Sigma, eigenvalues, is_psd = get_covariance_matrix()
    frontier, min_var_result = compute_efficient_frontier(EXPECTED_RETURNS, Sigma)

    print("Global minimum-variance portfolio:")
    print(f"  return: {min_var_result['portfolio_return']:.4f}, risk: {min_var_result['portfolio_risk']:.4f}")
    print(f"  weights: {dict(zip(ASSET_NAMES, np.round(min_var_result['w'], 4)))}")

    print(f"\nFrontier points computed: {len(frontier)} / requested")
    print(f"{'return':>8} {'risk':>8}")
    for p in frontier[::5]:
        print(f"{p['achieved_return']:>8.4f} {p['risk']:>8.4f}")