"""
markowitz.py

Markowitz Mean-Variance portfolio optimization, formulated and solved as a
convex Quadratic Program using CVXPY:

    minimize    w^T Sigma w
    subject to  sum(w) = 1                     (exact budget constraint)
                w >= 0                          (long-only)
                mu^T w >= target_return          (optional, for frontier sweep)

Convex because Sigma is PSD by construction (any genuine covariance matrix
is), and the constraints are linear -- the same convex-QP guarantee
established for SVM in Task 2 applies here.
"""

import numpy as np
import cvxpy as cp


def fit_min_variance_portfolio(mu, Sigma, target_return=None, solver=cp.CLARABEL, verbose=False):
    """
    Solve the long-only, fully-invested minimum-variance portfolio, optionally
    subject to a target expected return (used to trace the efficient frontier).

    Parameters
    ----------
    mu : ndarray (n_assets,)
        Expected returns.
    Sigma : ndarray (n_assets, n_assets)
        Covariance matrix (must be PSD -- see data.py's verification).
    target_return : float or None
        If given, adds the constraint mu^T w >= target_return.
    solver : cvxpy solver
        CLARABEL (interior-point), matching the choice validated in Task 2.
    verbose : bool

    Returns
    -------
    dict with keys:
        w                    -- optimal weights, shape (n_assets,)
        status                -- solver status string
        objective_value       -- optimal portfolio variance (w^T Sigma w)
        portfolio_return      -- mu^T w at the optimum
        portfolio_risk        -- sqrt(variance), i.e. portfolio std dev
        problem                -- the cvxpy.Problem object
        budget_constraint      -- sum(w) == 1 constraint (for dual/KKT checks)
        nonneg_constraint      -- w >= 0 constraint
        return_constraint      -- mu^Tw >= target_return constraint, or None
    """
    n = len(mu)
    w = cp.Variable(n)

    objective = cp.Minimize(cp.quad_form(w, Sigma))

    budget_constraint = cp.sum(w) == 1
    nonneg_constraint = w >= 0
    constraints = [budget_constraint, nonneg_constraint]

    return_constraint = None
    if target_return is not None:
        return_constraint = mu @ w >= target_return
        constraints.append(return_constraint)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver, verbose=verbose)

    w_val = w.value
    portfolio_variance = problem.value
    portfolio_return = float(mu @ w_val) if w_val is not None else None
    portfolio_risk = float(np.sqrt(portfolio_variance)) if portfolio_variance is not None and portfolio_variance >= 0 else None

    return {
        "w": w_val,
        "status": problem.status,
        "objective_value": portfolio_variance,
        "portfolio_return": portfolio_return,
        "portfolio_risk": portfolio_risk,
        "problem": problem,
        "budget_constraint": budget_constraint,
        "nonneg_constraint": nonneg_constraint,
        "return_constraint": return_constraint,
    }


if __name__ == "__main__":
    from data import get_covariance_matrix, EXPECTED_RETURNS, ASSET_NAMES

    Sigma, eigenvalues, is_psd = get_covariance_matrix()
    assert is_psd, "Covariance matrix is not PSD -- cannot proceed."

    result = fit_min_variance_portfolio(EXPECTED_RETURNS, Sigma)

    print(f"Status: {result['status']}")
    print(f"Portfolio variance: {result['objective_value']:.6f}")
    print(f"Portfolio risk (std dev): {result['portfolio_risk']:.4f}")
    print(f"Portfolio expected return: {result['portfolio_return']:.4f}")
    print("\nWeights:")
    for name, weight in zip(ASSET_NAMES, result["w"]):
        print(f"  {name:20s}: {weight:.4f}")
    print(f"\nSum of weights: {result['w'].sum():.6f}")
    print(f"All weights >= 0 (long-only respected)? {np.all(result['w'] >= -1e-8)}")