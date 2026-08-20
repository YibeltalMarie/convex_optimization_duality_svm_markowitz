"""
svm_primal.py

Primal Soft-Margin SVM, formulated and solved as a convex Quadratic Program
using CVXPY:

    minimize    (1/2) ||w||^2 + C * sum(xi_i)
    subject to  y_i (w^T x_i + b) >= 1 - xi_i    for all i
                xi_i >= 0                         for all i

This is a convex QP: the objective's Hessian w.r.t. w is the identity matrix
(trivially PSD), the C*sum(xi) term is linear (zero curvature), and every
constraint is linear in (w, b, xi) -- so the feasible region is convex too.
"""

import numpy as np
import cvxpy as cp


def fit_svm_primal(X, y, C=1.0, solver=cp.CLARABEL, verbose=False):
    """
    Solve the primal Soft-Margin SVM QP.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,), labels in {-1, +1}
    C : float
        Penalty parameter trading off margin width against total slack.
    solver : cvxpy solver
        CLARABEL (interior-point) is the default. OSQP (first-order/ADMM)
        was tried first as the more "obvious" QP solver, but was observed to
        report "solution may be inaccurate" / status "user_limit" at large C
        (e.g. C=100), where the objective becomes badly scaled since
        C*sum(xi) then dominates (1/2)||w||^2 by orders of magnitude.
        CLARABEL's interior-point method converged cleanly (status "optimal",
        feasibility violations ~1e-12) across the entire C sweep used in
        this project -- see verification.py / the analysis notebook for the
        convergence check that motivated this choice.
    verbose : bool
        If True, print solver iteration output.

    Returns
    -------
    dict with keys:
        w, b, xi            -- fitted parameters
        status               -- solver status string (should be "optimal")
        objective_value      -- optimal value of (1/2)||w||^2 + C*sum(xi)
        problem               -- the cvxpy.Problem object, for further inspection
        margin_constraint     -- the y_i(w^Tx_i+b) >= 1-xi_i constraint object
                                  (its .dual_value gives the KKT alpha_i's)
        nonneg_constraint     -- the xi >= 0 constraint object
    """
    n_samples, n_features = X.shape

    w = cp.Variable(n_features)
    b = cp.Variable()
    xi = cp.Variable(n_samples)

    objective = cp.Minimize(0.5 * cp.sum_squares(w) + C * cp.sum(xi))

    # cp.multiply does elementwise multiplication -- this is one vectorized
    # constraint bundling all n_samples inequalities y_i(w^Tx_i+b) >= 1-xi_i.
    # Its .dual_value after solving returns one multiplier per point (the
    # alpha_i's from the Lagrangian dual), which we'll use later for KKT
    # verification and support vector identification.
    margin_constraint = cp.multiply(y, X @ w + b) >= 1 - xi
    nonneg_constraint = xi >= 0

    problem = cp.Problem(objective, [margin_constraint, nonneg_constraint])
    problem.solve(solver=solver, verbose=verbose)

    return {
        "w": w.value,
        "b": float(b.value),  # CLARABEL returns a 0-d ndarray here, not a plain float;
                               # cast explicitly so downstream round()/f-string usage works
        "xi": xi.value,
        "status": problem.status,
        "objective_value": problem.value,
        "problem": problem,
        "margin_constraint": margin_constraint,
        "nonneg_constraint": nonneg_constraint,
    }


if __name__ == "__main__":
    from dataset import generate_dataset

    X, y = generate_dataset()
    result = fit_svm_primal(X, y, C=1.0)

    print(f"Status: {result['status']}")
    print(f"Objective value: {result['objective_value']:.4f}")
    print(f"w: {result['w']}")
    print(f"b: {result['b']:.4f}")
    print(f"||w||: {np.linalg.norm(result['w']):.4f}")
    print(f"Margin width (2/||w||): {2 / np.linalg.norm(result['w']):.4f}")
    n_violations = (result["xi"] > 1e-4).sum()
    print(f"Points with xi > 1e-4 (inside margin or misclassified): {n_violations} / {len(y)}")