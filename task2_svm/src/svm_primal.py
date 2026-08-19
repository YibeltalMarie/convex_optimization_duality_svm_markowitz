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


def fit_svm_primal(X, y, C=1.0, solver=cp.OSQP, verbose=False):
    """
    Solve the primal Soft-Margin SVM QP.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,), labels in {-1, +1}
    C : float
        Penalty parameter trading off margin width against total slack.
    solver : cvxpy solver
        OSQP is a solver specialized for QPs -- a good default here since
        we've proven this problem is exactly that.
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
        "b": b.value,
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