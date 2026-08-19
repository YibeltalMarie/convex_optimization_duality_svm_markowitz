"""
verification.py

Convergence and optimality verification for the CVXPY SVM solution:
  - solver status and objective sanity check
  - primal feasibility (constraint violations, checked directly, not just
    trusted from solver status)
  - KKT condition verification: stationarity, dual feasibility, and
    complementary slackness -- computed numerically against the fitted model
  - support vector identification via alpha_i (0 / (0,C) / C classification)
  - primal vs. dual objective comparison (a live strong-duality check)
"""

import numpy as np


def check_convergence(result, X, y, tol=1e-4):
    """
    Verify the solver actually converged to a feasible, optimal point,
    checked directly against the raw constraint definitions rather than
    just trusting problem.status.
    """
    w, b, xi = result["w"], result["b"], result["xi"]
    status = result["status"]

    margin_lhs = y * (X @ w + b)
    margin_violation = np.maximum(0, (1 - xi) - margin_lhs)  # should be ~0
    xi_violation = np.maximum(0, -xi)                          # should be ~0

    report = {
        "status": status,
        "is_optimal": status == "optimal",
        "objective_value": result["objective_value"],
        "max_margin_violation": margin_violation.max(),
        "max_xi_violation": xi_violation.max(),
        "primal_feasible": margin_violation.max() < tol and xi_violation.max() < tol,
    }
    return report


def get_duals(result):
    """Extract alpha_i (margin constraint) and mu_i (xi >= 0 constraint)."""
    alpha = result["margin_constraint"].dual_value
    mu = result["nonneg_constraint"].dual_value
    return alpha, mu


def verify_kkt(X, y, result, C, tol=1e-3):
    """
    Numerically verify all four KKT conditions against the fitted solution.
    tol is looser than machine precision -- OSQP solves to a numerical
    tolerance, not exactly, so small residuals are expected and fine.
    """
    w, b, xi = result["w"], result["b"], result["xi"]
    alpha, mu = get_duals(result)

    # --- Stationarity ---
    w_reconstructed = (alpha * y) @ X          # w = sum(alpha_i y_i x_i)
    stationarity_w_gap = np.linalg.norm(w - w_reconstructed)
    stationarity_b_gap = abs(np.sum(alpha * y))          # should be ~0
    stationarity_xi_gap = np.max(np.abs(alpha + mu - C))  # alpha_i + mu_i = C

    # --- Primal feasibility (recomputed directly) ---
    margin_lhs = y * (X @ w + b)
    primal_feasible = np.all(margin_lhs >= 1 - xi - tol) and np.all(xi >= -tol)

    # --- Dual feasibility ---
    dual_feasible = np.all(alpha >= -tol) and np.all(alpha <= C + tol) and np.all(mu >= -tol)

    # --- Complementary slackness (both pairs) ---
    slack_A = alpha * (margin_lhs - (1 - xi))     # should be ~0 for every i
    slack_B = mu * xi                              # should be ~0 for every i

    return {
        "stationarity_w_gap": stationarity_w_gap,
        "stationarity_b_gap": stationarity_b_gap,
        "stationarity_xi_gap": stationarity_xi_gap,
        "stationarity_ok": stationarity_w_gap < tol and stationarity_b_gap < tol and stationarity_xi_gap < tol,
        "primal_feasible": primal_feasible,
        "dual_feasible": dual_feasible,
        "max_complementary_slack_A": np.max(np.abs(slack_A)),
        "max_complementary_slack_B": np.max(np.abs(slack_B)),
        "complementary_slackness_ok": np.max(np.abs(slack_A)) < tol and np.max(np.abs(slack_B)) < tol,
    }


def classify_support_vectors(alpha, C, tol=1e-3):
    """Three-way classification straight from complementary slackness."""
    non_sv = alpha < tol
    margin_sv = (alpha >= tol) & (alpha <= C - tol)
    bound_sv = alpha > C - tol
    return {
        "non_support_vectors": int(non_sv.sum()),
        "margin_support_vectors": int(margin_sv.sum()),
        "bound_support_vectors": int(bound_sv.sum()),
        "non_sv_mask": non_sv,
        "margin_sv_mask": margin_sv,
        "bound_sv_mask": bound_sv,
    }


def dual_objective(X, y, alpha):
    """
    Evaluate the SVM dual objective at a given alpha:
        sum(alpha_i) - 1/2 * sum_i sum_j alpha_i alpha_j y_i y_j (x_i . x_j)
    """
    Gram = X @ X.T
    Q = np.outer(y, y) * Gram
    return alpha.sum() - 0.5 * alpha @ Q @ alpha


if __name__ == "__main__":
    from dataset import generate_dataset
    from svm_primal import fit_svm_primal

    X, y = generate_dataset()
    C = 1.0
    result = fit_svm_primal(X, y, C=C)

    print("=== Convergence check ===")
    conv = check_convergence(result, X, y)
    for k, v in conv.items():
        print(f"  {k}: {v}")

    print("\n=== KKT verification ===")
    kkt = verify_kkt(X, y, result, C)
    for k, v in kkt.items():
        print(f"  {k}: {v}")

    alpha, mu = get_duals(result)
    print("\n=== Support vector classification ===")
    sv = classify_support_vectors(alpha, C)
    print(f"  non-support vectors: {sv['non_support_vectors']}")
    print(f"  margin support vectors (0 < alpha < C): {sv['margin_support_vectors']}")
    print(f"  bound support vectors (alpha = C): {sv['bound_support_vectors']}")

    print("\n=== Strong duality check ===")
    d_obj = dual_objective(X, y, alpha)
    p_obj = result["objective_value"]
    print(f"  primal objective: {p_obj:.4f}")
    print(f"  dual objective:   {d_obj:.4f}")
    print(f"  duality gap:      {abs(p_obj - d_obj):.6f}")