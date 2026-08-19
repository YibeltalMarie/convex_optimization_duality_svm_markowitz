"""
svm_sklearn.py

LinearSVC benchmark wrapper -- same dataset, comparable settings, used to
cross-check weights, bias, and accuracy against the CVXPY primal solution.

Note on comparability: LinearSVC by default optimizes a different (but
equivalent-in-spirit) formulation, solved via a different algorithm
(coordinate descent, not a QP solver), with its own convergence tolerance.
Settings matched deliberately here:
    - loss='hinge'      matches our exact hinge-loss-equivalent formulation
                         (NOT squared-hinge, which is LinearSVC's default
                         and is a genuinely different problem)
    - C                  same penalty parameter, same meaning in both
    - fit_intercept=True matches our b variable
Exact numerical equality is still not expected -- see interpretation notes
in the analysis notebook.
"""

import numpy as np
from sklearn.svm import LinearSVC


def fit_svm_sklearn(X, y, C=1.0, max_iter=10000):
    """
    Fit scikit-learn's LinearSVC on the same data for comparison.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,), labels in {-1, +1}
    C : float
    max_iter : int
        LinearSVC's default (1000) can fail to converge for some C values;
        raised here so convergence failure is a real signal, not a silent
        artifact of too few iterations.

    Returns
    -------
    dict with keys:
        w         -- weight vector, shape (n_features,)
        b         -- bias/intercept, float
        model     -- the fitted LinearSVC object
        accuracy  -- training accuracy
    """
    model = LinearSVC(C=C, loss="hinge", fit_intercept=True, max_iter=max_iter)
    model.fit(X, y)

    w = model.coef_[0]
    b = model.intercept_[0]
    accuracy = model.score(X, y)

    return {
        "w": w,
        "b": b,
        "model": model,
        "accuracy": accuracy,
    }


if __name__ == "__main__":
    from dataset import generate_dataset
    from svm_primal import fit_svm_primal

    X, y = generate_dataset()

    cvxpy_result = fit_svm_primal(X, y, C=1.0)
    sklearn_result = fit_svm_sklearn(X, y, C=1.0)

    cvxpy_preds = np.sign(X @ cvxpy_result["w"] + cvxpy_result["b"])
    cvxpy_acc = (cvxpy_preds == y).mean()

    print("=== CVXPY primal ===")
    print(f"w: {cvxpy_result['w']}")
    print(f"b: {cvxpy_result['b']:.4f}")
    print(f"accuracy: {cvxpy_acc:.4f}")

    print("\n=== sklearn LinearSVC ===")
    print(f"w: {sklearn_result['w']}")
    print(f"b: {sklearn_result['b']:.4f}")
    print(f"accuracy: {sklearn_result['accuracy']:.4f}")

    w_diff = np.linalg.norm(cvxpy_result["w"] - sklearn_result["w"])
    b_diff = abs(cvxpy_result["b"] - sklearn_result["b"])
    print(f"\n||w_cvxpy - w_sklearn||: {w_diff:.4f}")
    print(f"|b_cvxpy - b_sklearn|: {b_diff:.4f}")