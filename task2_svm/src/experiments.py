"""
experiments.py

Hyperparameter (C) sweep experiment. Fits the primal SVM across a range of
C values on a train split and evaluates on a held-out test split, so the
overfitting/underfitting trade-off predicted by the C exchange-rate argument
can actually be observed from data, not just asserted from theory.

A train/test split is introduced specifically here (not used elsewhere in
Task 2) because generalization is the whole point of this experiment --
training-set-only accuracy cannot distinguish "fits well" from "memorized".
"""

import numpy as np
from sklearn.model_selection import train_test_split

from svm_primal import fit_svm_primal
from verification import get_duals, classify_support_vectors


def run_C_sweep(X, y, C_values, test_size=0.3, random_state=42):
    """
    Parameters
    ----------
    X, y : full dataset
    C_values : list of float
    test_size : float
        Fraction held out for testing generalization.
    random_state : int
        Shared seed -- same split used for every C, so results are comparable.

    Returns
    -------
    results : list of dict, one per C value
    split : (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results = []
    for C in C_values:
        r = fit_svm_primal(X_train, y_train, C=C)
        w, b, xi = r["w"], r["b"], r["xi"]

        margin_width = 2 / np.linalg.norm(w)

        train_preds = np.sign(X_train @ w + b)
        train_acc = (train_preds == y_train).mean()

        test_preds = np.sign(X_test @ w + b)
        test_acc = (test_preds == y_test).mean()

        n_violations = int((xi > 1e-4).sum())

        alpha, _ = get_duals(r)
        sv = classify_support_vectors(alpha, C)
        n_sv = sv["margin_support_vectors"] + sv["bound_support_vectors"]

        results.append({
            "C": C,
            "w": w, "b": b,
            "margin_width": margin_width,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "num_violations": n_violations,
            "n_support_vectors": n_sv,
            "objective_value": r["objective_value"],
        })

    return results, (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    from dataset import generate_dataset

    X, y = generate_dataset()
    C_values = [0.01, 0.1, 1, 10, 100]
    results, split = run_C_sweep(X, y, C_values)

    print(f"{'C':>8} {'train_acc':>10} {'test_acc':>9} {'margin':>8} {'#SV':>5} {'#viol':>6}")
    for r in results:
        print(f"{r['C']:>8} {r['train_accuracy']:>10.4f} {r['test_accuracy']:>9.4f} "
              f"{r['margin_width']:>8.4f} {r['n_support_vectors']:>5} {r['num_violations']:>6}")
