# Task 2 — Soft-Margin SVM

Primal Soft-Margin SVM formulated as a convex Quadratic Program in CVXPY,
benchmarked against scikit-learn's LinearSVC, with numerical KKT verification
and a live strong-duality check.

## Structure
- `src/dataset.py` — synthetic 2D dataset generation (make_blobs, moderate overlap)
- `src/svm_primal.py` — CVXPY primal SVM implementation (CLARABEL solver)
- `src/svm_sklearn.py` — LinearSVC benchmark wrapper
- `src/verification.py` — convergence, KKT, support vector classification, dual objective
- `src/experiments.py` — held-out train/test split C-sweep (generalization check)
- `src/viz.py` — decision boundary and hyperparameter-sweep plots
- `notebooks/svm_analysis.ipynb` — full walkthrough and interpretation (pre-executed)
- `results/figures/` — saved output plots
- `results/c_sweep_summary.json` — raw C-sweep numbers
- `report.md` — standalone write-up: formulation, results, interpretation, limitations

## Run
Open `notebooks/svm_analysis.ipynb` and run top to bottom, or run any `src/*.py` module
directly (each has a `__main__` block that reproduces its part of the analysis).

## Key results (C=1.0)
- CVXPY primal and LinearSVC agree to ~0.005 in weights, identical training accuracy (0.9467)
- All four KKT conditions verified numerically to ~1e-10
- Duality gap: ~1e-9 (primal objective == dual objective)
- 22 support vectors identified via alpha_i (3 margin, 19 bound), matching slack count exactly
