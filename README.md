# Convex Optimization

Convex optimization from first principles, applied across three tasks: duality theory,
Soft-Margin SVM, and Markowitz portfolio optimization. Built to demonstrate genuine
understanding — not just working code — with every claim (convexity, convergence, KKT
conditions, strong duality) checked numerically against the actual implementations rather
than only asserted theoretically.

## Structure
```
.
├── task1_duality/
│ └── duality_presentation.pptx
├── task2_svm/
│ ├── src/
│ ├── notebooks/svm_analysis.ipynb
│ ├── results/figures/
│ └── report.md
├── task3_markowitz/
│ ├── src/
│ ├── notebooks/markowitz_analysis.ipynb
│ ├── results/figures/
│ └── report.md
└── requirements.txt
```

## Task 1 — The Principle of Duality

A 5–10 minute presentation on primal/dual formulations, weak & strong duality, and the KKT
conditions, built from first principles (Lagrangian → dual function → weak duality proof →
strong duality via convexity + Slater's condition → KKT as a certificate of optimality).
Includes two "beyond the requirement" sections: the shadow-price interpretation of the
optimal multiplier, and the geometric (supporting-hyperplane) view of duality, with a
worked non-convex counterexample showing where strong duality breaks down.

See `task1_duality/duality_presentation.pptx`. Speaker notes contain the full derivations
and anticipated mentor Q&A.

## Task 2 — Soft-Margin SVM as a Convex QP

The primal Soft-Margin SVM, derived from the maximum-margin geometric idea through to the
CVXPY implementation, benchmarked against scikit-learn's `LinearSVC`.

**Beyond the core requirement:**
- Numerical KKT verification (stationarity, feasibility, complementary slackness) checked
  directly against the fitted model, not just asserted from theory
- Support vector identification via the recovered dual variables $\alpha_i$, using the
  three-way classification (non-SV / margin-SV / bound-SV) that falls out of complementary
  slackness
- A live strong-duality check: primal objective vs. dual objective on the actual fitted
  model (duality gap ≈ 0)
- A held-out train/test generalization check across the $C$ sweep, not training accuracy
  alone

**Reproduce:**
```bash
cd task2_svm/src
python dataset.py          # confirm dataset
python svm_primal.py       # confirm CVXPY solve
python svm_sklearn.py      # confirm benchmark
python verification.py     # confirm KKT + strong duality
jupyter nbconvert --to notebook --execute --inplace ../notebooks/svm_analysis.ipynb
```

See `task2_svm/report.md` for full results and interpretation.

## Task 3 — Markowitz Mean-Variance Portfolio Optimization

Long-only, fully-invested minimum-variance portfolio optimization over a 5-asset universe,
plus the full efficient frontier. Covariance matrix built by hand and explicitly verified
PSD via its eigenvalues (reusing the same convexity check from Task 1/2) before being
trusted.

**Notable finding:** the diversification benefit is visible directly in the optimizer's
output, not just asserted — the Gold/Hedge asset receives a real allocation in the
minimum-variance portfolio despite mediocre standalone return, because of its negative
correlation with the growth stocks.

**Reproduce:**
```bash
cd task3_markowitz/src
python data.py              # confirm covariance matrix is PSD
python markowitz.py         # confirm min-variance solve
python frontier.py          # confirm efficient frontier
python generate_figures.py  # regenerate all figures into ../results/figures/
python build_notebook.py    # (one-time) construct the notebook
jupyter nbconvert --to notebook --execute --inplace ../notebooks/markowitz_analysis.ipynb
```

See `task3_markowitz/report.md` for full results and interpretation.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Common thread across all three tasks

Every task is built on the same theoretical spine: convexity (checked via the Hessian /
eigenvalues, not assumed) guarantees that any locally optimal point a solver finds is
globally optimal; strong duality (via Slater's condition) guarantees the dual problem gives
the exact same answer as the primal; and the KKT conditions tie both together into a
verifiable certificate of optimality. Task 2 and Task 3 each apply this same machinery to a
genuinely different problem — classification margins and portfolio risk — and both include
explicit numerical checks confirming the theory holds on real, fitted models rather than only in the abstract.