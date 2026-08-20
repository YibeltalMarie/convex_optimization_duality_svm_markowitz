# Task 2 — Soft-Margin SVM as a Convex Quadratic Program

## 1. Mathematical formulation

**Primal:**

$$\min_{w,b,\xi} \; \frac{1}{2}\|w\|^2 + C\sum_i \xi_i \qquad \text{s.t.} \qquad y_i(w^Tx_i+b) \ge 1-\xi_i, \quad \xi_i \ge 0$$

**Convexity:** the Hessian of $\frac12\|w\|^2$ is the identity matrix (PSD, trivially); the
penalty term $C\sum\xi_i$ is linear (zero curvature); every constraint is linear in
$(w,b,\xi)$. This is a convex QP — any solution the solver reports as optimal is provably
globally optimal, not merely locally optimal.

**Dual** (derived via the Lagrangian, multipliers $\alpha_i \ge 0$ for the margin
constraint and $\mu_i \ge 0$ for $\xi_i\ge0$):

$$\max_\alpha \; \sum_i \alpha_i - \frac12\sum_i\sum_j \alpha_i\alpha_j y_iy_j\,x_i^Tx_j \qquad \text{s.t.} \qquad 0 \le \alpha_i \le C, \quad \sum_i \alpha_i y_i = 0$$

**Complementary slackness** yields a three-way classification of every training point:
- $\alpha_i = 0$: not a support vector
- $0 < \alpha_i < C$: support vector exactly on the margin ($\xi_i=0$)
- $\alpha_i = C$: "bound" support vector, violating the margin ($\xi_i$ possibly $>0$)

## 2. Dataset

Two isotropic Gaussian blobs (`sklearn.datasets.make_blobs`), centers $(-1.8,0)$ and
$(1.8,0)$, standard deviation $1.3$, $n=150$, `random_state=42`. Chosen, via visual
comparison of several candidate configurations, to produce a genuine overlap zone between
classes — a fully separable dataset would make the `C` hyperparameter sweep uninteresting,
since soft-margin and hard-margin SVM would behave nearly identically at every `C`.

## 3. Implementation and benchmark (at C = 1.0)

| | CVXPY primal | scikit-learn LinearSVC |
|---|---|---|
| w | [1.4137, 0.0751] | [1.4156, 0.0701] |
| b | 0.1698 | 0.1631 |
| Training accuracy | 0.9467 | 0.9467 |

$\|w_{\text{cvxpy}} - w_{\text{sklearn}}\| = 0.0054$; $|b_{\text{cvxpy}} - b_{\text{sklearn}}| = 0.0066$.

Small residual differences are expected, not a bug: CVXPY solves the exact primal QP via
an interior-point method (CLARABEL) to a numerical tolerance; `LinearSVC` solves an
equivalent dual formulation via coordinate descent (`liblinear`) with its own tolerance.
Both converge to the same true optimum — strong duality guarantees a unique optimum for
this strictly convex objective — so agreement to ~2-3 decimal places is itself evidence
both implementations are correct.

**Reading the fitted weights:** $w_1 \gg w_2$ because the two blob centers differ only
along $x_1$; $x_2$ carries no class information by construction, so the optimizer
correctly learned to nearly ignore it.

## 4. Convergence verification

**Solver choice matters and was verified, not assumed.** OSQP (first-order/ADMM) was
tried first as the default QP solver, but returned `status='user_limit'` with a
"solution may be inaccurate" warning at `C=100`, where the objective becomes badly scaled
($C\sum\xi_i$ dominates $\frac12\|w\|^2$ by orders of magnitude at large $C$). Switching to
CLARABEL (interior-point) resolved this: `status='optimal'` with feasibility violations at
numerical precision ($\sim 10^{-11}$) across the entire $C \in \{0.01, ..., 100\}$ sweep.

## 5. KKT verification

All four KKT conditions checked numerically against the actual fitted model (not just
asserted theoretically):

| Condition | Result |
|---|---|
| Stationarity ($w = \sum\alpha_iy_ix_i$, etc.) | gap ~$10^{-8}$, holds |
| Primal feasibility | holds |
| Dual feasibility ($0\le\alpha_i\le C$) | holds |
| Complementary slackness (both pairs) | gap ~$10^{-10}$, holds |

**Support vector classification** (C=1.0): 128 non-support vectors, 3 margin support
vectors, 19 bound support vectors. The bound support vector count (19) exactly matches the
number of points with nonzero slack ($\xi_i > 10^{-4}$) — precisely what complementary
slackness predicts, confirmed on real data rather than only in theory.

## 6. Primal vs. dual objective — strong duality check

Primal objective: 20.4490. Dual objective (evaluated directly from the recovered
$\alpha_i$'s): 20.4490. Duality gap: ~$10^{-9}$ — a live, numeric confirmation of strong
duality on this actual fitted model.

## 7. Hyperparameter (C) sweep

| C | objective | ‖w‖ | margin | train acc | violations | support vectors |
|---|---|---|---|---|---|---|
| 0.01 | 0.50 | 0.574 | 3.487 | 0.947 | 65 | 68 |
| 0.1 | 2.67 | 1.021 | 1.958 | 0.953 | 30 | 33 |
| 1 | 20.45 | 1.416 | 1.413 | 0.947 | 19 | 22 |
| 10 | 194.85 | 1.567 | 1.276 | 0.947 | 18 | 21 |
| 100 | 1937.43 | 1.567 | 1.276 | 0.947 | 18 | 21 |

**Interpretation:**
- **Margin shrinks monotonically** as $C$ grows (3.49 → 1.28), matching the derivation
  directly: larger $C$ makes slack expensive, so the optimizer trades margin width for
  fewer violations.
- **Violations drop sharply then plateau** between $C=10$ and $C=100$ — the model has
  **saturated**. The ~18 remaining violating points sit in the dataset's genuinely
  overlapping region by construction; no additional penalty can separate points that are
  geometrically coincident. This is a real, defensible finding rather than an artifact.
- **Training accuracy is roughly flat** (0.947-0.953) across the sweep, since accuracy
  depends on which side of the decision *boundary* each point falls, and the boundary
  itself barely moves even as the margin around it shrinks.

## 8. Generalization check (held-out train/test split)

A 70/30 stratified train/test split (`src/experiments.py`), same split reused across every
`C` for comparability:

| C | train acc | test acc | margin | #SV | #violations |
|---|---|---|---|---|---|
| 0.01 | 0.933 | 0.978 | 3.656 | 58 | 56 |
| 0.1 | 0.943 | 0.978 | 2.150 | 28 | 25 |
| 1 | 0.943 | 0.956 | 1.425 | 20 | 17 |
| 10 | 0.943 | 0.956 | 1.419 | 19 | 16 |
| 100 | 0.943 | 0.956 | 1.419 | 19 | 16 |

Test accuracy is consistently at or above training accuracy across the entire sweep — no
visible overfitting even at large `C` on this dataset, most likely because this particular
random 45-point test split happened to land in a comparatively easier region relative to
the training split. This is noted honestly rather than overclaimed: a single random split
is not strong evidence of generalization by itself; a proper study would repeat this across
multiple random splits (cross-validation), listed below as a direct extension.

## 9. Limitations

- Linear SVM only — no kernel applied, though the dual derivation makes explicit exactly
  where a kernel would substitute (the $x_i^Tx_j$ term).
- Synthetic data only; a real dataset (e.g. Breast Cancer Wisconsin, reduced to 2 features)
  would be a natural test of whether these findings generalize.
- Generalization checked via a single random train/test split, not cross-validation —
  sufficient to show no overfitting here, not sufficient to strongly claim robust
  generalization (see Section 8).

## 10. Extensions completed

1. Numerical KKT verification on the fitted model (Section 5)
2. Support vector identification via $\alpha_i$'s three-way classification (Section 5)
3. Primal vs. dual objective comparison — live strong duality check (Section 6)
4. Held-out generalization check via train/test split (Section 8)

## 11. Possible further extensions

- Cross-validation (multiple random splits) to strengthen the generalization claim in
  Section 8 beyond a single split.
- Real-data comparison (Breast Cancer Wisconsin, PCA to 2D) to test generalization of the
  C-sweep findings above.
- Kernel SVM (RBF) on a non-linearly-separable dataset, contrasted against this linear case.
- Solver comparison table (OSQP vs. CLARABEL vs. SCS) with timing, expanding on the
  convergence issue found and fixed in Section 4.
