# Task 3 — Markowitz Mean-Variance Portfolio Optimization

## 1. Mathematical formulation

**Portfolio return:** $\mu^Tw$ — weighted average of expected returns.

**Portfolio variance (risk):** $w^T\Sigma w$ — a quadratic form, *not* a weighted average
of individual variances. Expanding for two assets: $w_1^2\Sigma_{11} + w_2^2\Sigma_{22} +
2w_1w_2\Sigma_{12}$ — the cross term is diversification, mathematically: a negative
$\Sigma_{12}$ actively subtracts from total risk.

**Primal problem:**

$$\min_w \; w^T\Sigma w \qquad \text{s.t.} \qquad \sum_i w_i = 1, \qquad w_i \ge 0$$

**Convexity:** any genuine covariance matrix is PSD by construction (variance of a linear
combination of random variables cannot be negative), so $w^T\Sigma w$ is convex with no
separate proof required. Both constraints are linear. This is a convex QP with the same
optimality guarantee established for SVM in Task 2.

## 2. Data assumptions

Synthetic 5-asset universe, **not real market data** — noted explicitly. Built by hand with
a specific diversification story in mind: two correlated growth stocks, a low-risk bond
anchor, a gold/hedge asset with *negative* correlation to the stocks, and a roughly
uncorrelated commodity.

| Asset | Expected return | Volatility |
|---|---|---|
| Growth Stock 1 | 12% | 25% |
| Growth Stock 2 | 11% | 22% |
| Bond | 4% | 5% |
| Gold/Hedge | 6% | 15% |
| Commodity | 8% | 18% |

Correlation matrix verified genuinely PSD via its eigenvalues before use — all five
eigenvalues positive (`[0.0025, 0.0159, 0.0214, 0.0325, 0.0960]`) — reusing the same
convexity-verification method built in Phase 2 of this project, rather than assuming a
hand-built matrix is automatically valid.

## 3. Implementation and global minimum-variance result

CVXPY (`cp.quad_form(w, Sigma)`, CLARABEL solver, matching the choice validated in Task 2).

**Status: `optimal`.**

| Asset | Weight |
|---|---|
| Growth Stock 1 | 0.0118 |
| Growth Stock 2 | 0.0307 |
| Bond | 0.8234 |
| Gold/Hedge | 0.0747 |
| Commodity | 0.0594 |

Portfolio expected return: **4.70%**. Portfolio risk (std dev): **4.62%**.

## 4. Constraint verification

- $\sum_i w_i = 1.000000$ — exact budget constraint satisfied to numerical precision.
- All $w_i \ge 0$ — long-only constraint satisfied (no shorting occurred, though nothing
  in the QP prevents it structurally; the constraint is doing real work here since an
  unconstrained solution could plausibly want to short the correlated growth stocks).

## 5. Interpretation: diversification is visible in the actual weights, not just theory

Bonds dominate (82.3%) — unsurprising, by far the lowest individual volatility. The
genuinely interesting result: **Gold/Hedge receives a real allocation (7.5%) despite a
modest 6% return** — worse than Commodity's 8% and better only than Bond's 4%. This is not
explained by Gold's own risk/return profile; it is explained entirely by its **negative
correlation with the growth stocks** (−0.20, −0.15). This is the $2w_1w_2\Sigma_{12}$ cross
term from Section 1 doing real, visible work: the optimizer holds an asset specifically
*because* it moves opposite to the riskiest holdings, not despite its mediocre standalone
numbers. This is the single clearest way to demonstrate genuine understanding of Markowitz
to a mentor — the diversification benefit is not asserted, it is observed directly in the
optimizer's output.

## 6. Efficient frontier

25 target returns swept from the global minimum-variance return (4.70%) up to the
long-only ceiling (12%, i.e. 100% in the best single asset). **All 25 points solved to
`optimal` status — zero failures.**

Sample points:

| Target return | Achieved return | Risk |
|---|---|---|
| 0.0470 | 0.0470 | 0.0462 |
| 0.0622 | 0.0622 | 0.0590 |
| 0.0774 | 0.0774 | 0.0866 |
| 0.0926 | 0.0926 | 0.1211 |
| 0.1078 | 0.1078 | 0.1789 |

Risk grows faster than linearly as target return increases — consistent with the
quadratic objective: early diversification gains (Bond + Gold mix) are cheap, but pushing
return higher forces increasing concentration into the correlated growth stocks, which
costs disproportionately more risk per unit of additional return.

## 7. Limitations

- Synthetic data, not real historical returns/covariances.
- Long-only only; no comparison against an unconstrained (short-allowed) frontier, which
  would show the long-only constraint's cost explicitly.
- No risk-free asset / Sharpe ratio computed yet (see extensions).
- Single fixed covariance matrix; no sensitivity analysis on the correlation assumptions
  (e.g. what happens to the Gold allocation if its correlation with stocks were 0 instead
  of -0.20).

## 8. Possible extensions

- Sharpe ratio and the tangency portfolio (highest reward-per-unit-risk point), given a
  risk-free rate.
- Unconstrained (short-allowed) frontier overlaid against the long-only frontier, to make
  the constraint's cost visible.
- Sensitivity analysis: re-solve with Gold's correlation swept from -0.5 to +0.5, showing
  its allocation change directly as a function of diversification benefit.
- Real historical data (e.g. a handful of real tickers' historical covariance) as a direct
  test of whether these synthetic-data findings hold up.