"""
data.py

Defines the (synthetic, but realistically structured) 5-asset universe used
throughout Task 3: expected returns, volatilities, and a correlation
structure designed to demonstrate genuine diversification -- not an
arbitrary random matrix.

Asset roles:
    Growth Stock 1  -- high risk/return
    Growth Stock 2  -- correlated with Stock 1 (same sector, corr=0.70)
    Bond            -- low-risk anchor, low correlation with stocks
    Gold/Hedge      -- slightly NEGATIVELY correlated with stocks (the diversifier)
    Commodity       -- roughly uncorrelated with everything

This is synthetic data, not real market data -- noted explicitly in the
report. A real historical covariance matrix is listed as a direct extension.
"""

import numpy as np

ASSET_NAMES = ["Growth Stock 1", "Growth Stock 2", "Bond", "Gold/Hedge", "Commodity"]

EXPECTED_RETURNS = np.array([0.12, 0.11, 0.04, 0.06, 0.08])
VOLATILITIES     = np.array([0.25, 0.22, 0.05, 0.15, 0.18])

# Correlation matrix, built by hand to reflect the asset roles above.
CORRELATION = np.array([
    [ 1.00,  0.70,  0.05, -0.20,  0.10],
    [ 0.70,  1.00,  0.05, -0.15,  0.05],
    [ 0.05,  0.05,  1.00,  0.10,  0.00],
    [-0.20, -0.15,  0.10,  1.00,  0.05],
    [ 0.10,  0.05,  0.00,  0.05,  1.00],
])


def get_covariance_matrix():
    """
    Convert the correlation matrix + volatilities into a covariance matrix:
        Sigma_ij = corr_ij * vol_i * vol_j
    and verify it is genuinely positive semidefinite via its eigenvalues --
    reusing the exact convexity check built earlier in this project, rather
    than assuming a hand-built matrix is automatically valid.

    Returns
    -------
    Sigma : ndarray (5, 5)
    eigenvalues : ndarray (5,)
    is_psd : bool
    """
    D = np.diag(VOLATILITIES)
    Sigma = D @ CORRELATION @ D

    eigenvalues = np.linalg.eigvalsh(Sigma)
    is_psd = bool(np.all(eigenvalues >= -1e-10))

    return Sigma, eigenvalues, is_psd


if __name__ == "__main__":
    Sigma, eigenvalues, is_psd = get_covariance_matrix()
    print("Assets:", ASSET_NAMES)
    print("Expected returns:", EXPECTED_RETURNS)
    print("Volatilities:", VOLATILITIES)
    print("\nCovariance matrix:")
    print(np.round(Sigma, 4))
    print("\nEigenvalues:", np.round(eigenvalues, 6))
    print(f"PSD (all eigenvalues >= 0)? {is_psd}")