"""
dataset.py

Generates the synthetic 2D binary classification dataset used throughout Task 2.

Design choice: two isotropic Gaussian blobs with moderate overlap, so that
soft-margin behavior (slack variables) is genuinely exercised -- a fully
separable dataset would make the C hyperparameter sweep uninteresting, since
hard-margin and soft-margin SVM would behave almost identically at any C.

Chosen via visual inspection: centers=[-1.8,0]/[1.8,0], std=1.3 gives clean
bulk separation with a real, non-trivial mixed zone near the boundary.
"""

import numpy as np
from sklearn.datasets import make_blobs

CENTERS = [[-1.8, 0], [1.8, 0]]
CLUSTER_STD = 1.3


def generate_dataset(n_samples=150, centers=CENTERS, cluster_std=CLUSTER_STD, random_state=42):
    """
    Generate a 2D binary classification dataset with controlled overlap.

    Parameters
    ----------
    n_samples : int
        Total number of points to generate.
    centers : list of [x, y]
        The two cluster centers.
    cluster_std : float
        Standard deviation of each Gaussian blob. Larger values produce more
        overlap between classes (more support vectors, more interesting C sweep).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Class labels in {-1, +1} (SVM convention -- see derivation notes).
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )

    # make_blobs returns labels in {0, 1}; the SVM formulation we derived relies
    # on labels in {-1, +1} so that y_i * (w^T x_i + b) correctly encodes
    # "correctly classified" via its sign. Remap here, once, at the source.
    y = np.where(y == 0, -1, 1)

    return X, y


if __name__ == "__main__":
    X, y = generate_dataset()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class counts: +1 -> {(y == 1).sum()}, -1 -> {(y == -1).sum()}")
    print(f"Feature ranges: x1 in [{X[:,0].min():.2f}, {X[:,0].max():.2f}], "
          f"x2 in [{X[:,1].min():.2f}, {X[:,1].max():.2f}]")