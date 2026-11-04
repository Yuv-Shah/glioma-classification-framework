import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

RANDOM_STATE = 42


def writable_array(values):
    """Give scikit-learn an independent writable numeric array."""
    return np.array(values, dtype=float, copy=True)


def make_mice_imputer():
    """Return a fold-fit chained-equation-style numeric imputer.

    With complete data, IterativeImputer returns values unchanged. When a fold
    contains missing data, it estimates values iteratively from other training-fold
    features. This creates one completed dataset; it is not pooled multiple
    imputation inference.
    """
    return IterativeImputer(
        max_iter=5,
        initial_strategy="median",
        n_nearest_features=20,
        skip_complete=True,
        keep_empty_features=True,
        random_state=RANDOM_STATE,
    )
