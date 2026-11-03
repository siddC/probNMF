"""probnmf: probabilistic and Bayesian NMF.

Nonnegative matrix factorizations as statistical models for nonnegative data that
is not Gaussian: overdispersed counts, compound Poisson data, ordinal ratings, and
their Bayesian versions with automatic rank selection.

Every estimator shares one API: ``X`` is (n_samples, n_features), missing entries are NaN or
``mask=False``, ``score`` is the mean log-likelihood per observed entry, and ``sample`` draws
from the fitted model. See https://probnmf.readthedocs.io for the model pages.
"""

from . import datasets, metrics
from ._base import BaseFactorization

__version__ = "0.1.0.dev0"

__all__ = ["BaseFactorization", "datasets", "metrics", "__version__"]
