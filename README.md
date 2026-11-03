# probNMF

[![CI](https://github.com/siddC/probNMF/actions/workflows/ci.yml/badge.svg)](https://github.com/siddC/probNMF/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/probnmf/badge/?version=latest)](https://probnmf.readthedocs.io)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)
![Python 3.12–3.14](https://img.shields.io/badge/python-3.12%E2%80%933.14-blue.svg)

Probabilistic and Bayesian NMF: Poisson, negative binomial, Tweedie and ordinal NMF with estimated dispersion, missing data, held-out likelihood and automatic rank selection, with a scikit-learn API.

> **Status: pre-alpha.** The shared engine, the API contract and the test harness for the
> correctness standard are in place. The first models land in v0.1; see the
> [roadmap](ROADMAP.md).

## Why probNMF

Existing NMF packages offer losses and solvers; probNMF offers likelihoods. Each model
states its observation model, estimates dispersion where it has one, takes missing-data
masks, reports a held-out log-likelihood and, in its Bayesian or ARD form, chooses the
rank. Frobenius and β-divergence NMF without a likelihood stay in scikit-learn.

The fitted solution is a nonnegative cone (the rows of H generate a cone in the positive orthant).

## Models

| Class | Model | Inference | Source papers | Release |
| --- | --- | --- | --- | --- |
| `PoissonNMF` | X ~ Poisson(WH) | MM (KL multiplicative updates) | Lee & Seung 2001; Févotte & Idier 2011 | v0.1 |
| `NegBinNMF` | X ~ NB(WH, α); α shared, per sample or per feature | MM for W, H; Newton–Raphson for α | Gouvert, Oberlin & Févotte 2020; Pelizzola, Laursen & Hobolth 2023 | v0.1 |
| `NBMF`, `BooleanMF`, `BayesianBooleanMF` | Re-exported from nbmf-mm and boolmf | As implemented there | Magron & Févotte 2022; see boolmf | v0.1 |
| `ARDNMF` | β-divergence likelihood with ARD priors | MAP by MM | Tan & Févotte 2013 | v0.2 |
| `BayesianPoissonNMF` | Poisson with Gamma priors | Variational Bayes (CAVI) | Cemgil 2009 | v0.2 |
| `TweedieNMF` | Tweedie(WH, p, φ), 1 ≤ p ≤ 2 | MM via β = 2 − p; profile likelihood | Févotte & Idier 2011; Şimşekli, Cemgil & Yılmaz 2013; James, Hobolth & Pelizzola 2026 | v0.2 |
| `BayesianNegBinNMF` | NB as Gamma–Poisson with Gamma priors | Variational Bayes (CAVI) | Gouvert, Oberlin & Févotte 2018 | v0.3 |
| `BayesianNBMF` | Mean-parameterized Bernoulli with Beta and Dirichlet priors | Gibbs; variational option | Lumbreras, Filstroff & Févotte 2020 | v0.3 |
| `SemiNMF` | Gaussian; W ≥ 0, H signed | MM | Ding, Li & Jordan 2010 | v0.3 |
| `ConvexNMF` | H = GX, G ≥ 0; Gaussian, NB or Tweedie cost | MM | Ding, Li & Jordan 2010; James, Hobolth & Pelizzola 2026 | v0.3 |
| `GenPoissonNMF` | Generalized Poisson, per-sample dispersion | MM; closed-form θ | Ohashi, Abe & Sakaori 2025 | v0.4 |
| `OrdinalNMF` | Ordinal levels as thresholded latent counts | Variational inference by augmentation | Gouvert, Oberlin & Févotte 2020 (ICML) | v0.4 |
| `GraphNMF` | Graph-Laplacian penalty on sample loadings | MM | Cai, He, Han & Huang 2011 | v0.4 |

> **Blocked upstream:** the nbmf-mm and boolmf re-exports wait on two releases.
> nbmf-mm 0.1.4 requires Python below 3.13 (probNMF supports 3.12 to 3.14), and boolmf
> is not on PyPI yet. Both become hard dependencies once those releases exist.

Full citations are on each model's documentation page.

## One API for every model

```python
model = SomeModel(n_components=8, random_state=0)
Z = model.fit_transform(X, mask=observed)   # X: (n_samples, n_features); Z: (n_samples, k)
model.components_                           # (n_components_, n_features)
model.score(X_test, mask=heldout)           # mean held-out log-likelihood per observed entry
model.sample(100, random_state=1)           # synthetic data from the fitted model
```

| Member | Contract |
| --- | --- |
| `fit(X, y=None, mask=None)` | Fit the model; returns self |
| `transform(X, mask=None)` | Latent values for new samples, components held fixed |
| `fit_transform(X, mask=None)` | Fit, then return the latent values of the training samples |
| `inverse_transform(Z)` | Mean of the observation model on the data scale |
| `score(X, mask=None)` | Mean log-likelihood per observed entry (posterior predictive for Bayesian models) |
| `score_samples(X, mask=None)` | Log-likelihood per sample |
| `sample(n_samples, random_state=None)` | Synthetic data from the fitted model |
| `components_`, `n_components_` | Fitted components; effective number after any ARD pruning |
| `objective_`, `n_iter_`, `converged_` | Minimized objective per iteration, iterations used, convergence flag |
| `loglik_` | Final training log-likelihood |

Missing entries are NaN in `X` or `False` in `mask`; every likelihood, update and score sums
over observed entries only. float64 by default; float32 input stays float32.

## Correctness

No estimator ships without passing six layers of tests:

1. **Monotone objective**: MM and EM objectives never increase, ELBOs never decrease.
2. **Limiting cases**: each model reduces to its documented special case.
3. **Reproduction**: each source paper's experiments are rerun and must match the published
   numbers to the precision the paper reports.
4. **Recovery**: factors and rank are recovered from data simulated by the model, over 10 seeds.
5. **Stationarity**: KKT conditions hold at convergence.
6. **Conformance**: scikit-learn's estimator checks, pickling, float32 and float64.

Models are implemented from their papers only (no author code is read or translated).
Maintained public packages may be run as black-box cross-checks on shared inputs.

## Install

probNMF is not on PyPI yet. Until the first release:

```bash
pip install "git+https://github.com/siddC/probNMF"
```

Optional extras: `[jax]` for the JAX backend and `[mcmc]` (NumPyro) for the models that
sample. Python 3.12 to 3.14.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Releases follow [RELEASING.md](RELEASING.md).

## Citation

If you use probNMF, please cite it; [CITATION.cff](CITATION.cff) has the details. A JOSS
paper will follow v1.0.

## License

BSD 3-Clause; see [LICENSE](LICENSE).
