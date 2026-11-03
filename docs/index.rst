probNMF
=======

Probabilistic and Bayesian NMF: Poisson, negative binomial, Tweedie and ordinal NMF with estimated dispersion, missing data, held-out likelihood and automatic rank selection, with a scikit-learn API.

.. note::

   Pre-alpha. The shared engine, the API contract and the test harness are in place; the
   first models land in v0.1.

Existing NMF packages offer losses and solvers; probNMF offers likelihoods. Each model
states its observation model, estimates dispersion where it has one, takes missing-data
masks, reports a held-out log-likelihood and, in its Bayesian or ARD form, chooses the
rank. Frobenius and β-divergence NMF without a likelihood stay in scikit-learn.

The fitted solution is a nonnegative cone (the rows of H generate a cone in the positive orthant).

.. toctree::
   :maxdepth: 2

   install
   models
   api_contract
   correctness
   auto_examples/index
   api
