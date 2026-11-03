Models
======

Every model states its likelihood, fits it with an objective that provably improves at each
iteration, handles missing entries and reports a held-out log-likelihood. Each model gets its
own page here when it is released.

.. list-table::
   :header-rows: 1
   :widths: 18 30 22 22 8

   * - Class
     - Model
     - Inference
     - Source papers
     - Release
   * - ``PoissonNMF``
     - X ~ Poisson(WH)
     - MM (KL multiplicative updates)
     - Lee & Seung 2001; Févotte & Idier 2011
     - v0.1
   * - ``NegBinNMF``
     - X ~ NB(WH, α); α shared, per sample or per feature
     - MM for W, H; Newton–Raphson for α
     - Gouvert, Oberlin & Févotte 2020; Pelizzola, Laursen & Hobolth 2023
     - v0.1
   * - ``NBMF``, ``BooleanMF``, ``BayesianBooleanMF``
     - Re-exported from nbmf-mm and boolmf
     - As implemented there
     - Magron & Févotte 2022; see boolmf
     - v0.1
   * - ``ARDNMF``
     - β-divergence likelihood with ARD priors
     - MAP by MM
     - Tan & Févotte 2013
     - v0.2
   * - ``BayesianPoissonNMF``
     - Poisson with Gamma priors
     - Variational Bayes (CAVI)
     - Cemgil 2009
     - v0.2
   * - ``TweedieNMF``
     - Tweedie(WH, p, φ), 1 ≤ p ≤ 2
     - MM via β = 2 − p; profile likelihood
     - Févotte & Idier 2011; Şimşekli, Cemgil & Yılmaz 2013; James, Hobolth & Pelizzola 2026
     - v0.2
   * - ``BayesianNegBinNMF``
     - NB as Gamma–Poisson with Gamma priors
     - Variational Bayes (CAVI)
     - Gouvert, Oberlin & Févotte 2018
     - v0.3
   * - ``BayesianNBMF``
     - Mean-parameterized Bernoulli with Beta and Dirichlet priors
     - Gibbs; variational option
     - Lumbreras, Filstroff & Févotte 2020
     - v0.3
   * - ``SemiNMF``
     - Gaussian; W ≥ 0, H signed
     - MM
     - Ding, Li & Jordan 2010
     - v0.3
   * - ``ConvexNMF``
     - H = GX, G ≥ 0; Gaussian, NB or Tweedie cost
     - MM
     - Ding, Li & Jordan 2010; James, Hobolth & Pelizzola 2026
     - v0.3
   * - ``GenPoissonNMF``
     - Generalized Poisson, per-sample dispersion
     - MM; closed-form θ
     - Ohashi, Abe & Sakaori 2025
     - v0.4
   * - ``OrdinalNMF``
     - Ordinal levels as thresholded latent counts
     - Variational inference by augmentation
     - Gouvert, Oberlin & Févotte 2020 (ICML)
     - v0.4
   * - ``GraphNMF``
     - Graph-Laplacian penalty on sample loadings
     - MM
     - Cai, He, Han & Huang 2011
     - v0.4
