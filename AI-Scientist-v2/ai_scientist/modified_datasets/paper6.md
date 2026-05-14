# Title: Conformal Prediction as Bayesian Quadrature

## Keywords
Conformal Prediction (CP), Bayesian Quadrature (BQ), Uncertainty Quantification, Bayesian Statistics, Distribution-Free Methods, Conditional Guarantees (or Data-Conditional Guarantee), Conformal Risk Control (CRC), Posterior Risk, Quantile Spacings (or Dirichlet Distribution), Probabilistic Numerics

## TL;DR

Current conformal prediction (CP) methods for uncertainty quantification rely on frequentist statistics, which makes it difficult to incorporate prior knowledge and only provides a marginal (average) guarantee on the expected loss, often leading to unacceptably high risk in individual deployment scenarios.

This paper proposes a new framework that reinterprets CP from a Bayesian perspective using Bayesian Quadrature (BQ). By modeling the uncertainty in the loss distribution's quantile function, the approach provides a more complete view—a full posterior distribution over the expected loss, which yields interpretable, data-conditional guarantees. This method is non-parametric, recovers existing CP techniques (like Split CP and Conformal Risk Control) as special cases (the posterior mean) , and experimentally achieves significantly lower failure rates (better conditional guarantees) and smaller prediction sets compared to traditional approaches and baselines.

## Abstract

As machine learning-based prediction systems are
increasingly used in high-stakes situations, it is
important to understand how such predictive mod-
els will perform upon deployment. Distribution-
free uncertainty quantification techniques such as
conformal prediction provide guarantees about
the loss black-box models will incur even when
the details of the models are hidden. However,
such methods are based on frequentist probability,
which unduly limits their applicability. We revisit
the central aspects of conformal prediction from
a Bayesian perspective and thereby illuminate the
shortcomings of frequentist guarantees. We pro-
pose a practical alternative based on Bayesian
quadrature that provides interpretable guarantees
and offers a richer representation of the likely
range of losses to be observed at test time.
