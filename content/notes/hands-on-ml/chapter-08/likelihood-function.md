---
title: "Likelihood Function"
date: "2026-09-03"
chapter: 8
chapterTitle: "Unsupervised Learning Techniques"
bookSlug: "hands-on-ml"
tags: ["machine-learning", "unsupervised-learning", "likelihood", "maximum-likelihood", "bayesian-inference", "regularization"]
description: "Probability versus likelihood, maximum likelihood and MAP estimation, log-likelihood, and how Gaussian and Laplace priors lead to L2 and L1 regularization."
---

A **likelihood function** describes how a model supports different parameter values after the data has been observed. It uses the same mathematical expression as the sampling model, but holds the observations fixed and varies the parameters.

These ideas underpin [Gaussian mixture fitting and expectation-maximization](/notes/hands-on-ml/chapter-08/unsupervised-learning-techniques/#expectation-maximization) in Chapter 08.

## Probability and likelihood

Let $p(x\mid\theta)$ denote a model's probability mass function or probability density function, where $x$ is an observation and $\theta$ is a parameter or parameter vector.

- **Probability modeling:** Fix $\theta$ and consider the possible outcomes $x$.
- **Likelihood:** Fix the observed $x$ and compare possible parameter values $\theta$.

The likelihood is defined by:

$$
\mathcal{L}(\theta\mid x)=p(x\mid\theta).
$$

The numerical expression is the same; the argument being varied changes. For continuous data, $p(x\mid\theta)$ is a density at $x$, rather than the probability of observing that exact value.

### Viewing the same model along different axes

Imagine a surface with axes $x$ and $\theta$:

- A slice at a fixed $\theta$ gives the sampling distribution over $x$.
- A slice at a fixed observed $x$ gives a likelihood curve over $\theta$.

![Original four-panel figure showing a model over observations and parameters, a probability density at a fixed parameter, and likelihood and log-likelihood curves for a fixed observation.](/images/notes/hands-on-ml/likelihood-function/probability-likelihood.png)

*The figure fixes $\theta=1.3$ to show a density over observations, and fixes $x=2.5$ to show likelihood and log-likelihood over the parameter. The maxima of the latter two curves occur at the same parameter value.*

For a continuous sampling density, normalization is over the observations:

$$
\int p(x\mid\theta)\,dx=1.
$$

For discrete data, the corresponding sum equals one. There is no requirement that integrating the likelihood over $\theta$ gives one; that integral may not even be finite.

> **Likelihood is not a posterior:** $\mathcal{L}(\theta\mid x)$ is a function of the parameter, but it does not by itself assign probabilities to parameter values. A posterior distribution also requires a prior and normalization.

## Maximum likelihood estimation

For an observed dataset $X$, **maximum likelihood estimation (MLE)** chooses parameter values maximizing its joint probability mass or density:

$$
\hat{\theta}_{\mathrm{MLE}}
=\operatorname*{argmax}_{\theta}p(X\mid\theta)
=\operatorname*{argmax}_{\theta}\mathcal{L}(\theta\mid X).
$$

MLE asks which parameter values make the observed data most compatible with the assumed model. It does not incorporate a prior distribution over those parameters.

## Why use log-likelihood?

The natural logarithm is strictly increasing, so maximizing a positive likelihood is equivalent to maximizing its logarithm:

$$
\hat{\theta}_{\mathrm{MLE}}
=\operatorname*{argmax}_{\theta}\ln\mathcal{L}(\theta\mid X).
$$

If $X=(x_1,\ldots,x_m)$ consists of observations that are independent given $\theta$ and share the same sampling model, then:

$$
\mathcal{L}(\theta\mid X)
=\prod_{i=1}^{m}p(x_i\mid\theta),
$$

and the log-likelihood becomes:

$$
\ell(\theta)
=\ln\mathcal{L}(\theta\mid X)
=\sum_{i=1}^{m}\ln p(x_i\mid\theta).
$$

This turns products into sums, simplifies differentiation, and helps avoid numerical underflow from multiplying many small values. Equivalently, an optimizer can minimize the **negative log-likelihood**, $-\ell(\theta)$.

## Maximum a posteriori estimation

When prior information about the parameters is available, use a prior distribution $p(\theta)$. Bayes' rule gives the posterior:

$$
p(\theta\mid X)
=\frac{p(X\mid\theta)p(\theta)}{p(X)}.
$$

Here:

- $p(\theta)$ is the **prior**, expressing information before observing $X$.
- $p(X\mid\theta)$ is the **likelihood**.
- $p(\theta\mid X)$ is the **posterior**, incorporating the observations.
- $p(X)$ is the **evidence**, which normalizes the posterior.

For a fixed dataset, $p(X)$ does not depend on $\theta$. Therefore, when finding the posterior maximum:

$$
p(\theta\mid X)\propto p(X\mid\theta)p(\theta).
$$

**Maximum a posteriori (MAP) estimation** chooses the posterior mode:

$$
\begin{aligned}
\hat{\theta}_{\mathrm{MAP}}
&=\operatorname*{argmax}_{\theta}p(\theta\mid X)\\
&=\operatorname*{argmax}_{\theta}
\mathcal{L}(\theta\mid X)p(\theta)\\
&=\operatorname*{argmax}_{\theta}
\left[\ell(\theta)+\ln p(\theta)\right].
\end{aligned}
$$

MLE optimizes the likelihood; MAP combines it with the prior. A prior constant over the relevant parameter domain leaves the same optimization objective as MLE, up to an additive constant in log space.

## Priors as regularization

Writing MAP as a minimization makes the regularization connection explicit:

$$
\hat{\theta}_{\mathrm{MAP}}
=\operatorname*{argmin}_{\theta}
\left[-\ell(\theta)-\ln p(\theta)\right].
$$

The negative log-prior acts as a penalty. Its shape determines which parameter values are discouraged.

![Original comparison of a sharply peaked Laplace density and a smoother Gaussian density, illustrating two prior shapes.](/images/notes/hands-on-ml/likelihood-function/gaussian-laplace-priors.png)

### Gaussian prior and L2 regularization

For weights $\mathbf{w}$ with an isotropic zero-mean Gaussian prior and fixed scale $\tau>0$:

$$
p(\mathbf{w})\propto
\exp\!\left(-\frac{\lVert\mathbf{w}\rVert_2^2}{2\tau^2}\right).
$$

Taking the negative logarithm gives:

$$
-\ln p(\mathbf{w})
=\frac{1}{2\tau^2}\lVert\mathbf{w}\rVert_2^2+C,
$$

where $C$ does not depend on $\mathbf{w}$. MAP therefore adds an **L2 penalty** to the negative log-likelihood.

### Laplace prior and L1 regularization

Independent zero-centered Laplace priors with a fixed common scale $b>0$ give:

$$
p(\mathbf{w})\propto
\exp\!\left(-\frac{\lVert\mathbf{w}\rVert_1}{b}\right),
$$

so:

$$
-\ln p(\mathbf{w})
=\frac{1}{b}\lVert\mathbf{w}\rVert_1+C.
$$

This produces an **L1 penalty**. In linear regression with Gaussian observation noise, these Gaussian and Laplace prior choices give the ridge and lasso connections, respectively. The [Stanford CS229 exercise on MAP and regularization](https://cs229.stanford.edu/summer2020/ps2.pdf) develops both equivalences.

The effective penalty coefficient depends on the prior scale and how the likelihood loss is normalized. Smaller prior scales express a stronger preference for weights near zero.
