---
title: "Mahalanobis Distance"
date: "2026-09-04"
chapter: 8
chapterTitle: "Unsupervised Learning Techniques"
bookSlug: "hands-on-ml"
tags: ["machine-learning", "unsupervised-learning", "mahalanobis-distance", "covariance", "whitening", "anomaly-detection"]
description: "Mahalanobis distance through covariance geometry and whitening, with an eigendecomposition derivation, special cases, and the Gaussian chi-square connection."
---

**Mahalanobis distance** measures how far a point is from a distribution's center while accounting for feature scales and correlations. A displacement is more unusual along a direction with little variation than along one in which the data naturally spreads out.

This note develops the distance used in [robust covariance and Fast-MCD](/notes/hands-on-ml/chapter-08/unsupervised-learning-techniques/#robust-covariance-and-fast-mcd).

## Definition

For a point $\mathbf{x}\in\mathbb{R}^{D}$, a mean vector $\boldsymbol{\mu}$, and a positive-definite covariance matrix $\Sigma$, the **squared Mahalanobis distance** is:

$$
D_M^2(\mathbf{x})
=(\mathbf{x}-\boldsymbol{\mu})^{\mathsf T}
\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}).
$$

The distance itself is its square root:

$$
D_M(\mathbf{x})
=\sqrt{(\mathbf{x}-\boldsymbol{\mu})^{\mathsf T}
\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}.
$$

Here:

- $D$ is the number of features.
- $\boldsymbol{\mu}$ describes the distribution's center.
- $\Sigma$ describes its spread and correlations.
- $\Sigma^{-1}$ weights displacements according to that covariance structure.

> **Distance versus squared distance:** The quadratic form gives $D_M^2$. Take its square root to obtain $D_M$. The distinction also matters when choosing a statistical threshold.

## Why Euclidean distance can miss unusual points

Euclidean distance weights coordinate directions equally and ignores the distribution's covariance. It remains a valid geometric distance, but equal Euclidean distances need not indicate equally unusual observations.

For correlated features, points may spread along a narrow diagonal band. A point along that band can be quite typical, while another point at the same Euclidean distance from the center but away from the band can be unusual.

![Uncorrelated and correlated observations, comparing two points at equal Euclidean distances from the center.](/images/notes/hands-on-ml/mahalanobis-distance/correlated-features.png)

*In the correlated example, the pink point departs from the main direction of variation more strongly than the purple point.*

Euclidean distance has spherical constant-distance contours. Mahalanobis distance uses ellipsoidal contours aligned with the covariance structure.

![Circular Euclidean distance contours compared with elongated Mahalanobis contours following correlated observations.](/images/notes/hands-on-ml/mahalanobis-distance/distance-contours.png)

## Geometric intuition: whitening

Mahalanobis distance is Euclidean distance after **whitening** the data:

1. **Center:** Subtract the mean so that the reference center becomes the origin.
2. **Decorrelate:** Express the centered vector along the covariance matrix's eigenvectors, which are the principal directions of variation.
3. **Rescale:** Divide each projected coordinate by the standard deviation along that direction.
4. **Measure:** Calculate Euclidean distance from the origin in the transformed space.

Whitening gives uncorrelated coordinates with unit variance. Standardizing the original features individually adjusts their marginal scales but generally leaves correlations between them.

## Deriving the whitening transformation

### Eigendecompose the covariance matrix

A symmetric, positive-definite covariance matrix has the decomposition:

$$
\Sigma=V\Lambda V^{\mathsf T},
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_D),
$$

where the columns $\mathbf{v}_1,\ldots,\mathbf{v}_D$ of $V$ are orthonormal eigenvectors, $V^{\mathsf T}V=I_D$, and every eigenvalue $\lambda_j$ is positive.

Therefore:

$$
\begin{aligned}
\Sigma^{-1}
&=V\Lambda^{-1}V^{\mathsf T}\\
&=V\Lambda^{-1/2}\Lambda^{-1/2}V^{\mathsf T}\\
&=\left(\Lambda^{-1/2}V^{\mathsf T}\right)^{\mathsf T}
\left(\Lambda^{-1/2}V^{\mathsf T}\right).
\end{aligned}
$$

Define the whitening matrix:

$$
W=\Lambda^{-1/2}V^{\mathsf T}.
$$

Then $\Sigma^{-1}=W^{\mathsf T}W$.

### Center, project, and rescale

First center the observation:

$$
\boldsymbol{\delta}=\mathbf{x}-\boldsymbol{\mu}.
$$

Project onto the eigenvectors:

$$
\mathbf{y}=V^{\mathsf T}\boldsymbol{\delta}
=\begin{bmatrix}
\mathbf{v}_1^{\mathsf T}(\mathbf{x}-\boldsymbol{\mu})\\
\vdots\\
\mathbf{v}_D^{\mathsf T}(\mathbf{x}-\boldsymbol{\mu})
\end{bmatrix}.
$$

The variance along eigenvector $j$ is $\lambda_j$, so its standard deviation is $\sqrt{\lambda_j}$. Rescale each coordinate:

$$
\mathbf{z}=\Lambda^{-1/2}\mathbf{y}
=W(\mathbf{x}-\boldsymbol{\mu}),
\qquad
z_j=\frac{y_j}{\sqrt{\lambda_j}}.
$$

For a random vector with covariance $\Sigma$, the transformed covariance is:

$$
\operatorname{Cov}(\mathbf{z})
=W\Sigma W^{\mathsf T}=I_D.
$$

### Apply Euclidean distance

Substituting the factorization into the original quadratic form gives:

$$
\begin{aligned}
D_M^2(\mathbf{x})
&=\boldsymbol{\delta}^{\mathsf T}\Sigma^{-1}\boldsymbol{\delta}\\
&=\boldsymbol{\delta}^{\mathsf T}W^{\mathsf T}W\boldsymbol{\delta}\\
&=\mathbf{z}^{\mathsf T}\mathbf{z}\\
&=\sum_{j=1}^{D}z_j^2
=\sum_{j=1}^{D}\frac{y_j^2}{\lambda_j}.
\end{aligned}
$$

Thus, $D_M(\mathbf{x})=\lVert\mathbf{z}\rVert_2$. Directions with large variance receive less weight; directions with small variance receive more.

> **The inverse requires care:** Covariance matrices are always positive semidefinite, but they can be singular. This derivation requires strictly positive eigenvalues. Redundant features or too few observations may require a reduced representation or an appropriately regularized covariance estimate before using the inverse formula.

## Special cases and distributional properties

### Uncorrelated features

If the covariance is diagonal:

$$
\Sigma=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_D^2),
$$

the distance simplifies to:

$$
D_M(\mathbf{x})
=\sqrt{\sum_{j=1}^{D}
\left(\frac{x_j-\mu_j}{\sigma_j}\right)^2}.
$$

This is Euclidean distance after standardizing each feature by its own standard deviation. A diagonal covariance means the features are uncorrelated; independence requires additional assumptions.

### Identity covariance

When $\Sigma=I_D$, all directions have unit variance and zero covariance, so:

$$
D_M(\mathbf{x})=\lVert\mathbf{x}-\boldsymbol{\mu}\rVert_2.
$$

More generally, $\Sigma=\sigma^2I_D$ gives Euclidean distance divided by the common standard deviation $\sigma$.

### Gaussian data and the chi-square distribution

Suppose the point comes from the multivariate Gaussian distribution:

$$
\mathbf{x}\sim\mathcal{N}(\boldsymbol{\mu},\Sigma).
$$

Using the true mean and positive-definite covariance, whitening gives:

$$
\mathbf{z}\sim\mathcal{N}(\mathbf{0},I_D).
$$

Its coordinates are independent standard normal variables. Their squared sum therefore follows a chi-square distribution with $D$ degrees of freedom:

$$
D_M^2(\mathbf{x})
=\sum_{j=1}^{D}z_j^2
\sim\chi_D^2.
$$

This is the defining sum-of-squares relationship described in the [NIST chi-square reference](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm).

![Original chi-square curves for several degrees of freedom, retaining the handwritten sum-of-squared-standard-normal-variables annotation.](/images/notes/hands-on-ml/mahalanobis-distance/chi-square-distributions.png)

*These are probability density curves. The degrees of freedom equal the number of independent standard normal coordinates being squared and summed.*

## Connection to anomaly detection

Under the Gaussian model with known parameters, a reference rule at tail level $\alpha$ is:

$$
D_M^2(\mathbf{x})>\chi^2_{D,1-\alpha},
$$

where $\chi^2_{D,1-\alpha}$ is the corresponding chi-square quantile. This flags points far from the center relative to the modeled covariance geometry.

In practice, the mean and covariance are estimated, sometimes from data that already contains outliers. This changes the statistical behavior of the distances, so the exact chi-square result should not be assumed for every fitted model. Robust covariance estimation can help prevent outliers from distorting the reference geometry; see the [Chapter 08 overview](/notes/hands-on-ml/chapter-08/unsupervised-learning-techniques/#robust-covariance-and-fast-mcd).

For implementation details, [SciPy's `mahalanobis` reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html) defines the distance and its inverse-covariance argument. The [scikit-learn robust covariance example](https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html) illustrates how covariance estimation affects outlier separation.
