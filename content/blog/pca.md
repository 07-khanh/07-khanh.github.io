---
title: "Principal Component Analysis - PCA"
date: "2026-08-26"
description: "How principal component analysis finds directions of maximum variance, and how projections, eigenvectors, and SVD connect."
---
The core idea is surprisingly simple:

> PCA finds a new set of orthogonal directions (axes) that capture as much variation in the data as possible.

This is because high variance usually means that direction contains a lot of information about how the observation differ.

Then we can represent the data using those new axes instead of the original ones, or we can also keep the most important directions and throw away the less important ones.

For example, suppose we have this data in 2D space. 

![Selecting the subspace on which to project](/images/pca-variance.jpg)

Notice that the points don't really fill the entire 2D space. They mostly lie along **one direction**. So although we have two features, data essentially has **one important dimension**.

Then, instead of describing every point using $x_1, x_2$, we can describe it approximately using: position along this diagonal direction (i.e, we project the data onto a new axis, which preserves the most variance.)

Consider your shadow on the ground when the sun is directly overhead: it’s a small blob that doesn’t look anything like you. But your shadow on a wall at sunrise is much larger and it does look like you. Another way to justify choosing the axis that maximizes the variance is that it is also the axis that minimizes the mean squared distance between the original dataset and its projection onto that axis.

---
### Principal Components

PCA identifies the set of orthogonal axes that account for the largest amount of variance in the training set, so that it can later projects data on. The $i$th axis is called the $i$th principal component (PC) of the data.

![Principal Components PC1 and PC2](/images/principle-components.jpg)

In other words, suppose your centered data points are

$$
\mathbf x_1,\mathbf x_2,\ldots,\mathbf x_n.
$$

You have to find a unit vector $\mathbf w$ so that when you project every data point onto this direction:

$$
z_i=\mathbf x_i^T\mathbf w.
$$

or written in matrix form:

$$
z = X_c \mathbf w
$$

the projected values $z_i$ have the largest variance.

The variance of the projected data is:

$$
\text{Var}(z) = \frac{1}{n} \sum_{i=1}^n (z_i - \bar{z})^2 = \frac{1}{n} \sum_{i=1}^n (x_i^T w)^2
$$

So mathematically:

$$
\boxed{ \mathbf w_1 = \arg\max_{\|\mathbf w\|=1} \operatorname{Var}(X_c\mathbf w) }
$$

where $X$ is the centered data matrix.

That $\mathbf w_1$ becomes the first principal component direction.

---

### Covariance matrix

Let's rewrite the variance formula above in matrix form:

$$
\text{Var}(z) = \frac{1}{n} \sum_{i=1}^n (w^T x_i)(x_i^T w) = w^T \left( \frac{1}{n} \sum_{i=1}^n x_i x_i^T \right) w = w^T \Sigma w
$$

where $\Sigma = \frac{1}{n} \sum_{i=1}^n x_i x_i^T = \frac{1}{n} X_c^T X_c$ is the sample covariance matrix of the centered data matrix $X_c \in \mathbb{R}^{n \times d}$.

So PCA becomes:

$$
\max_{w} \quad w^T \Sigma w
$$

$$
\text{subject to} \quad w^T w = 1
$$

To remind you about the covariance matrix, suppose:

$$
X_c= \begin{bmatrix} x_{11}&x_{12}\\ x_{21}&x_{22}\\ x_{31}&x_{32} \end{bmatrix}
$$

Then:

$$
X_c^TX_c = \begin{bmatrix} x_{11}^2+x_{21}^2+x_{31}^2& x_{11}x_{12}+x_{21}x_{22}+x_{31}x_{32}\\ x_{11}x_{12}+x_{21}x_{22}+x_{31}x_{32}& x_{12}^2+x_{22}^2+x_{32}^2 \end{bmatrix}
$$

Then as the data is centered, after dividing this by $n$, the diagonal entries becomes the variance of features, while the off-diagonal becomes the covariance between features.

---

### Solving the optimization problem

We want to solve

$$
\max_{w} \quad w^T \Sigma w
$$

$$
\text{subject to} \quad w^T w = 1
$$

This is a constrained optimization problem and we can solve this using Lagrange multipliers method.

We cosntruct:

$$
\mathcal{L}(w, \lambda) = w^T \Sigma w - \lambda (w^T w - 1)
$$

Then take the gradient with respect to $w$ and set it to zero:

$$
\nabla_w \mathcal{L}(w, \lambda) = 2\Sigma w - 2\lambda w = 0

\Rightarrow \Sigma w = \lambda w
$$

And this is exactly the **eigenvector equation**. So it turns out, the directions that maximize projected variance are eigenvectors of the covariance matrix.

When you multiply both sides by $w^T$:

$$
w^T \Sigma w = w^T (\lambda w) = \lambda (w^T w) = \lambda
$$

But earlier we established $ w^T\Sigma w $ is the variance of the projected data. Therefore, $\lambda$ is the variance captured by that principal direction!

Thus, to maximize the variance $w^T \Sigma w$, you choose the eigenvector $w_1$ corresponding to the largest eigenvalue $\lambda_{\max}$ of the covariance matrix $\Sigma$. Subsequent principal components are found by solving the same problem with additional orthogonality constraints: $w_k^T w_j = 0$ for all $j < k$.

In practice, PCA is very often calculated using something called Singular Value Decomposition, or SVD, rather than explicitly calculating the covariance matrix and then its eigenvectors.

SVD decomposes the centered data matrix as:

$$
\boxed{ X_c=U\Sigma V^T }
$$

Don't confuse this $\Sigma$ with the covariance matrix—we're using the same symbol here traditionally, which is unfortunate.

Let's call the SVD diagonal matrix $S$ instead:

$$
X_c=USV^T
$$

Then:

$U$ = left singular vectors
$S$ = singular values
$V$ = right singular vectors

For PCA: The rows/columns of $V$ give the principal directions. More precisely, the columns of $V$ are the eigenvectors of $X_c^TX_c$.

---

### Algorithm

① First, we usually center the data by subtracting the mean of each feature. This will move the data so that its mean is at the origin. This is important because PCA cares about variation around the center, not where the entire dataset happens to be located.

② Apply SVD directly to the centered data:

$$
X_c=USV^T
$$

Take the columns of $V$ as the principal component directions that have already been sorted in the correct order:

$$
V = [v_1,v_2,\ldots,v_d]
$$

③ Once you have identified all the principal components, you can reduce the dimensionality of the dataset down to $d$ dimensions by projecting it onto the hyperplane defined by the first $d$ principal components. Selecting this hyperplane ensures that the projection
will preserve as much variance as possible.

$$
Z=X_c V_d
$$

---

### Explained Variance Ratio

Another useful piece of information is the explained variance ratio of each principal component.

Suppose the eigenvalues are:

$$
\lambda_1, \lambda_2, \dots, \lambda_d
$$

Total variance is:
$$
\sum_{j=1}^{d} \lambda_j
$$

The explained variance ratio of PC $j$ is:
$$
\frac{\lambda_j}{\sum_{k=1}^{d} \lambda_k}
$$

For the first $k$ components, cumulative explained variance is:

$$
\frac{\lambda_1 + \dots + \lambda_k}{\lambda_1 + \dots + \lambda_d}
$$

The ratio indicates the proportion of the dataset’s variance that lies along each principal component. For example, if a data have explained variance ratio of $[0.82279334, 0.10821224]$ then about $82\%$ of the dataset’s variance lies along the first PC, and about $11\%$ lies along the second PC. We can usually use this information to choose the right number of PCs.






