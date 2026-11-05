---
title: "Dimensionality Reduction"
date: "2026-08-24"
chapter: 7
chapterTitle: "Dimensionality Reduction"
bookSlug: "hands-on-ml"
tags: ["machine-learning", "dimensionality-reduction", "pca", "random-projection", "manifold-learning", "lle", "isomap", "scikit-learn"]
description: "The curse of dimensionality, projection and manifold learning, PCA, random projection, LLE, and a comparison of MDS, Isomap, t-SNE, LDA, and UMAP."
---

**Dimensionality reduction** represents data using fewer features while retaining useful information. It can reduce computation and storage, support visualization, and sometimes remove noise. The important question is which information each method tries to retain, and whether that information helps the task.

Throughout these notes, $m$ is the number of instances, $n$ is the original number of features, and $d<n$ is the target dimension. The data matrix is $X\in\mathbb{R}^{m\times n}$, with one instance per row.

## The curse of dimensionality

As the number of dimensions grows, covering the feature space becomes much harder:

- **Sparse observations:** Maintaining the same sampling resolution along every axis requires exponentially more observations as dimensions are added. With a fixed dataset, points become sparse relative to the full space.
- **Less informative distances:** For many high-dimensional distributions, nearest and farthest distances become similar relative to their magnitude. This can make neighborhoods less meaningful for distance-based methods such as k-nearest neighbors.
- **More opportunities to fit noise:** Irrelevant features can obscure patterns and increase overfitting. Regularization and feature selection become more important.
- **Harder interpretation:** Relationships among many features are difficult to inspect or visualize directly.

These effects depend on the data's distribution and intrinsic structure. A large number of recorded features does not necessarily mean that all of them vary independently.

## Projection and manifold learning

### Projection onto a lower-dimensional subspace

Real-world observations often occupy only a small part of the available feature space. Some features may be nearly constant, while others are strongly correlated. The data may therefore lie close to a lower-dimensional subspace.

![Three-dimensional observations projected onto a two-dimensional plane, with the original handwritten arrow connecting the input to its new coordinates.](/images/notes/hands-on-ml/dimensionality-reduction/projection.png)

The new axes $z_1$ and $z_2$ define new features. This is different from simply deleting the original feature $x_3$: projection discards the component perpendicular to the chosen plane. In the illustrated example, that component contains mostly noise; this is not guaranteed for every dataset.

### Manifolds and intrinsic dimension

A **manifold** can be curved or complicated globally while resembling a simple, low-dimensional Euclidean space locally.

- A line in a plane is one-dimensional: one coordinate locates a point along it.
- A circle is also one-dimensional, even though it is embedded in two dimensions.
- The surface of a sphere is two-dimensional, although it is embedded in three dimensions. A sufficiently small patch looks approximately flat.

![Original hand-drawn line and circle, both labeled as one-dimensional manifolds embedded in a two-dimensional space.](/images/notes/hands-on-ml/dimensionality-reduction/manifold-examples.png)

![A sphere with a small tangent-plane patch illustrating its locally two-dimensional surface.](/images/notes/hands-on-ml/dimensionality-reduction/sphere.png)

The **Swiss roll** illustrates the same idea: the observations occupy a rolled-up two-dimensional sheet in three-dimensional space. Unrolling the sheet can expose its intrinsic coordinates.

![A Swiss-roll dataset in three dimensions alongside a lower-dimensional representation of the unrolled sheet.](/images/notes/hands-on-ml/dimensionality-reduction/swiss-roll.png)

The **manifold hypothesis** suggests that many high-dimensional real-world datasets lie near a much lower-dimensional manifold. For example, only a tiny fraction of arbitrary pixel arrangements resemble handwritten digits. The meaningful degrees of freedom in digit images can be far fewer than the number of pixels.

> **A useful representation depends on the task:** Recovering a lower-dimensional manifold does not guarantee a simpler classification boundary or better predictions. A boundary that is simple in the original space may become complicated after unrolling.

![Two Swiss-roll classification examples showing that a decision boundary can become either simpler or more complicated in a lower-dimensional representation.](/images/notes/hands-on-ml/dimensionality-reduction/decision-boundaries.png)

## Principal component analysis

**Principal component analysis (PCA)** finds orthogonal directions that capture the most variance in centered data, then projects onto the leading directions. For a fixed dimension, this also minimizes squared reconstruction error among orthogonal linear projections.

For the underlying derivation, see my [PCA article](/blog/pca/). In scikit-learn, use [`sklearn.decomposition.PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

### Choosing the number of components

The attribute `explained_variance_ratio_` gives the fraction of total variance captured by each component. If those fractions are $r_1,\ldots,r_n$, a variance target such as $95\%$ suggests choosing the smallest $d$ satisfying:

$$
\sum_{j=1}^{d}r_j\ge0.95.
$$

For example, assuming `X_train` is prepared:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95, svd_solver="full")
X_reduced = pca.fit_transform(X_train)
n_dimensions = pca.n_components_
```

Alternatively, plot cumulative explained variance against the number of dimensions and look for an **elbow**, where additional components provide diminishing gains.

![Cumulative explained variance increasing with the number of PCA components, with an elbow and a marked variance target.](/images/notes/hands-on-ml/dimensionality-reduction/pca-variance.png)

If PCA is preprocessing for a supervised task, tune `n_components` using validation performance, just like other hyperparameters. `RandomizedSearchCV` is one option. Fit PCA within the training folds rather than on the full dataset before cross-validation.

### Compression and reconstruction

Use `inverse_transform()` to reconstruct an approximation in the original feature space. With $C_d\in\mathbb{R}^{d\times n}$ containing the retained principal axes and $\boldsymbol{\mu}$ the training mean:

$$
Z=(X-\mathbf{1}\boldsymbol{\mu}^{\mathsf T})C_d^{\mathsf T},
\qquad
\hat{X}=ZC_d+\mathbf{1}\boldsymbol{\mu}^{\mathsf T}.
$$

Discarded components are not recovered, so compression generally introduces reconstruction error. Retaining a large fraction of variance also does not guarantee retaining all information useful for predicting a target.

### Randomized PCA

Set `svd_solver="randomized"` with an integer `n_components` to approximate the leading components using randomized truncated SVD. It can be much faster than full SVD when the desired number of components is small relative to the matrix dimensions.

### Incremental PCA

[`IncrementalPCA`](https://scikit-learn.org/stable/modules/decomposition.html#incremental-pca) processes data in mini-batches, making it useful when the full dataset does not fit comfortably in memory.

- Call `partial_fit()` on successive batches to learn the components.
- After fitting, transform batches using the learned representation.
- A memory-mapped array, such as one created with `numpy.memmap`, can support processing data stored on disk.

## Random projection

For very high-dimensional data, even approximating principal components may be expensive. **Random projection** uses a random linear map instead of learning directions of maximum variance:

$$
R\in\mathbb{R}^{d\times n},
\qquad
Z=XR^{\mathsf T}.
$$

With appropriate scaling and a sufficiently large target dimension, this can approximately preserve pairwise distances with high probability.

### The Johnson-Lindenstrauss intuition

For a finite collection of points and distortion tolerance $0<\varepsilon<1$, the desired relationship is:

$$
\begin{aligned}
(1-\varepsilon)\lVert\mathbf{x}_i-\mathbf{x}_j\rVert_2^2
&\le\lVert R\mathbf{x}_i-R\mathbf{x}_j\rVert_2^2\\
&\le(1+\varepsilon)\lVert\mathbf{x}_i-\mathbf{x}_j\rVert_2^2.
\end{aligned}
$$

A commonly used conservative dimension bound is:

$$
d\ge\frac{4\ln m}{\varepsilon^2/2-\varepsilon^3/3}.
$$

The bound depends on the number of instances and allowed distortion, rather than directly on the original feature count $n$. [`johnson_lindenstrauss_min_dim`](https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.johnson_lindenstrauss_min_dim.html) estimates this dimension. It is a conservative probabilistic bound, not the smallest dimension that will work on every particular dataset; it can even exceed the original dimension.

### Gaussian and sparse projection matrices

`GaussianRandomProjection` draws independently scaled Gaussian entries:

$$
R_{ij}\sim\mathcal{N}\!\left(0,\frac{1}{d}\right),
$$

where $1/d$ is the variance.

`SparseRandomProjection` uses many zero entries, which can reduce memory and computation. If $\rho$ is the density, each entry is nonzero with probability $\rho$. A nonzero entry has either sign with equal probability and magnitude:

$$
v=\frac{1}{\sqrt{d\rho}}.
$$

Consequently:

$$
\mathbb{E}[R_{ij}]=0,
\qquad
\operatorname{Var}(R_{ij})=\rho v^2=\frac{1}{d}.
$$

The automatic density is $\rho=1/\sqrt{n}$. See the [scikit-learn random projection guide](https://scikit-learn.org/stable/modules/random_projection.html) for the matrix constructions.

![Original example of a sparse random projection matrix with three rows, five columns, zeros, and signed nonzero entries of approximately 0.816.](/images/notes/hands-on-ml/dimensionality-reduction/sparse-projection-matrix.png)

The example matrix is:

$$
R=\begin{bmatrix}
0&0.816&-0.816&0&0.816\\
-0.816&0&0&0.816&0\\
0.816&0&-0.816&0&0
\end{bmatrix}.
$$

### Approximate inverse transformation

A pseudoinverse gives an approximate reconstruction:

$$
\hat{X}=Z\left(R^{\mathsf T}\right)^{+}
=Z\left(R^{+}\right)^{\mathsf T},
$$

where $+$ denotes the Moore-Penrose pseudoinverse, computable with `scipy.linalg.pinv()`.

> **Approximate reconstruction:** When $d<n$, a random projection generally discards information. The pseudoinverse does not undo that loss or recover the original data exactly.

## Locally linear embedding

**Locally linear embedding (LLE)** is a nonlinear manifold-learning method. Instead of finding one global linear projection, it describes each point using its neighbors, then finds lower-dimensional coordinates that retain those local reconstruction relationships.

![A Swiss roll and its LLE embedding, showing an unrolled but globally distorted shape.](/images/notes/hands-on-ml/dimensionality-reduction/lle-swiss-roll.png)

The illustrated roll is unwrapped, but the output is not a rectangle. LLE preserves local reconstruction structure rather than all global distances or the exact shape of the original sheet.

### Step 1: Find neighbors

For each instance $\mathbf{x}_i$, find its $k$ nearest neighbors, denoted by $\mathcal{N}_i$. The local-linearity assumption is that sufficiently small neighborhoods approximately occupy flat patches of the manifold.

### Step 2: Learn reconstruction weights

Approximate each point as a weighted combination of its neighbors:

$$
\mathbf{x}_i\approx\sum_{j=1}^{m}w_{ij}\mathbf{x}_j.
$$

Collect the weights in $W=(w_{ij})$ and solve:

$$
\hat{W}=\operatorname*{argmin}_{W}
\sum_{i=1}^{m}
\left\lVert\mathbf{x}_i-\sum_{j=1}^{m}w_{ij}\mathbf{x}_j\right\rVert_2^2,
$$

subject to:

$$
w_{ij}=0\quad\text{if }j\notin\mathcal{N}_i,
\qquad
\sum_{j=1}^{m}w_{ij}=1\quad\text{for every }i.
$$

The sum-to-one constraint makes the reconstruction invariant to a common translation of the data. The weights encode the local relationships between instances.

### Step 3: Find an embedding with the same weights

Keep $\hat{W}$ fixed and find coordinates $\mathbf{z}_i\in\mathbb{R}^{d}$ minimizing:

$$
\hat{Z}=\operatorname*{argmin}_{Z}
\sum_{i=1}^{m}
\left\lVert\mathbf{z}_i-\sum_{j=1}^{m}\hat{w}_{ij}\mathbf{z}_j\right\rVert_2^2.
$$

This embedding also needs centering and scale constraints, for example:

$$
\sum_{i=1}^{m}\mathbf{z}_i=\mathbf{0},
\qquad
\frac{1}{m}\sum_{i=1}^{m}\mathbf{z}_i\mathbf{z}_i^{\mathsf T}=I_d.
$$

Without a scale constraint, collapsing every point to the same location would trivially minimize the objective. These constraints and the sum-to-one condition follow the [LLE formulation by Saul and Roweis](https://www.jmlr.org/papers/volume4/saul03a/saul03a.pdf).

> **Equation corrections:** The reconstruction weights sum to $1$, not $0$. The embedding objective is not an unconstrained minimization: its normalization prevents a collapsed solution.

Use `sklearn.manifold.LocallyLinearEmbedding`. Standard LLE has computational stages that can grow quadratically with the number of instances, so it may become expensive on large datasets. See the [manifold learning guide](https://scikit-learn.org/stable/modules/manifold.html) for complexity details.

## Other dimensionality reduction methods

### Multidimensional scaling

**Multidimensional scaling (MDS)** places points in a lower-dimensional space so that their pairwise distances approximate a supplied dissimilarity matrix $D$:

$$
\lVert\mathbf{z}_i-\mathbf{z}_j\rVert_2\approx D_{ij}.
$$

For ordinary Euclidean input distances, $D_{ij}=\lVert\mathbf{x}_i-\mathbf{x}_j\rVert_2$. Other choices of dissimilarity allow different relationships to be represented.

### Isomap

**Isomap**, short for **isometric mapping**, aims to preserve geodesic distances: distances measured along the manifold, rather than straight through the surrounding space.

![A Swiss roll illustrating the difference between a geodesic path along the surface and a straight Euclidean path through the surrounding space.](/images/notes/hands-on-ml/dimensionality-reduction/isomap-geodesic.png)

Two points on neighboring layers of a rolled sheet may be close in three-dimensional Euclidean distance but far apart along the sheet. A linear projection can overlap such layers; Isomap uses a neighborhood graph to approximate the surface geometry.

![Original comparison of PCA and Isomap illustrating why straight-line distances can miss the structure of a curved manifold.](/images/notes/hands-on-ml/dimensionality-reduction/isomap-comparison.png)

The algorithm has three main stages:

1. **Construct a neighborhood graph.** Connect each instance to nearby instances, using Euclidean distances as edge weights.
2. **Compute shortest paths.** Approximate pairwise geodesic distances using a graph algorithm such as Dijkstra or Floyd-Warshall.
3. **Embed the distance matrix.** Apply classical MDS to the estimated geodesic distances to obtain lower-dimensional coordinates.

![Original Isomap reference panel describing nearest-neighbor graph construction and shortest-path distance computation.](/images/notes/hands-on-ml/dimensionality-reduction/isomap-steps.png)

The neighborhood size matters: too few neighbors can disconnect the graph, while too many can introduce shortcuts across folds. Use [`sklearn.manifold.Isomap`](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.Isomap.html), and inspect whether the resulting neighborhood graph represents the intended geometry.

### t-distributed stochastic neighbor embedding

**t-SNE** is a nonlinear method mainly used to visualize high-dimensional data in two or three dimensions. It emphasizes neighborhood similarities, making it useful for exploring local groups of related instances.

Treat it primarily as an exploratory visualization method, rather than a default preprocessing step for a predictive model. Separation in a plot should not be interpreted as proof of classes or as faithful preservation of all distances.

### Linear discriminant analysis

**Linear discriminant analysis (LDA)** uses class labels to learn linear directions that separate classes relative to their within-class variation. It can serve as a classifier or provide a supervised lower-dimensional representation before another classifier.

PCA maximizes variance without using labels; LDA explicitly uses the labels when choosing directions.

![PCA and LDA projection axes compared: PCA emphasizes overall variance while LDA emphasizes separation between labeled classes.](/images/notes/hands-on-ml/dimensionality-reduction/pca-lda-axes.png)

![Labeled observations projected using PCA and LDA, illustrating that maximum variance and maximum class separation can select different directions.](/images/notes/hands-on-ml/dimensionality-reduction/pca-lda-projections.png)

With $K$ classes, LDA provides at most $\min(n,K-1)$ discriminant components. See [`LinearDiscriminantAnalysis`](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html).

### Uniform manifold approximation and projection

**UMAP** is a nonlinear method used for visualization and more general dimensionality reduction. Its main ideas are:

1. Assume that data has meaningful lower-dimensional manifold structure.
2. Build a fuzzy topological representation of local neighborhoods, which can be viewed as a weighted graph.
3. Optimize a lower-dimensional embedding that retains those neighborhood relationships.

![UMAP diagram showing neighborhood relationships constructed in the original space and optimized in a lower-dimensional embedding.](/images/notes/hands-on-ml/dimensionality-reduction/umap-process.png)

UMAP offers a local-versus-global tradeoff: larger `n_neighbors` values emphasize broader structure, while smaller values emphasize local detail. Its [parameter guide](https://umap-learn.readthedocs.io/en/latest/parameters.html) illustrates this behavior.

UMAP can be a practical alternative to t-SNE, but speed and preservation of global structure depend on the dataset, parameters, and implementation. Neither method guarantees that all visible cluster distances accurately represent the original geometry.

## Comparing methods and combining reductions

The following table describes each method's objective or emphasis, rather than a guarantee of exact preservation:

| Method | Main emphasis | Uses class labels? |
| --- | --- | --- |
| PCA | Variance captured by a linear subspace | No |
| Random projection | Approximate pairwise Euclidean distances | No |
| MDS | Agreement with supplied pairwise dissimilarities | No |
| LLE | Local linear reconstruction relationships | No |
| Isomap | Estimated geodesic distances along the manifold | No |
| t-SNE | Local neighborhood similarities for visualization | No |
| LDA | Class separation relative to within-class variation | Yes |
| UMAP | Neighborhood structure, with a local/global tradeoff | No, in its standard unsupervised use |

![Original summary table comparing the main structures emphasized by PCA, random projection, MDS, LLE, Isomap, t-SNE, and UMAP.](/images/notes/hands-on-ml/dimensionality-reduction/methods-summary.png)

*The original reference table is preserved above; the text table also includes LDA and clarifies that these are objectives.*

![MDS, Isomap, and t-SNE embeddings of the same Swiss-roll data, illustrating their different representations.](/images/notes/hands-on-ml/dimensionality-reduction/embedding-comparison.png)

It is also possible to chain methods. For example, PCA or random projection can first reduce a very large feature space before applying a more expensive method such as LLE. This may improve runtime, but the first stage must retain the neighborhood information needed by the second. Compare the result with applying the second method directly rather than assuming equal quality.
