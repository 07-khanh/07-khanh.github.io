---
title: "Unsupervised Learning Techniques"
date: "2026-08-28"
chapter: 8
chapterTitle: "Unsupervised Learning Techniques"
bookSlug: "hands-on-ml"
tags: ["machine-learning", "unsupervised-learning", "clustering", "k-means", "dbscan", "hdbscan", "gaussian-mixtures", "anomaly-detection", "scikit-learn"]
description: "Clustering with K-means, DBSCAN and HDBSCAN; Gaussian and Bayesian mixtures; and anomaly detection with robust covariance, isolation forests, LOF, and reconstruction error."
---

**Unsupervised learning** discovers structure in data without requiring a target label for every training instance. This chapter focuses on clustering, density estimation, and anomaly detection, together with applications to segmentation and learning from limited labels.

Throughout these notes, $m$ is the number of instances, $D$ is the number of features, and $K$ is the number of clusters or mixture components when one is specified. Each instance is a vector $\mathbf{x}_i\in\mathbb{R}^{D}$.

## K-means clustering

K-means groups observations around $K$ centroids. Its hard assignment for a point is the index of the nearest centroid:

$$
\hat{c}(\mathbf{x})=\operatorname*{argmin}_{j\in\{1,\ldots,K\}}
\lVert\mathbf{x}-\boldsymbol{\mu}_j\rVert_2^2.
$$

![Unlabeled observations arranged in several compact groups before clustering.](/images/notes/hands-on-ml/unsupervised-learning/clustering-data.png)

![K-means centroids and their colored Voronoi regions, retaining the original handwritten centroid annotation.](/images/notes/hands-on-ml/unsupervised-learning/kmeans-voronoi.png)

The nearest-centroid rule partitions the space into a **Voronoi tessellation**. K-means uses distances to centroids, so it can struggle when groups have very different diameters, densities, or shapes.

### Hard assignments and soft representations

A hard assignment returns one cluster index. An alternative representation gives each point one score per cluster, such as its distance to each centroid. Gaussian radial basis function similarities are another option:

$$
\phi_j(\mathbf{x})
=\exp\!\left(-\frac{\lVert\mathbf{x}-\boldsymbol{\mu}_j\rVert_2^2}{2\sigma^2}\right)
=\exp\!\left(-\gamma\lVert\mathbf{x}-\boldsymbol{\mu}_j\rVert_2^2\right),
\qquad \gamma=\frac{1}{2\sigma^2}.
$$

These scores can become features for another model. A $K$-dimensional distance representation reduces dimensionality only if $K<D$; otherwise it is a feature transformation or expansion. Distances and RBF similarities are not automatically cluster-membership probabilities.

### Lloyd's algorithm and inertia

The standard iterative procedure is:

1. Initialize $K$ centroids.
2. Assign every point to its nearest centroid.
3. Replace each centroid with the mean of its assigned points.
4. Repeat assignment and updating until convergence or a stopping criterion is reached.

![Four-stage illustration of K-means initialization, assignment, centroid updating, and repeated refinement.](/images/notes/hands-on-ml/unsupervised-learning/kmeans-iterations.png)

The objective is the within-cluster sum of squared distances, also called **inertia**:

$$
J=\sum_{i=1}^{m}\min_j
\lVert\mathbf{x}_i-\boldsymbol{\mu}_j\rVert_2^2.
$$

Ideal assignment and mean-update steps do not increase this objective. Convergence does not guarantee finding the global minimum: the result depends on initialization.

For $T$ iterations, the usual assignment work is approximately $O(TmKD)$. K-means is often fast in practice, but the number of iterations can be large on difficult datasets; this is not a universal linear-time guarantee.

### Centroid initialization

If approximate centroids are known, provide an array through `init`. Otherwise, use randomized initialization and compare multiple runs by inertia.

**K-means++** favors initial centers that are far from centers already selected:

1. Choose the first centroid uniformly from the training observations.
2. For every observation, calculate its distance to the nearest chosen centroid:

$$
\delta(\mathbf{x}_i)=\min_{\boldsymbol{\mu}\in C}
\lVert\mathbf{x}_i-\boldsymbol{\mu}\rVert_2,
$$

where $C$ is the current set of centers.

3. Sample the next center with probability:

$$
\Pr(\text{next center}=\mathbf{x}_i)
=\frac{\delta(\mathbf{x}_i)^2}{\sum_{r=1}^{m}\delta(\mathbf{x}_r)^2}.
$$

4. Repeat until $K$ centers have been selected, then run ordinary K-means updates.

![K-means++ example showing why points farther from existing centers are more likely to become the next center.](/images/notes/hands-on-ml/unsupervised-learning/kmeans-plus-plus.png)

Scikit-learn uses a **greedy K-means++** variant that tries several candidates at a selection step. In the current [`KMeans` API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), `n_init="auto"` uses one run with `init="k-means++"` and ten with `init="random"`. Set `n_init` explicitly when a particular number of restarts is intended.

### Accelerated and mini-batch variants

- **Elkan's algorithm:** Set `algorithm="elkan"` to use triangle-inequality bounds that avoid some unnecessary distance calculations. It targets the same K-means objective, but its speed benefit depends on the dataset and it uses additional memory.
- **Mini-batch K-means:** `MiniBatchKMeans` updates centers from small batches. It can improve runtime on large datasets, at the cost of approximating full-batch optimization.

### Limitations

- The number of clusters must be specified.
- Several initializations may be needed to avoid poor solutions.
- Compact, roughly spherical groups of comparable scale are easier to represent than elongated or curved groups.
- Feature scaling strongly affects Euclidean distances and therefore the result.

## Choosing the number of clusters

Inertia alone cannot determine the best $K$: the best achievable inertia decreases as more centers are allowed. A separate criterion is needed to balance fit and complexity.

![Examples with too few clusters merging groups and too many clusters splitting individual groups.](/images/notes/hands-on-ml/unsupervised-learning/cluster-count.png)

### The elbow method

Fit K-means for several values of $K$ and plot inertia. Look for an **elbow**, after which additional clusters bring much smaller improvements.

![Inertia versus cluster count, with a marked elbow where additional centers produce diminishing improvements.](/images/notes/hands-on-ml/unsupervised-learning/inertia-elbow.png)

This is a useful heuristic, but an elbow may be weak, ambiguous, or absent.

### Silhouette scores and diagrams

For a point $i$, define:

- $a_i$: its mean distance to the other points in its own cluster.
- $b_i$: the smallest mean distance to points in any other cluster.

Its **silhouette coefficient** is:

$$
s_i=\frac{b_i-a_i}{\max(a_i,b_i)},
\qquad -1\le s_i\le1.
$$

- Near $1$: the point is much closer to its own cluster than to neighboring clusters.
- Near $0$: it lies near an overlapping boundary.
- Below $0$: another cluster may fit the point better.

The silhouette score averages these coefficients. Scikit-learn defines the coefficient of a singleton cluster as zero; the score requires between two and $m-1$ distinct labels. See [`silhouette_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).

![Mean silhouette score plotted against the number of clusters.](/images/notes/hands-on-ml/unsupervised-learning/silhouette-scores.png)

A **silhouette diagram** groups the sorted coefficients by cluster. Horizontal extent indicates coefficient values; the vertical thickness of a cluster's band reflects its number of instances. The dashed line in the original figure marks the overall mean silhouette score.

![Annotated silhouette diagrams for three, four, five, and six clusters, showing coefficient distributions and the overall mean.](/images/notes/hands-on-ml/unsupervised-learning/silhouette-diagrams.png)

Look for negative coefficients and clusters containing many low-scoring points. In this example, both four and five clusters are plausible, and five gives more balanced sizes. Balanced sizes are not a requirement when the underlying groups are genuinely unequal.

## Applications of clustering

### Image segmentation

**Segmentation** partitions an image into meaningful pixel regions. Different tasks use different notions of a region:

- **Color segmentation:** Group pixels using color similarity, for example by replacing each pixel with its assigned color centroid.
- **Semantic segmentation:** Assign a category to each pixel, such as road, person, or car. Different objects of the same category share a class label.
- **Instance segmentation:** Distinguish individual objects, including multiple objects from the same category.

![Room image divided into colored pixel regions, illustrating segmentation.](/images/notes/hands-on-ml/unsupervised-learning/color-segmentation.png)

![Street scene with pixel regions labeled by semantic categories.](/images/notes/hands-on-ml/unsupervised-learning/semantic-segmentation.png)

![Street scene with separate colored masks and boxes for individual object instances.](/images/notes/hands-on-ml/unsupervised-learning/instance-segmentation.png)

Color clustering is a simple segmentation tool. Semantic and instance labels require additional information beyond a K-means grouping of pixel colors; the figures illustrate distinct segmentation tasks.

### Semi-supervised learning

Semi-supervised learning combines a small labeled dataset with a larger unlabeled dataset. Clustering can help by:

1. Grouping related labeled and unlabeled observations.
2. Using reliable labels within a group to propose pseudo-labels for nearby observations.
3. Training a classifier using the resulting labeled and pseudo-labeled data.

Another strategy is to ask an expert to label representative observations near cluster centers, reducing redundant labeling. A centroid need not be an actual observation, so labeling the nearest representative instance is often more practical.

Points far from their assigned centers can be excluded from a propagation step when their labels are uncertain. This can reduce incorrect propagation, but also risks excluding legitimate minority patterns.

> **Clusters are not automatically classes:** Propagating a label assumes local groups have meaningful label consistency. `LabelPropagation` and `LabelSpreading` are graph-based semi-supervised algorithms, rather than simple majority votes over K-means clusters.

### Active learning

In **active learning**, the model requests labels for selected observations. Common strategies include:

- **Uncertainty sampling:** Query observations about which the model is least certain.
- **Diversity sampling:** Cover different parts of the data rather than requesting redundant examples.
- **Expected model improvement:** Prioritize labels expected to improve the model most.
- **Query by committee:** Request labels where multiple models disagree.

## DBSCAN

**Density-based spatial clustering of applications with noise (DBSCAN)** groups dense, connected regions and marks isolated observations as noise. Unlike K-means, it does not require spherical clusters or assign every point to a cluster.

![DBSCAN and K-means compared on curved groups, separated blobs, and data without a clear separation.](/images/notes/hands-on-ml/unsupervised-learning/dbscan-comparison.png)

Two parameters define a neighborhood-density rule:

- `eps`, written $\varepsilon$: the neighborhood radius.
- `min_samples`: the minimum neighborhood population for a point to be a core point, including the point itself.

### Core, border, and noise points

- A **core point** has at least `min_samples` points in its $\varepsilon$-neighborhood.
- A **border point** is not core, but lies within the neighborhood of a core point.
- A **noise point** is neither core nor border. Scikit-learn marks ordinary noise with label `-1`.

![DBSCAN neighborhood diagram labeling core points, a border point, and noise.](/images/notes/hands-on-ml/unsupervised-learning/dbscan-neighborhoods.png)

The algorithm visits an unprocessed point, checks whether it is core, and expands a cluster through reachable core neighborhoods. Border points join reachable clusters without expanding them. A point initially marked as noise can later be included as a border point.

DBSCAN works well when a common density scale separates meaningful groups. A single global radius can struggle with strongly varying densities. Distance computations, memory use, and high dimensionality can also become limiting.

## HDBSCAN

**Hierarchical DBSCAN (HDBSCAN)** builds a density-based cluster hierarchy and extracts persistent groups. It can accommodate groups with different densities without selecting one global DBSCAN radius.

### Mutual reachability distance

Let $\operatorname{core}_k(\mathbf{x})$ be a neighborhood-based core distance: the radius needed to reach a specified neighbor count. For the usual unscaled construction, define:

$$
d_{\mathrm{mr},k}(\mathbf{a},\mathbf{b})
=\max\!\left\{
\operatorname{core}_k(\mathbf{a}),
\operatorname{core}_k(\mathbf{b}),
d(\mathbf{a},\mathbf{b})
\right\}.
$$

Large core distances indicate sparse regions. Mutual reachability increases some connections through those regions, reducing the chance that isolated points create bridges between dense groups.

![HDBSCAN core-distance neighborhoods illustrating local density around several observations.](/images/notes/hands-on-ml/unsupervised-learning/hdbscan-core-distances.png)

![The same observations illustrating mutual reachability as the maximum of two core distances and the pairwise distance.](/images/notes/hands-on-ml/unsupervised-learning/hdbscan-mutual-reachability.png)

Neighbor-count conventions differ between implementations. Scikit-learn's [`HDBSCAN`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html) includes the point itself in `min_samples`; the `hdbscan` package does not. Account for that difference when comparing parameter values.

### Minimum spanning tree and hierarchy

Conceptually, connect every pair of observations with an edge weighted by mutual reachability. A **minimum spanning tree (MST)** retains enough edges to recover the connected-component hierarchy at different distance thresholds.

![Minimum spanning tree whose edge colors indicate mutual reachability distance.](/images/notes/hands-on-ml/unsupervised-learning/hdbscan-mst.png)

Sort MST edges by increasing weight and merge the components they connect. The resulting hierarchy shows how groups join as the distance threshold grows. Reading it in reverse shows groups splitting as density requirements become stricter.

![HDBSCAN single-linkage hierarchy showing component merges at different distance thresholds.](/images/notes/hands-on-ml/unsupervised-learning/hdbscan-hierarchy.png)

This represents a hierarchy of density connectivity, rather than repeatedly fitting independent ordinary DBSCAN models with arbitrary border assignments.

### Condensing the tree

Use `min_cluster_size` to distinguish persistent branches from small groups falling out of a cluster:

- A child branch below the minimum size is treated as points leaving the parent.
- When both children meet the minimum size, retain a genuine split into two branches.

This produces a smaller **condensed tree**. Branch widths indicate membership counts, and shrink as points leave. It is often plotted using inverse distance:

$$
\lambda=\frac{1}{\varepsilon}.
$$

![Condensed HDBSCAN tree with branch widths representing cluster membership and inverse distance on the vertical axis.](/images/notes/hands-on-ml/unsupervised-learning/hdbscan-condensed-tree.png)

### Extracting stable clusters

For a cluster $C$, let $\lambda_{\mathrm{birth}}(C)$ be the level where it appears and $\lambda_p$ the level where point $p$ leaves it. Its stability is:

$$
\operatorname{stability}(C)
=\sum_{p\in C}\left(\lambda_p-\lambda_{\mathrm{birth}}(C)\right).
$$

The excess-of-mass selection procedure compares a parent's stability with the combined selected stability of its descendants. It selects either the parent or descendants, rather than returning overlapping ancestor and descendant clusters.

![Condensed tree with the original outlines marking selected persistent clusters.](/images/notes/hands-on-ml/unsupervised-learning/hdbscan-selected-clusters.png)

The [HDBSCAN explanation](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) walks through these stages. Results still depend on `min_cluster_size`, `min_samples`, the distance metric, and the data. Noise labeling can be aggressive; runtime and memory depend on the implementation and neighbor-search difficulty.

## Other clustering algorithms

### Agglomerative clustering

**Agglomerative clustering** starts with one cluster per observation, then repeatedly merges the closest pair according to a linkage criterion. The hierarchy can be cut at a distance threshold or at a desired number of clusters.

![Agglomerative clustering illustrated as observations merging into progressively larger groups.](/images/notes/hands-on-ml/unsupervised-learning/agglomerative-process.png)

![An agglomerative dendrogram with a horizontal cut selecting a flat clustering.](/images/notes/hands-on-ml/unsupervised-learning/agglomerative-dendrogram.png)

The distance metric and linkage rule define what counts as a close pair of groups. Maintaining the hierarchy and inter-cluster relationships can be computationally expensive.

### BIRCH

**BIRCH**, implemented as `Birch`, is designed for large datasets. It compresses observations into a tree of compact clustering summaries, then clusters those summaries rather than repeatedly comparing every original observation.

### Mean shift and kernel density estimation

**Mean shift** searches for modes, or local peaks, of a density estimate. Points move toward nearby high-density regions, and positions converging to the same mode form a group. The bandwidth controls the scale of the modes, so a cluster count need not be specified directly.

![Static preview from the original mean-shift animation reference, showing observations and density contours.](/images/notes/hands-on-ml/unsupervised-learning/mean-shift-preview.png)

**Kernel density estimation (KDE)** replaces discrete observations with a smooth density estimate:

$$
\hat{f}(\mathbf{x})
=\frac{1}{mh^D}\sum_{i=1}^{m}
\mathcal{K}\!\left(\frac{\mathbf{x}-\mathbf{x}_i}{h}\right),
$$

where $h>0$ is the bandwidth and $\mathcal{K}$ is a normalized kernel. Larger bandwidths produce smoother estimates.

For a Gaussian-kernel version, the mean-shift vector is a weighted local mean minus the current position:

$$
\mathbf{v}(\mathbf{x})
=\frac{\sum_{i=1}^{m}\mathbf{x}_i
\exp\!\left(-\lVert\mathbf{x}-\mathbf{x}_i\rVert_2^2/(2h^2)\right)}
{\sum_{i=1}^{m}
\exp\!\left(-\lVert\mathbf{x}-\mathbf{x}_i\rVert_2^2/(2h^2)\right)}
-\mathbf{x}.
$$

Update $\mathbf{x}\leftarrow\mathbf{x}+\mathbf{v}(\mathbf{x})$ until the position stabilizes. This Gaussian formulation illustrates the density-gradient intuition; scikit-learn's [`MeanShift`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html) uses a flat kernel. Repeated neighborhood searches can make mean shift costly on large datasets.

### Affinity propagation

**Affinity propagation** exchanges messages between observations to identify representative observations called **exemplars**:

- **Responsibilities** express how suitable a candidate exemplar is for a point relative to competing candidates.
- **Availabilities** express how appropriate it is for a point to choose a candidate, taking support from other observations into account.

![Original static video thumbnail introducing affinity propagation.](/images/notes/hands-on-ml/unsupervised-learning/affinity-propagation-preview.png)

![Affinity propagation diagram distinguishing responsibility messages from availability messages.](/images/notes/hands-on-ml/unsupervised-learning/affinity-messages.png)

The number of exemplars emerges from similarities and preferences. Pairwise message storage makes the method less suitable for very large datasets.

### Spectral clustering

**Spectral clustering** constructs a low-dimensional representation from a similarity graph, typically using graph-Laplacian eigenvectors, then assigns clusters in that representation, often with K-means.

It can capture curved or otherwise complex groups. The similarity construction matters, and dense pairwise representations can be expensive. Groups with very different sizes or weak graph separation can also be challenging.

## Gaussian mixture models

A **Gaussian mixture model (GMM)** assumes observations arise from a mixture of Gaussian components. It estimates a probability density and supports soft assignments to components.

![Two weighted one-dimensional Gaussian components and their combined mixture density.](/images/notes/hands-on-ml/unsupervised-learning/gaussian-mixture.png)

### The mixture density

For $K$ components:

$$
p(\mathbf{x})=\sum_{k=1}^{K}
\pi_k\,\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k,\Sigma_k),
\qquad \pi_k\ge0,
\qquad \sum_{k=1}^{K}\pi_k=1.
$$

Each component has:

- A **mixing weight** $\pi_k$, the probability of selecting that component.
- A **mean** $\boldsymbol{\mu}_k\in\mathbb{R}^{D}$.
- A **covariance matrix** $\Sigma_k$, determining spread, shape, and orientation.

For a positive-definite covariance matrix, the Gaussian density is:

$$
\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k,\Sigma_k)
=\frac{\exp\!\left[
-\tfrac12(\mathbf{x}-\boldsymbol{\mu}_k)^{\mathsf T}
\Sigma_k^{-1}(\mathbf{x}-\boldsymbol{\mu}_k)
\right]}
{(2\pi)^{D/2}|\Sigma_k|^{1/2}}.
$$

The latent variable $z_i$ identifies which component generated observation $i$. Its value is unknown, so fitting must infer component membership together with the parameters.

### Expectation-maximization

**Expectation-maximization (EM)** alternates between soft assignment and parameter updates.

1. **Initialize:** Choose initial means, covariances, and weights, for example using K-means assignments.
2. **E-step:** Compute each component's responsibility for each point using Bayes' rule:

$$
\gamma_{ik}
=\Pr(z_i=k\mid\mathbf{x}_i)
=\frac{\pi_k\mathcal{N}(\mathbf{x}_i\mid\boldsymbol{\mu}_k,\Sigma_k)}
{\sum_{j=1}^{K}\pi_j\mathcal{N}(\mathbf{x}_i\mid\boldsymbol{\mu}_j,\Sigma_j)}.
$$

3. **M-step:** Use the responsibilities as weights. First compute the effective component count:

$$
N_k=\sum_{i=1}^{m}\gamma_{ik}.
$$

Then update:

$$
\pi_k^{\mathrm{new}}=\frac{N_k}{m},
\qquad
\boldsymbol{\mu}_k^{\mathrm{new}}
=\frac{1}{N_k}\sum_{i=1}^{m}\gamma_{ik}\mathbf{x}_i,
$$

$$
\Sigma_k^{\mathrm{new}}
=\frac{1}{N_k}\sum_{i=1}^{m}\gamma_{ik}
(\mathbf{x}_i-\boldsymbol{\mu}_k^{\mathrm{new}})
(\mathbf{x}_i-\boldsymbol{\mu}_k^{\mathrm{new}})^{\mathsf T}.
$$

Repeat until improvement in the total log-likelihood becomes sufficiently small, or an iteration limit is reached:

$$
\ell(\theta)
=\sum_{i=1}^{m}\ln p(\mathbf{x}_i\mid\theta)
=\sum_{i=1}^{m}\ln\!\left[
\sum_{k=1}^{K}\pi_k\mathcal{N}(\mathbf{x}_i\mid\boldsymbol{\mu}_k,\Sigma_k)
\right].
$$

The ideal EM updates do not decrease likelihood, but initialization still matters. Multiple restarts, such as explicitly setting `n_init=10`, can help avoid poor solutions. Covariance regularization helps prevent numerical problems from components collapsing onto too few observations.

For the distinction between probability and likelihood, and the connection between MLE, MAP, and regularization, see the companion note on the [likelihood function](/notes/hands-on-ml/chapter-08/likelihood-function/).

### Covariance types and prediction

Use `sklearn.mixture.GaussianMixture`. Its `covariance_type` controls component geometry:

| Type | Covariance structure | Geometry |
| --- | --- | --- |
| `full` | A separate full covariance per component | Independently oriented ellipsoids |
| `tied` | One full covariance shared by all components | Shared shape and orientation |
| `diag` | A separate diagonal covariance per component | Axis-aligned ellipsoids |
| `spherical` | One variance per component | Spheres, potentially of different sizes |

![Gaussian mixture fits using spherical, diagonal, tied, and full covariance structures.](/images/notes/hands-on-ml/unsupervised-learning/covariance-types.png)

For a new point, evaluate the same responsibility formula using the fitted parameters. `predict_proba()` returns component responsibilities; `predict()` chooses the largest. Component indices are not semantic class labels without an additional interpretation or label mapping.

A GMM is also generative: `sample()` can draw new observations by choosing a component and sampling from its Gaussian. Computation depends strongly on feature dimension, component count, and covariance type.

### Anomaly detection using density

Observations in low-density regions can be flagged as potential anomalies. `score_samples()` returns one log-density value per observation:

$$
\mathbf{x}\text{ is flagged if }\ln p(\mathbf{x})<\tau.
$$

Two practical threshold-selection strategies are:

- If an anomaly fraction $\alpha$ is assumed, use the corresponding lower quantile of reference log-densities:

$$
\tau=Q_{\alpha}\!\left(
\{\ln p(\mathbf{x}_i)\}_{i=1}^{m}
\right).
$$

- If labeled validation examples are available, sweep thresholds and compare precision-recall behavior, $F_1$, or an $F_{\beta}$ score appropriate to the task. Negate log-density when using a scoring convention where larger values indicate anomalies.

> **Density is not a probability of being an anomaly:** A density threshold encodes a modeling choice. A chosen quantile flags roughly that fraction of the reference data; it does not establish the true anomaly rate.

### Selecting the component count

Training likelihood tends to improve when components are added, so maximizing it alone encourages excessive complexity. Two penalized criteria are:

$$
\operatorname{BIC}=-2\ell(\hat{\theta})+q\ln m,
\qquad
\operatorname{AIC}=-2\ell(\hat{\theta})+2q,
$$

where $q$ is the number of fitted free parameters and $\ell(\hat{\theta})$ is the fitted model's total log-likelihood.

Choose smaller values. Compare component counts and covariance types on a consistent dataset. For ordinary sample sizes where $\ln m>2$, BIC imposes a stronger per-parameter penalty than AIC.

![AIC and BIC plotted against the number of Gaussian components, with a minimum marked.](/images/notes/hands-on-ml/unsupervised-learning/gmm-information-criteria.png)

## Bayesian Gaussian mixtures

Bayesian mixtures put prior distributions on weights, means, and covariances, then infer their posterior distributions. For example, a mean vector can be assigned a Gaussian prior:

$$
\boldsymbol{\mu}_k\sim\mathcal{N}(\mathbf{m}_0,S_0).
$$

Priors regularize the model. With a suitable weight prior and a sufficiently large component limit, unnecessary components can receive weights close to zero. The upper limit and prior settings still need to be chosen; the inferred effective count is not independent of these choices.

### Posterior predictive inference

Let $\theta$ collect the mixture parameters and let $X$ denote the observed training data. Exact Bayesian prediction integrates over parameter uncertainty:

$$
p(\mathbf{x}_*\mid X)
=\int p(\mathbf{x}_*\mid\theta)\,p(\theta\mid X)\,d\theta
=\sum_{k=1}^{K}a_k(\mathbf{x}_*),
$$

where:

$$
a_k(\mathbf{x}_*)
=p(\mathbf{x}_*,z_*=k\mid X)
=\int\pi_k\mathcal{N}(\mathbf{x}_*\mid\boldsymbol{\mu}_k,\Sigma_k)
\,p(\theta\mid X)\,d\theta.
$$

Consequently:

$$
\Pr(z_*=k\mid\mathbf{x}_*,X)
=\frac{a_k(\mathbf{x}_*)}{\sum_{j=1}^{K}a_j(\mathbf{x}_*)},
\qquad
\hat{z}_*=\operatorname*{argmax}_k
\Pr(z_*=k\mid\mathbf{x}_*,X).
$$

This integrates the joint component contribution before normalizing. Replacing parameters by posterior means is a different, plug-in approximation. For example, with $\bar{\pi}_k=\mathbb{E}[\pi_k\mid X]$, and corresponding mean estimates $\bar{\boldsymbol{\mu}}_k$ and $\bar{\Sigma}_k$, it would use:

$$
r_k^{\mathrm{plug}}(\mathbf{x}_*)
=\frac{\bar{\pi}_k\mathcal{N}(\mathbf{x}_*\mid\bar{\boldsymbol{\mu}}_k,\bar{\Sigma}_k)}
{\sum_{j=1}^{K}\bar{\pi}_j\mathcal{N}(\mathbf{x}_*\mid\bar{\boldsymbol{\mu}}_j,\bar{\Sigma}_j)}.
$$

### Variational inference in scikit-learn

[`BayesianGaussianMixture`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html) uses **variational inference** to approximate the posterior. It optimizes a lower bound on model evidence, including priors.

Its assignment calculations use expected log-weights and expected Gaussian log-density terms under the variational distribution. They are not simply ordinary Gaussian responsibilities evaluated at posterior mean parameters. This distinction can be seen in the [implementation of the variational log-probability and weight calculations](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/mixture/_bayesian_mixture.py).

### Limits of Gaussian components

Each Gaussian component has ellipsoidal density contours. A mixture can approximate more complicated densities using several components, but a curved semantic group may then be split across many components.

![Curved two-moons data and a Gaussian mixture fit illustrating the mismatch between curved groups and individual Gaussian components.](/images/notes/hands-on-ml/unsupervised-learning/gmm-nonlinear-limitations.png)

Increasing flexibility costs computation and can complicate the interpretation of components as clusters. Model choice should reflect whether the goal is density estimation, grouping, or prediction.

## Anomaly and novelty detection

- **Outlier detection:** Training data may already contain anomalies. The method seeks unusual observations mixed into that dataset.
- **Novelty detection:** Training data is intended to represent normal behavior. The fitted model then flags unusual new observations.

This distinction affects which fitting and scoring APIs are appropriate. The [scikit-learn guide](https://scikit-learn.org/stable/modules/outlier_detection.html) compares the methods below.

### Robust covariance and Fast-MCD

The **minimum covariance determinant (MCD)** approach estimates a robust center and covariance by finding a sufficiently large, tightly concentrated subset of observations.

For subsets $H$ of a chosen size $h$, the idealized objective is:

$$
H^*=\operatorname*{argmin}_{H:\,|H|=h}\det(\Sigma_H),
$$

where $\Sigma_H$ is the subset covariance. At a fixed contour level, ellipsoid volume is proportional to $\sqrt{\det(\Sigma_H)}$. Minimizing the determinant seeks a compact subset without letting distant points dominate the fit.

Exhaustively checking $\binom{m}{h}$ subsets is impractical. **Fast-MCD** uses efficient search and refinement, with consistency corrections and reweighting in practical estimators. Scikit-learn provides robust covariance estimation through `MinCovDet` and an outlier detector through `EllipticEnvelope`.

Using robust estimates $\hat{\boldsymbol{\mu}}$ and $\hat{\Sigma}$, define the robust Mahalanobis distance:

$$
\operatorname{RMD}(\mathbf{x})
=\sqrt{(\mathbf{x}-\hat{\boldsymbol{\mu}})^{\mathsf T}
\hat{\Sigma}^{-1}(\mathbf{x}-\hat{\boldsymbol{\mu}})}.
$$

The companion note on [Mahalanobis distance](/notes/hands-on-ml/chapter-08/mahalanobis-distance/) derives this expression through whitening and explains how feature correlations change the distance geometry.

Under an ideal Gaussian model with known parameters, squared Mahalanobis distance follows $\chi_D^2$. A reference rule might flag:

$$
\operatorname{RMD}(\mathbf{x})^2>\chi^2_{D,0.99}.
$$

Estimated robust parameters and finite samples affect that approximation; the $99\%$ cutoff is an illustrative rule, not an exact guarantee or the default `EllipticEnvelope` decision rule.

This approach suits an approximately elliptical central population. Multiple unrelated modes, strongly non-elliptical geometry, or too few observations relative to dimension can undermine the covariance estimate.

### Isolation Forest

**Isolation Forest** uses random partition trees. Its intuition is that rare, distinct observations often require fewer random splits to isolate than typical observations.

![Random splits isolating an anomalous observation with fewer cuts than a typical observation.](/images/notes/hands-on-ml/unsupervised-learning/isolation-splits.png)

![An isolation tree showing a shallow path for an easy-to-isolate point and deeper paths for typical points.](/images/notes/hands-on-ml/unsupervised-learning/isolation-tree.png)

For each tree:

1. Draw a random subsample of size $\psi$, often a few hundred observations.
2. At a node, randomly choose a feature and a split value between its minimum and maximum within that node.
3. Recursively partition until observations are isolated or a stopping rule is reached.
4. Measure path lengths and average across trees, including the usual remaining-leaf-size correction when a leaf contains multiple observations.

Normalize the mean path length using the expected unsuccessful-search length in a random binary search tree:

$$
c(\psi)=2H_{\psi-1}-\frac{2(\psi-1)}{\psi},
\qquad
H_r=\sum_{j=1}^{r}\frac{1}{j},
$$

with $H_r\approx\ln r+\gamma_{\mathrm{E}}$ for large $r$, where $\gamma_{\mathrm{E}}\approx0.577$ is Euler's constant. Here $\psi\ge2$ is the tree subsample size, rather than necessarily the full training-set size.

The original anomaly-score convention is:

$$
s(\mathbf{x},\psi)
=2^{-\mathbb{E}[h(\mathbf{x})]/c(\psi)}.
$$

Shorter paths give larger scores. A mean path length equal to $c(\psi)$ gives $s=0.5$. These are measures of relative isolation, rather than probabilities or certain declarations of an anomaly.

> **Check the score direction:** [`IsolationForest.score_samples()`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) uses the opposite convention from the original score: lower values mean more abnormal observations. Its thresholded predictions use `-1` for outliers and `+1` for inliers.

Subsampling can make Isolation Forest practical on large datasets without fitting a Gaussian density. Many irrelevant features and the standard trees' axis-aligned cuts can make meaningful anomalies harder to isolate. Runtime still depends on the number of trees, subsample size, and feature processing.

### Local outlier factor

**Local outlier factor (LOF)** compares a point's local density with the densities of its neighbors. It can identify an observation that is sparse relative to its surroundings, even when other parts of the dataset have different densities.

Let $k\text{-distance}(A)$ be the distance from $A$ to its $k$th nearest other observation. In the original definition, $N_k(A)$ contains observations at or within that distance, excluding $A$ itself; ties can make its size exceed $k$.

Define the **reachability distance** from $A$ with respect to $B$ as:

$$
\operatorname{reach}_k(A,B)
=\max\!\left\{k\text{-distance}(B),\,d(A,B)\right\}.
$$

This puts a floor on short distances using $B$'s neighborhood scale. It smooths local density estimates and is generally asymmetric.

![Original LOF diagram illustrating reachability distances around a point and its neighboring observations.](/images/notes/hands-on-ml/unsupervised-learning/lof-reachability.png)

The **local reachability density** is the reciprocal of the average reachability distance:

$$
\operatorname{lrd}_k(A)
=\frac{|N_k(A)|}{\sum_{B\in N_k(A)}\operatorname{reach}_k(A,B)}.
$$

LOF then averages the ratio of neighboring densities to the point's own density:

$$
\operatorname{LOF}_k(A)
=\frac{1}{|N_k(A)|}
\sum_{B\in N_k(A)}
\frac{\operatorname{lrd}_k(B)}{\operatorname{lrd}_k(A)}.
$$

- Around $1$: density is similar to the surrounding neighborhood.
- Below $1$: the point is denser than its neighbors on this measure.
- Substantially above $1$: the point is sparser and may be an outlier.

LOF is sensitive to neighborhood size, feature scaling, and the distance metric. Neighborhood search can become expensive in high dimensions, and a score still needs an appropriate decision threshold.

In [`LocalOutlierFactor`](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html), the default mode detects outliers in the fitted data using `fit_predict()` and exposes `negative_outlier_factor_`. Set `novelty=True` when the goal is scoring new observations after fitting; in that mode, use `predict()`, `decision_function()`, and `score_samples()` only on unseen data. Larger original LOF values correspond to more negative scikit-learn scores.

### Reconstruction error

A lower-dimensional model can learn to reconstruct common patterns. Unusual observations may have a large reconstruction error:

$$
e(\mathbf{x})=\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert_2^2.
$$

PCA and other models with a suitable `inverse_transform()` can support this approach. See [compression and reconstruction in Chapter 07](/notes/hands-on-ml/chapter-07/dimensionality-reduction/#compression-and-reconstruction). Choose the representation and error threshold using the task's data; anomalies that lie within the learned subspace may reconstruct well.

### One-class SVM

**One-class SVM** learns a boundary around a reference population and can flag new observations outside it. `sklearn.svm.OneClassSVM` supports nonlinear boundaries through kernels. Feature scaling and the kernel and `nu` parameters affect the result; the standard kernel method can be expensive on large datasets. See the [novelty detection guide](https://scikit-learn.org/stable/modules/outlier_detection.html#novelty-detection).
