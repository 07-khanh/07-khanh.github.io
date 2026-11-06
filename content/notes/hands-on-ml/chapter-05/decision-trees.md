---
title: "Decision Trees"
date: "2026-08-18"
chapter: 5
chapterTitle: "Decision Trees"
bookSlug: "hands-on-ml"
tags: ["machine-learning", "decision-trees", "classification", "regression", "regularization", "scikit-learn"]
description: "How decision trees predict classes and values, estimate probabilities, choose CART splits, and control overfitting, with notes on impurity, complexity, and axis sensitivity."
---

Decision trees make predictions by following a sequence of feature-threshold tests. These notes cover how to interpret those tests, how CART chooses them, and why regularization and ensembles are useful for controlling a tree's flexibility.

## Training and visualizing a decision tree

Use `sklearn.tree.DecisionTreeClassifier` to fit a classification tree. A visualization makes its sequence of decisions explicit:

- `sklearn.tree.plot_tree` draws a fitted tree with Matplotlib.
- `sklearn.tree.export_graphviz` exports a Graphviz representation, which can be rendered using Graphviz tooling.

![An Iris classification tree splitting first on petal length at 2.45 cm and then on petal width at 1.75 cm, with class counts and Gini impurity at each node.](/images/notes/hands-on-ml/decision-trees/iris-tree.png)

*The example classifies Iris flowers using petal length and petal width.*

## Making predictions

Start at the root and evaluate the test at each internal node. Follow the left branch when the condition is true and the right branch otherwise, until reaching a leaf.

For the tree above:

1. If petal length is at most $2.45\,\text{cm}$, predict **Iris setosa**.
2. Otherwise, test petal width:
   - At most $1.75\,\text{cm}$: predict **Iris versicolor**.
   - Greater than $1.75\,\text{cm}$: predict **Iris virginica**.

![Original handwritten sketch showing the two Iris feature tests and their paths to setosa, versicolor, and virginica.](/images/notes/hands-on-ml/decision-trees/prediction-sketch.png)

> **Interpretability:** A small decision tree is a white-box model: an individual prediction can be explained by its path of tests. Very large trees can still be difficult to understand in full.

### Reading the node labels

- **`samples`:** the number of training instances reaching that node.
- **`value`:** the class counts displayed in this exported diagram, in the order of the class labels.
- **`class`:** the predicted class, chosen from the largest class count. Ties require a tie-breaking rule.
- **`gini`:** the node's Gini impurity, which measures how mixed its classes are.

To inspect a fitted scikit-learn tree programmatically, use its `tree_` attribute, including the child indices, split features, and thresholds. Class summaries may be represented as proportions rather than the counts shown in this figure; account for the library version and sample weights. See the [tree-structure example](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html).

### Gini impurity

For a node $i$ containing instances from $K$ classes:

$$
G_i = 1 - \sum_{k=1}^{K}p_{i,k}^{\,2},
$$

where $p_{i,k}$ is the fraction of instances in node $i$ belonging to class $k$.

- A pure node contains only one class and has $G_i=0$.
- More evenly mixed class proportions give higher impurity.
- For $K$ equally represented classes, $G_i=1-1/K$.

For example, the leaf with class counts $[0,49,5]$ has:

$$
G_i
=1-\left(\frac{0}{54}\right)^2
-\left(\frac{49}{54}\right)^2
-\left(\frac{5}{54}\right)^2
\approx 0.168.
$$

![Iris decision regions with axis-aligned boundaries labeled by tree depth.](/images/notes/hands-on-ml/decision-trees/iris-decision-boundaries.png)

*Each split adds a boundary perpendicular to one feature axis. Later splits only affect the region belonging to their node.*

## Estimating class probabilities

A classification tree can estimate probabilities as well as predict a class:

1. Traverse the tree to find the leaf reached by the new instance.
2. Return the class proportions among the training instances in that leaf.

For a leaf $\ell$:

$$
\hat{p}_k(\mathbf{x})
=\frac{m_{\ell,k}}{m_\ell},
$$

where $m_{\ell,k}$ is the number of class-$k$ instances in that leaf and $m_\ell$ is its total number of instances. With sample weights, use weighted class totals instead.

![The Iris tree with the original red arrows following a prediction path and a handwritten probability vector beside the leaf with counts 0, 49, and 5.](/images/notes/hands-on-ml/decision-trees/leaf-probabilities.png)

For a flower with petal length $5\,\text{cm}$ and petal width $1.5\,\text{cm}$:

- The first test is false because $5>2.45$.
- The second test is true because $1.5\leq1.75$.
- The flower reaches the leaf with class counts $[0,49,5]$.

The estimated probabilities, ordered as setosa, versicolor, and virginica, are:

$$
\hat{\mathbf{p}}
=\left(0,\frac{49}{54},\frac{5}{54}\right)
\approx(0,0.907,0.093).
$$

The predicted class is versicolor. Use `predict_proba()` to obtain the probability vector.

> **Same leaf, same probabilities:** These estimates depend on the leaf's training data, not on how close the new instance is to a split boundary. Small leaves can produce extreme estimates based on very little evidence.

## The CART training algorithm

Scikit-learn uses a version of **Classification and Regression Trees (CART)**. At each node, CART selects a feature and a threshold to create two child regions, then repeats the process recursively.

For a feature $j$ and threshold $t$, instances satisfying $x_j\leq t$ go left; the rest go right. The search considers feasible thresholds and eligible features, subject to settings such as `max_features`.

### The classification split objective

CART conducts an exhaustive search across every single feature and every possible split point (threshold) within those features to find the single best split. When using Gini impurity as the metric, the best split is the one that minimizes the sample-weighted impurity of the children:

$$
J(j,t)
=\frac{m_{\text{left}}}{m}G_{\text{left}}
+\frac{m_{\text{right}}}{m}G_{\text{right}},
\qquad
m=m_{\text{left}}+m_{\text{right}}.
$$

Here, $m$ is the number of instances at the **current node**, not necessarily the size of the entire training set. If sample weights are used, the corresponding weighted totals replace these counts.

- Evaluate candidate splits using the chosen impurity criterion.
- Select the best admissible split at the current node.
- Recursively split the child nodes.
- Stop according to node purity, split feasibility, depth, leaf-size constraints, and the other stopping settings.

> **A greedy algorithm:** CART chooses the best current split without searching all possible future trees. A locally good split does not guarantee a globally optimal tree. Optimal-tree learning is computationally hard under common formulations; this motivates greedy search, rather than implying that CART itself performs exhaustive exponential-time search. See the [decision-tree guide](https://scikit-learn.org/stable/modules/tree.html).

## Computational complexity

Let $m$ be the number of training instances, $n$ the number of features, and $d$ the tree depth.

- **Prediction:** traversing one root-to-leaf path takes $O(d)$ feature comparisons. A balanced tree often has $d=O(\log m)$, but an unbalanced tree can be much deeper.
- **Training:** an often-used estimate for efficient construction of a balanced tree is $O(nm\log m)$. The actual cost depends on split search, sorting and caching, tree shape, feature subsampling, and stopping constraints.
- Limiting depth or requiring larger leaves can reduce the amount of training work while also limiting overfitting.

> **An estimate, not a universal bound:** Neither balanced depth nor the quoted training cost is guaranteed for every dataset and implementation. The [complexity discussion](https://scikit-learn.org/stable/modules/tree.html#complexity) describes the assumptions behind these estimates.

## Gini impurity versus entropy

Entropy is an alternative measure of class mixture. For node $i$:

$$
H_i=-\sum_{\substack{k=1\\p_{i,k}>0}}^{K}
p_{i,k}\log_2(p_{i,k}).
$$

Terms with zero probability contribute zero by the convention $0\log 0=0$.

- Both Gini impurity and entropy are zero for a pure node.
- Both increase as the class distribution becomes more evenly mixed.
- They often produce similar trees, but can rank candidate splits differently.
- Gini does not require logarithms, although the practical runtime difference depends on the implementation and data.

For a classification tree, select the criterion using `criterion="gini"` or `criterion="entropy"`.

> **Further study:** The original notes ask why Gini and entropy sometimes create different class groupings. They assign different costs to mixed class distributions. Neither criterion guarantees balanced child sizes, so comparing the weighted costs of specific candidate splits is more reliable than treating either one as a rule for producing balanced trees.

## Regularization hyperparameters

A decision tree is **nonparametric**: its structure and number of fitted splits can grow with the data rather than being fixed by a short parameter vector. This flexibility also makes unrestricted trees prone to overfitting.

The following settings control how much detail a tree can learn:

| Hyperparameter | Role |
| --- | --- |
| `max_depth` | Caps the length of root-to-leaf paths. |
| `max_leaf_nodes` | Caps the number of terminal regions. |
| `min_samples_split` | Requires enough instances before an internal node can split. |
| `min_samples_leaf` | Requires enough instances in each resulting leaf. |
| `min_weight_fraction_leaf` | Requires each leaf to contain a minimum fraction of the total training-sample weight. |
| `max_features` | Restricts the features considered while searching for a split. |
| `min_impurity_decrease` | Requires a minimum weighted impurity improvement for a split. |
| `ccp_alpha` | Controls minimal cost-complexity pruning. |

Smaller depth and leaf-count limits, or larger minimum leaf sizes, generally restrict the tree. Feature subsampling changes the split search and can also affect variance; its effect on a single tree is not equivalent to a simple depth limit.

Increasing `ccp_alpha` favors smaller pruned subtrees. Compare settings using validation data. See the [`DecisionTreeClassifier` parameter reference](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

![Classification decision regions without restrictions compared with a tree requiring at least five samples per leaf.](/images/notes/hands-on-ml/decision-trees/classification-regularization.png)

*The minimum leaf-size constraint removes some of the narrow regions created by the unrestricted tree.*

As starting points for tuning, consider `max_depth` and `min_samples_leaf`; for high-dimensional data, also examine `max_features`.

## Regression trees

Decision trees can predict numeric targets using `sklearn.tree.DecisionTreeRegressor`. The traversal is the same as for classification, but each leaf returns a value instead of a class.

![A regression tree with feature thresholds, sample counts, squared errors, and numeric predicted values.](/images/notes/hands-on-ml/decision-trees/regression-tree.png)

### Predicting the mean target in a leaf

Under squared-error loss, the prediction for leaf $\ell$ is the mean target value of the training instances reaching it:

$$
\hat{y}_\ell=\frac{1}{m_\ell}\sum_{i\in I_\ell}y^{(i)},
$$

where $I_\ell$ is the set of training-instance indices in that leaf. This gives a **piecewise-constant** prediction function. With sample weights, the corresponding weighted mean is used.

![Regression predictions for maximum depths two and three, with handwritten annotations identifying split thresholds.](/images/notes/hands-on-ml/decision-trees/regression-depth.png)

*A deeper tree creates more constant-valued regions and follows finer details of the training data.*

### The regression split objective

The node's mean squared error is:

$$
\operatorname{MSE}_\ell
=\frac{1}{m_\ell}\sum_{i\in I_\ell}
\left(y^{(i)}-\hat{y}_\ell\right)^2.
$$

CART chooses the feature and threshold minimizing the weighted child errors:

$$
J(j,t)
=\frac{m_{\text{left}}}{m}\operatorname{MSE}_{\text{left}}
+\frac{m_{\text{right}}}{m}\operatorname{MSE}_{\text{right}}.
$$

This has the same structure as the classification objective, with MSE replacing class impurity.

### Preventing overfitting

![An unrestricted regression tree closely following individual training observations, compared with a smoother tree requiring ten samples per leaf.](/images/notes/hands-on-ml/decision-trees/regression-regularization.png)

An unrestricted regression tree can fit noise by creating very small leaves. Limits on depth, minimum leaf size, and pruning help control this behavior, just as they do in classification.

## Sensitivity to axis orientation

Standard CART splits test one feature at a time, producing boundaries perpendicular to feature axes. This makes the learned tree sensitive to rotations of the input space.

![The same two-class dataset before and after a 45-degree rotation, with the original handwritten diagonal showing a boundary an axis-aligned split cannot directly represent.](/images/notes/hands-on-ml/decision-trees/axis-orientation.png)

In the original orientation, one split separates the classes well. After a $45^\circ$ rotation, several axis-aligned splits are needed to approximate the diagonal separation.

One possible experiment is to scale the data and apply a [PCA transformation](/blog/pca/) before fitting the tree. This changes the coordinate system and may make the useful structure easier to represent with fewer splits.

> **Validate the transformation:** PCA follows directions of variance, not the class labels, so it is not guaranteed to improve predictions. Scaling alone does not remove axis-alignment limitations, and transformed features can make a tree harder to interpret.

## High variance and ensembles

A small change in the training data or hyperparameters can change an early split, which changes the regions available to every later split. Individual trees can therefore have high variance.

- Restricting the tree can improve stability.
- Combining predictions from many trees can reduce variance, especially when their errors are not strongly correlated.
- Random forests use this principle, averaging tree predictions for regression or class probabilities for classification.

This connects decision trees to the [ensemble-learning chapter](/notes/hands-on-ml/#chapter-06).
