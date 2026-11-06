---
title: "Ensemble Learning and Random Forests"
date: "2026-08-21"
chapter: 6
chapterTitle: "Ensemble Learning and Random Forests"
bookSlug: "hands-on-ml"
tags: ["machine-learning", "ensemble-learning", "random-forests", "bagging", "boosting", "adaboost", "gradient-boosting", "stacking", "scikit-learn"]
description: "Voting, bagging and out-of-bag evaluation, random forests, feature importance, AdaBoost, gradient boosting, and stacking, with annotated figures and worked formulas."
---

**Ensemble learning** combines the predictions of several models. The aim is to obtain better predictions than an individual model by combining useful predictors whose errors are not all the same: the wisdom of the crowd.

Ensembles are useful after building several promising predictors, and some, such as random forests, are also practical baselines. Their accuracy gains come with additional training, memory, and inference costs, so compare the benefit with the requirements of the task.

## Voting classifiers

### Hard voting

A **hard-voting classifier** predicts the class receiving the most votes from its base classifiers:

$$
\hat{y}(\mathbf{x})
=\operatorname*{argmax}_k
\sum_{b=1}^{B}\mathbb{1}\left[h_b(\mathbf{x})=k\right],
$$

where $B$ is the number of classifiers, $h_b$ is classifier $b$, and $\mathbb{1}[A]$ is $1$ when $A$ is true and $0$ otherwise. Ties need a tie-breaking rule.

![Diverse classifiers voting on the same new instance, with three votes for class 1 and one for class 2.](/images/notes/hands-on-ml/ensemble-learning/hard-voting.png)

Even individually weak classifiers can form a stronger ensemble when they are sufficiently accurate and make sufficiently different errors. Combining many copies that make identical mistakes offers little benefit.

### The law-of-large-numbers intuition

Consider independent tosses of a coin whose probability of heads is $p=0.51$. As the number of tosses grows, the observed fraction of heads approaches $0.51$, making a heads majority increasingly likely.

![Repeated biased-coin experiments with the heads ratio approaching 51 percent as the number of tosses increases.](/images/notes/hands-on-ml/ensemble-learning/law-of-large-numbers.png)

For $N$ independent tosses, the probability of a strict heads majority is:

$$
\Pr(\text{strict majority})
=\sum_{j=\lfloor N/2\rfloor+1}^{N}
\binom{N}{j}p^j(1-p)^{N-j}.
$$

For $N=1000$ and $p=0.51$, this is approximately $0.726$, or $72.6\%$. If ties are counted as heads wins, the probability is about $74.7\%$.

The same calculation illustrates an idealized ensemble of independent binary classifiers, each correct with probability $0.51$. In practice, models trained on related data usually have correlated errors, so this calculation is an intuition, not an accuracy guarantee.

> **Diversity matters:** Prefer models that contribute different useful information. Merely increasing the number of highly correlated predictors does not reproduce the independent-vote example.

### Soft voting

If every classifier provides class probabilities, **soft voting** averages those probabilities and predicts the class with the largest average:

$$
\hat{p}_k(\mathbf{x})
=\frac{1}{B}\sum_{b=1}^{B}\hat{p}_{b,k}(\mathbf{x}),
\qquad
\hat{y}(\mathbf{x})=\operatorname*{argmax}_k\hat{p}_k(\mathbf{x}).
$$

- Use `sklearn.ensemble.VotingClassifier` with `voting="hard"` or `voting="soft"`.
- Soft voting uses the strength of each probability estimate, rather than counting every predicted class equally.
- It can outperform hard voting when the probabilities are informative and well calibrated; it is not automatically better just because `predict_proba()` exists.
- `sklearn.calibration.CalibratedClassifierCV` can help calibrate probability estimates using appropriate validation data. See the companion note on [probability calibration](/notes/hands-on-ml/chapter-06/probability-calibration/) for reliability diagrams and calibration methods.

## Bagging and pasting

**Bagging** and **pasting** train multiple instances of the same base algorithm on different random subsets of the training data, then aggregate their predictions. These base models can usually be trained independently and in parallel.

- **Bagging**, or bootstrap aggregating, samples with replacement within each subset.
- **Pasting** samples without replacement within each subset.
- In either method, an instance may appear in the subsets used by several different estimators.

![Multiple base predictors trained on randomly sampled datasets drawn from a common training set.](/images/notes/hands-on-ml/ensemble-learning/bagging-process.png)

For regression, the aggregate is typically an average. For classification, an ensemble can use votes or average probabilities. In scikit-learn, `BaggingClassifier` averages probabilities when its base estimators provide them and otherwise uses voting. See the [BaggingClassifier reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html).

Bagging is especially useful for unstable, high-variance estimators such as deep decision trees. Its main purpose is variance reduction; it does not guarantee a reduction in both bias and variance.

### A scikit-learn example

Assuming `X_train` and `y_train` are prepared and contain at least 100 instances:

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=100,
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
)
bag_clf.fit(X_train, y_train)
```

- `n_estimators=500`: train 500 trees.
- `max_samples=100`: draw 100 training instances for each tree, with replacement in this example.
- `n_jobs=-1`: use the available CPU cores for supported parallel operations.

![Decision boundaries from a single tree and a bagged collection of trees, with the original handwritten generalization annotation.](/images/notes/hands-on-ml/ensemble-learning/bagging-boundaries.png)

*The ensemble smooths some of the individual tree's irregular decisions in this example.*

Bootstrap samples contain repeated observations and fewer distinct observations than their draw count. This can increase diversity among models and trade some individual-model fit for lower ensemble variance. Whether bagging or pasting works better depends on the data, subset size, and base learner; compare them using validation rather than treating either as universally preferable.

## Out-of-bag evaluation

A bootstrap sample leaves some training instances unused by a particular estimator. These are that estimator's **out-of-bag (OOB)** instances.

With $m$ training instances and $b$ independent draws with replacement, the probability that one particular instance is never selected is:

$$
\Pr(\text{OOB})=\left(1-\frac{1}{m}\right)^b.
$$

If each estimator draws $b=m$ instances:

$$
\Pr(\text{OOB})
=\left(1-\frac{1}{m}\right)^m
\longrightarrow e^{-1}\approx0.368.
$$

Thus, approximately $36.8\%$ of the training instances are left out of each full-size bootstrap sample when $m$ is large. A draw count such as `max_samples=100` does not generally give that same percentage unless it matches the training-set size.

### Computing an OOB prediction

For each training instance:

1. Find the fitted estimators whose bootstrap samples did not include it.
2. Aggregate only those estimators' predictions for that instance.
3. Compare the resulting prediction with its true target.

Repeating this for the training data gives an OOB estimate without reserving a separate validation subset for every tree. Use enough estimators that instances receive useful OOB predictions.

Set `oob_score=True` before fitting a bootstrap-enabled `BaggingClassifier`; inspect `oob_score_` afterward.

> **An evaluation estimate:** OOB evaluation is useful for bagging, but repeated model selection can still overfit an evaluation procedure. Keep the final evaluation separate from the choices made during training and tuning.

## Random subspaces and random patches

Sampling can act on feature columns as well as training-instance rows:

| Method | Instance sampling | Feature sampling |
| --- | --- | --- |
| Random subspaces | Keep the training instances | Use a random subset of features |
| Random patches | Use a random subset of instances | Use a random subset of features |

This can be useful for high-dimensional inputs, such as images or text representations. Different feature subsets give the estimators different views of the data.

In a bagging estimator, control row and feature sampling using `max_samples`, `bootstrap`, `max_features`, and `bootstrap_features`. Feature subsets are commonly sampled without replacement.

> **Subspaces, not arbitrary projections:** Selecting feature columns creates coordinate subspaces. It is not the same operation as multiplying the data by a general random projection matrix.

Additional randomization may increase bias while reducing correlation and variance. Its effect should be evaluated for the particular model and dataset.

## Random forests

A **random forest** combines many decision trees with randomized training. The usual construction uses bootstrap samples and can restrict the candidate features considered at each node.

- `RandomForestClassifier` averages the trees' class probabilities before choosing a class.
- `RandomForestRegressor` averages their numeric predictions.
- With bootstrap sampling and `max_samples=None`, each tree draws as many instances as there are in the training set.
- Randomizing the candidate features at each split can prevent the same dominant features from making all the trees too similar.

### Classifier and regressor defaults differ

The feature-sampling default is not the same for both tasks:

- [`RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) defaults to `max_features="sqrt"`, considering roughly $\sqrt{n}$ features per split when there are $n$ input features.
- [`RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) defaults to `max_features=1.0`, making all features eligible. Smaller values introduce more feature-sampling randomness.

These features are sampled during node-split searches, rather than selecting just one fixed feature subset for each entire tree.

### Extra-trees

**Extremely randomized trees**, or **extra-trees**, add randomness to the split thresholds themselves. Instead of fully optimizing the threshold of each candidate feature, they evaluate randomized threshold candidates and choose among them.

- Use `ExtraTreesClassifier` or `ExtraTreesRegressor`.
- Their default is `bootstrap=False`: each tree uses the full training data, while its splits introduce randomness.
- The extra randomization can reduce correlation and sometimes speed up training, with a possible bias trade-off. Compare the result with a random forest rather than assuming it must be faster or more accurate.

## Impurity-based feature importance

Tree ensembles can summarize how much each feature contributes to reductions in training impurity. This is the idea behind impurity-based `feature_importances_`.

For a node $v$, let $I(v)$ be its impurity and $N_v$ the number of training instances reaching it. Its weighted impurity reduction is:

$$
\Delta I(v)
=\frac{N_v}{N}I(v)
-\frac{N_{\text{left}}}{N}I(\text{left})
-\frac{N_{\text{right}}}{N}I(\text{right}),
$$

where $N$ is the training-instance count at that tree's root. With sample weights, replace counts with the corresponding weighted totals, including the root's total weight.

For feature $f$ in tree $t$, sum the reductions at nodes that split on that feature:

$$
q_t(f)=\sum_{\substack{v\in t\\\operatorname{feature}(v)=f}}\Delta I(v).
$$

Normalize a nonzero tree-importance vector, then average the tree importances across the forest:

$$
v_t(f)=\frac{q_t(f)}{\sum_{g=1}^{n}q_t(g)},
\qquad
\operatorname{Importance}(f)=\frac{1}{T}\sum_{t=1}^{T}v_t(f).
$$

The resulting nonzero ensemble vector is normalized so its entries sum to one. Trees with no splits require separate handling because their total reduction is zero.

> **Importance is not causation:** These scores describe the fitted model's training impurity reductions. Features with many candidate split points can be favored, and correlated features can share credit. Validate any feature-selection decision; held-out permutation importance provides another view of predictive usefulness.

## Boosting

**Boosting** builds an ensemble sequentially. Each new learner is trained using information about the current ensemble's mistakes.

- AdaBoost changes the importance of training instances and weights the learners' votes.
- Gradient boosting fits new learners to negative loss gradients, which are residuals for squared-error regression.

Both can use shallow trees, but they modify the training problem in different ways.

## AdaBoost

AdaBoost often uses **decision stumps**: trees with one split node and two leaves. A stump is simple, but a weighted collection of stumps can represent a more complex decision rule.

The three main ideas are:

1. Fit a learner using the current training-instance weights.
2. Give that learner a voting weight based on its weighted error.
3. Increase the relative importance of misclassified instances for the next round.

![AdaBoost repeatedly reweighting training instances, with handwritten annotations identifying examples receiving more or less weight.](/images/notes/hands-on-ml/ensemble-learning/adaboost-reweighting.png)

### Initial weights and weighted error

Initialize the $m$ sample weights uniformly:

$$
w_{i,1}=\frac{1}{m},\qquad \sum_{i=1}^{m}w_{i,1}=1.
$$

At round $t$, train classifier $h_t$ with the current weights. If the learner is a decision stump, it uses weighted data in its own split criterion, such as Gini impurity. AdaBoost then evaluates its weighted classification error separately:

$$
r_t=\sum_{i=1}^{m}w_{i,t}\,
\mathbb{1}\left[h_t(\mathbf{x}_i)\neq y_i\right].
$$

### The learner's amount of say

Using the binary, SAMME-style convention, the learner's voting weight is:

$$
\alpha_t=\eta\log\left(\frac{1-r_t}{r_t}\right),
$$

where $\eta>0$ is the learning rate and $0<r_t<1$.

- A small error gives a larger positive voting weight.
- At $r_t=0.5$, the theoretical voting weight is zero for binary classification.
- An error greater than $0.5$ makes the expression negative. Standard implementations do not simply keep accepting worse-than-chance learners; they apply stopping or rejection rules.
- A perfect learner has $r_t=0$, so it is handled as a stopping case rather than evaluating the logarithm literally.

![Learner voting weight plotted against weighted error, with the original handwritten zero crossing at error one-half and annotations for amount of say.](/images/notes/hands-on-ml/ensemble-learning/adaboost-voting-weight.png)

### Updating the training-instance weights

Increase the unnormalized weight of each misclassified instance:

$$
\widetilde{w}_{i,t+1}
=w_{i,t}\exp\left(
\alpha_t\mathbb{1}\left[h_t(\mathbf{x}_i)\neq y_i\right]
\right).
$$

Normalize the weights before the next round:

$$
w_{i,t+1}
=\frac{\widetilde{w}_{i,t+1}}
{\sum_{j=1}^{m}\widetilde{w}_{j,t+1}}.
$$

Correct predictions leave the unnormalized weights unchanged, but their relative weights decrease after normalization. Misclassified examples become more influential.

![The exponential relationship between a learner's amount of say and the multiplier applied to its misclassified instances.](/images/notes/hands-on-ml/ensemble-learning/adaboost-error-emphasis.png)

> **Keep conventions consistent:** Some binary AdaBoost derivations use half the log ratio and symmetric weight updates for correct and incorrect predictions. The formulas here use the full log ratio with the asymmetric update above. Mixing those conventions changes the update.

### Combining learners and extending to multiple classes

The final prediction is the class receiving the greatest total voting weight:

$$
\hat{y}(\mathbf{x})
=\operatorname*{argmax}_k
\sum_{t=1}^{T}\alpha_t\mathbb{1}\left[h_t(\mathbf{x})=k\right].
$$

For $K$ classes, **SAMME** adjusts the learner weight to:

$$
\alpha_t=\eta\left[
\log\left(\frac{1-r_t}{r_t}\right)+\log(K-1)
\right].
$$

A useful learner must beat the corresponding chance-error threshold:

$$
r_t<1-\frac{1}{K}.
$$

The extra $\log(K-1)$ term shifts the zero-weight threshold from binary error $0.5$ to the multiclass chance baseline. With $K=2$, it vanishes.

Scikit-learn provides [`AdaBoostClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html), with decision stumps as its default base learners. Its main controls include the number of estimators, learning rate, and base-learner complexity. For regression, `AdaBoostRegressor` uses AdaBoost.R2 rather than the classification formulas above.

## Gradient boosting

Gradient boosting also adds predictors sequentially, but trains them to improve the ensemble's current predictions in the direction of lower loss.

### The residual-fitting intuition

For squared-error regression, begin with a baseline prediction $F_0(\mathbf{x})$. Fit a tree to the residuals $y_i-F_0(\mathbf{x}_i)$, then add its prediction with learning rate $\eta$:

$$
F_1(\mathbf{x})=F_0(\mathbf{x})+\eta T_1(\mathbf{x}).
$$

The next tree fits residuals relative to the updated ensemble:

$$
T_2(\mathbf{x}_i)\approx y_i-F_1(\mathbf{x}_i),
\qquad
F_2(\mathbf{x})=F_1(\mathbf{x})+\eta T_2(\mathbf{x}).
$$

Continue building corrections, using validation behavior to decide when further trees stop helping.

![Three stages of gradient boosting, with residuals and fitted trees on the left and cumulative ensemble predictions on the right.](/images/notes/hands-on-ml/ensemble-learning/gradient-boosting-stages.png)

### Why these are called pseudo-residuals

For squared error:

$$
L(y,F)=\frac{1}{2}(y-F)^2,
\qquad
-\frac{\partial L}{\partial F}=y-F.
$$

The negative gradient equals the ordinary residual. For a general differentiable loss, the negative gradient is called a **pseudo-residual**; it need not equal $y-F$.

Instead of changing a fixed parameter vector as in ordinary gradient descent, gradient boosting fits a new function that moves the predictions along this negative-gradient direction: gradient descent in function space.

### The stagewise algorithm

First choose a constant model minimizing the training loss:

$$
F_0(\mathbf{x})=\operatorname*{argmin}_{c}
\sum_{i=1}^{m}L(y_i,c).
$$

- For squared error, $F_0=\bar{y}$, the mean target.
- For binary logistic loss, $F_0$ is a raw log-odds score. If $0<\bar{y}<1$:

$$
F_0=\log\left(\frac{\bar{y}}{1-\bar{y}}\right).
$$

At stage $t$, calculate the negative gradients at the current ensemble predictions:

$$
r_{i,t}
=-\left.
\frac{\partial L(y_i,F)}{\partial F}
\right|_{F=F_{t-1}(\mathbf{x}_i)}.
$$

Then:

1. Fit a shallow regression tree to the targets $r_{i,t}$ using the original features.
2. Let $R_{j,t}$ denote its leaf regions.
3. Choose each leaf's correction by minimizing the original loss in that region:

$$
\gamma_{j,t}
=\operatorname*{argmin}_{\gamma}
\sum_{\mathbf{x}_i\in R_{j,t}}
L\left(y_i,F_{t-1}(\mathbf{x}_i)+\gamma\right).
$$

4. Add the scaled leaf corrections to the ensemble:

$$
F_t(\mathbf{x})
=F_{t-1}(\mathbf{x})
+\eta\sum_j\gamma_{j,t}\mathbb{1}\left[\mathbf{x}\in R_{j,t}\right].
$$

The learning rate shrinks each stage's contribution. It works together with the number and complexity of the trees; it does not remove the need to monitor overfitting.

### Regression and classification leaf corrections

For squared error, the pseudo-residual is $r_{i,t}=y_i-F_{t-1}(\mathbf{x}_i)$ and the optimal leaf correction is its mean:

$$
\gamma_{j,t}
=\frac{1}{|R_{j,t}|}
\sum_{\mathbf{x}_i\in R_{j,t}}
\left[y_i-F_{t-1}(\mathbf{x}_i)\right],
$$

where $|R_{j,t}|$ is the number of training instances in that region.

For binary classification, distinguish the raw score from the predicted probability:

$$
p_i=\sigma\left(F_{t-1}(\mathbf{x}_i)\right),
\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

Writing logistic loss as a function of that raw score gives:

$$
L(y,F)=\log(1+e^F)-yF,
\qquad
r_{i,t}=y_i-p_i.
$$

A commonly used Newton approximation to the leaf correction is:

$$
\gamma_{j,t}\approx
\frac{\sum_{\mathbf{x}_i\in R_{j,t}}(y_i-p_i)}
{\sum_{\mathbf{x}_i\in R_{j,t}}p_i(1-p_i)}.
$$

> **Scores are not probabilities:** This correction updates a log-odds score, not a probability directly. The ratio is a Newton approximation, not a general closed-form solution of the leaf's exact logistic-loss minimization problem.

Apply a sigmoid to the final binary score to obtain a probability. Common multiclass implementations maintain class-specific scores and fit class-specific trees at each stage, then use softmax to obtain probabilities.

The [Gradient Boosting companion note](/notes/hands-on-ml/chapter-06/gradient-boost/) develops these residual and leaf-value calculations in more detail. Implementations include scikit-learn's gradient-boosting estimators and specialized libraries such as XGBoost, LightGBM, and CatBoost.

### Histogram-based gradient boosting

Histogram-based methods group continuous feature values into bins before searching for splits. This can make tree construction substantially cheaper on large tabular datasets. They retain the sequential loss-correction idea; the split-search implementation changes.

## Stacking

**Stacking** learns how to combine base-model predictions instead of using a fixed voting or averaging rule. A final estimator, also called a **meta-learner** or **blender**, takes those predictions as inputs.

![Several base predictors feeding their numeric predictions to a learned blender, with the original handwritten meta-learner annotation.](/images/notes/hands-on-ml/ensemble-learning/stacking-prediction.png)

### Training the meta-learner

Create its training features using **out-of-fold predictions**:

1. Split the training data into cross-validation folds.
2. For each base-model type, fit on the other folds and predict the held-out fold.
3. Collect these held-out predictions into a feature matrix, aligned with the original targets.
4. Train the meta-learner on that matrix.
5. Refit the base models on the full training set for predictions on new data.

![Stacking training diagram showing cross-validation predictions becoming the blender's training inputs while retaining the original targets.](/images/notes/hands-on-ml/ensemble-learning/stacking-training.png)

> **Avoid training-prediction leakage:** Each meta-training prediction should come from a base model that did not train on that instance. Feeding the blender only in-sample predictions can make the base models appear unrealistically accurate.

Scikit-learn provides [`StackingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) and `StackingRegressor`. The final estimator uses cross-validated predictions, while the fitted base estimators used at prediction time are trained on the full training data.

### Multiple stacking layers

It is possible to train several meta-learners as another layer, then combine them with a further learner. Each layer needs an appropriate held-out prediction scheme, and the additional layers increase computation and opportunities to overfit.

![A multilayer stacking ensemble with three levels of predictors and a final combined prediction.](/images/notes/hands-on-ml/ensemble-learning/stacking-layers.png)

## Comparing the methods

The following table is a readable summary of the chapter. These are starting points for experiments, not rules assigning one algorithm to each application.

| Method | Main idea | Main consideration |
| --- | --- | --- |
| Hard voting | Combine class labels from diverse predictors | Discards probability information; correlated errors limit gains |
| Soft voting | Average class probabilities | Depends on informative, well-calibrated probabilities |
| Bagging | Fit models on bootstrap samples | Primarily reduces variance; adds models and inference cost |
| Pasting | Fit models on samples drawn without replacement | Subset size and overlap still affect diversity |
| Random forest | Combine randomized decision trees | Tune tree complexity, feature sampling, and ensemble size |
| Extra-trees | Add randomized split thresholds | Changes the speed, bias, and variance trade-off |
| AdaBoost | Reweight training examples and learner votes | Difficult or mislabeled examples can become influential |
| Gradient boosting | Add learners along negative loss gradients | Tune learning rate, tree complexity, and stopping together |
| Histogram-based boosting | Search splits over binned feature values | Useful for reducing split-search work on larger datasets |
| Stacking | Learn an aggregation model | Requires properly held-out meta-training predictions |

### Original reference tables

The original table images are preserved below, including their example application areas. Treat those examples as illustrations rather than evidence that a method is always suitable for a particular domain.

![Original reference table comparing hard voting, soft voting, bagging, pasting, random forests, and extra-trees, with example use cases.](/images/notes/hands-on-ml/ensemble-learning/ensemble-reference-1.png)

![Continuation of the original reference table covering AdaBoost, gradient boosting, histogram-based boosting, and stacking.](/images/notes/hands-on-ml/ensemble-learning/ensemble-reference-2.png)
