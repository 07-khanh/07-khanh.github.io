---
title: "Probability Calibration"
date: "2026-08-21"
chapter: 6
chapterTitle: "Ensemble Learning and Random Forests"
bookSlug: "hands-on-ml"
tags: ["machine-learning", "probability-calibration", "classification", "ensemble-learning", "scikit-learn"]
description: "Reliability diagrams, classwise and confidence calibration, and fitting sigmoid or isotonic calibrators using held-out predictions."
---

**Probability calibration** makes a model's estimated probabilities agree with observed outcome frequencies. If a binary classifier assigns a positive-class probability close to $80\%$ to a group of instances, approximately $80\%$ of that group should actually be positive.

For a perfectly calibrated binary classifier, the ideal relationship is:

$$
\Pr\!\left(Y=1\mid\hat{p}(X)=p\right)=p,
$$

for probability values supported by the model's predictions. Here, $\hat{p}(X)$ is the estimated probability of the positive class.

> **A frequency interpretation:** Calibration describes groups of predictions. It does not guarantee the outcome of any individual prediction.

## When calibration matters

Calibration matters whenever a decision depends on the numerical probability, including:

- **Soft voting:** Averaging probabilities from several classifiers works best when their estimates have meaningful, comparable scales. See [soft voting in the ensemble learning note](/notes/hands-on-ml/chapter-06/ensemble-learning-and-random-forests/#soft-voting).
- **Risk estimation:** A probability may feed into an expected-cost calculation or another decision rule.
- **Custom decision thresholds:** Choosing a threshold such as $0.8$ assumes the estimated probability is meaningful at that level.

Ranking quality and calibration measure different properties. A classifier can order instances well while assigning misleading probabilities. Hard voting uses predicted labels rather than averaging probabilities, but calibrating a model can still change those labels if a fixed probability threshold is applied afterward.

## Reading a reliability diagram

A **calibration curve**, also called a **reliability diagram**, compares predicted probabilities with observed frequencies:

1. Divide predictions into bins according to their predicted probability.
2. For each bin, compute the mean predicted probability.
3. Compute the fraction of instances in that bin that actually belong to the positive class.
4. Plot the mean prediction on the horizontal axis and the observed fraction on the vertical axis.

The diagonal represents perfect agreement. A point below the diagonal means the model overestimates the positive-class probability in that bin; a point above it means the model underestimates it.

![Reliability curves and probability histograms for logistic regression, Naive Bayes, SVC, and random forest, with the original handwritten calibration annotations.](/images/notes/hands-on-ml/probability-calibration/calibration-curves.png)

*The original annotated figure. The top panel compares calibration; the lower panels show where each model's probability estimates are concentrated.*

### Interpreting the illustrated models

- **Logistic regression:** Its curve stays relatively close to the diagonal in this example. Logistic regression can produce well-calibrated probabilities when its assumptions and fit are appropriate.
- **Naive Bayes:** Many predictions concentrate near $0$ and $1$, while the reliability curve indicates overconfidence at the extremes.
- **SVC:** The curve shows underconfidence at the extremes: low probabilities are too high, and high probabilities are too low.
- **Random forest:** Averaging tree predictions can make extreme probabilities harder to reach. This example also shows some underconfidence toward the extremes.

> **Read both panels:** A histogram shows the distribution of predictions, not whether those predictions are correct. Interpret it alongside the reliability curve. These patterns describe the illustrated experiment, rather than guarantees for every fitted model or dataset.

## Calibration for multiclass classification

For $K$ classes, a model produces a probability vector:

$$
\hat{\mathbf{p}}(x)
=\left(\hat{p}_1(x),\ldots,\hat{p}_K(x)\right),
\qquad
\sum_{k=1}^{K}\hat{p}_k(x)=1.
$$

Two useful diagnostic views are classwise calibration and confidence calibration.

### Classwise calibration

Treat each class as a separate one-vs-rest problem. For class $k$, compare $\hat{p}_k(x)$ with the indicator that the true label is $k$:

$$
\Pr\!\left(Y=k\mid\hat{p}_k(X)=p\right)=p.
$$

This gives $K$ reliability curves, allowing inspection of the probability estimates for each class separately.

### Confidence calibration

Focus on the model's predicted label and its highest probability:

$$
\hat{y}(x)=\operatorname*{argmax}_k\hat{p}_k(x),
\qquad
\hat{c}(x)=\max_k\hat{p}_k(x).
$$

The ideal confidence-calibration relationship is:

$$
\Pr\!\left(Y=\hat{y}(X)\mid\hat{c}(X)=c\right)=c.
$$

For example, among predictions made with confidence close to $0.9$, approximately $90\%$ should have the correct predicted label. This produces one curve combining predictions across classes; it does not establish that each class's probabilities are calibrated.

## Fitting a calibration mapping

A calibrator learns a function $g$ that maps a model's score $f(x)$ to a probability:

$$
\hat{p}_{\mathrm{cal}}(x)=g(f(x)),
\qquad
g(f(x))\approx\Pr\!\left(Y=1\mid f(X)=f(x)\right).
$$

The input score may come from `decision_function()` or from an existing probability estimate. In scikit-learn, [`CalibratedClassifierCV`](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html) uses `decision_function()` when available and otherwise uses `predict_proba()`.

> **Keep calibration data independent:** Fit the calibrator using held-out predictions or out-of-fold predictions. Reusing the base model's in-sample predictions can make the calibration fit misleadingly optimistic. Evaluate the resulting probabilities on separate data.

### Sigmoid calibration

Sigmoid calibration, often called **Platt scaling**, fits a logistic mapping:

$$
g(f)=\frac{1}{1+\exp(Af+B)},
$$

where $A$ and $B$ are learned from the calibration scores and their true labels.

Because it fits only two parameters, it is a useful candidate when calibration data is limited. Its fixed shape may be too restrictive when the required correction is more irregular. In `CalibratedClassifierCV`, select it with `method="sigmoid"`.

### Isotonic regression

Isotonic regression fits a flexible **non-decreasing** mapping: a higher score cannot receive a lower calibrated probability.

After sorting calibration scores so that $f_{(1)}\le\cdots\le f_{(m)}$, the fitted values solve:

$$
\min_{0\le q_1\le\cdots\le q_m\le1}
\sum_{i=1}^{m}\left(y_{(i)}-q_i\right)^2,
$$

where $y_{(i)}\in\{0,1\}$ is the true label paired with sorted score $f_{(i)}$, and $q_i$ is its fitted probability. Equal input scores must share the same fitted value.

The **Pool Adjacent Violators (PAV)** algorithm builds this monotone fit:

1. Start with the observations ordered by their model scores.
2. Check neighboring blocks for a decrease in fitted probability.
3. Merge blocks that violate the non-decreasing constraint.
4. Assign each merged block its observed positive fraction, repeating until no violations remain.

The resulting mapping can correct distortions that a sigmoid cannot capture. Its flexibility also makes it more prone to overfitting small calibration datasets, so it is most useful with enough representative calibration data. Select it with `method="isotonic"`.

## Further reading

The [scikit-learn probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html), linked in the original notes, explains reliability diagrams and calibration methods in more detail.
