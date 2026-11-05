---
title: "Classification"
date: "2026-08-13"
chapter: 3
chapterTitle: "Classification"
bookSlug: "hands-on-ml"
tags: ["machine-learning", "classification", "evaluation", "scikit-learn"]
description: "Confusion matrices, precision and recall, decision thresholds, PR and ROC curves, and multiclass, multilabel, and multioutput classification."
---

These notes cover how to evaluate classification models and how classification extends beyond a single binary prediction. The main questions are: which mistakes does a model make, how does the decision threshold change those mistakes, and which metric best describes the result?

## Confusion matrices

A **confusion matrix** compares the actual labels with the model's predictions. It shows which predictions are correct and which kinds of errors the model makes.

For a binary classifier, let the positive class be the class we want to detect. For example, a handwritten-digit classifier might predict whether an image contains the digit **5**.

![Confusion matrix for detecting the digit 5, with actual labels as rows, predicted labels as columns, and TN, FP, FN, and TP marked.](/images/notes/hands-on-ml/classification/confusion-matrix.png)

*The four outcomes of a binary classification decision.*

### The four outcomes

| Actual class | Predicted negative | Predicted positive |
| --- | --- | --- |
| Negative | True negative (TN) | False positive (FP) |
| Positive | False negative (FN) | True positive (TP) |

- **True positive (TP):** a positive instance correctly classified as positive.
- **True negative (TN):** a negative instance correctly classified as negative.
- **False positive (FP):** a negative instance incorrectly classified as positive; a false alarm, or Type I error.
- **False negative (FN):** a positive instance incorrectly classified as negative; a missed detection, or Type II error.

In scikit-learn, use `sklearn.metrics.confusion_matrix`. For binary labels ordered as negative, then positive, its entries are arranged as shown in the table above.

## Precision and recall

Precision and recall answer different questions about positive predictions.

### Precision: how trustworthy are positive predictions?

**Precision** measures the fraction of predicted positives that are actually positive:

$$
\operatorname{Precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}.
$$

> **Intuition:** When the model predicts positive, how often is it right?

Use `sklearn.metrics.precision_score` to compute precision.

### Recall: how many positives did the model find?

**Recall**, also called **sensitivity** or the **true positive rate**, measures the fraction of actual positives that the model detects:

$$
\operatorname{Recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}.
$$

> **Intuition:** Out of all the actual positive instances, how many did the model catch?

Use `sklearn.metrics.recall_score` to compute recall.

### Combining them with the F1 score

The **F1 score** combines precision and recall using their harmonic mean:

$$
F_1
= \frac{2}{\frac{1}{\operatorname{Precision}} + \frac{1}{\operatorname{Recall}}}
= \frac{2\operatorname{Precision}\operatorname{Recall}}
{\operatorname{Precision} + \operatorname{Recall}}.
$$

Equivalently, in terms of the confusion-matrix counts:

$$
F_1
= \frac{2\mathrm{TP}}{2\mathrm{TP} + \mathrm{FP} + \mathrm{FN}}
= \frac{\mathrm{TP}}{\mathrm{TP} + \frac{\mathrm{FP} + \mathrm{FN}}{2}}.
$$

The harmonic mean is strongly affected by a low value, so a high F1 score requires both precision and recall to be reasonably high. Use `sklearn.metrics.f1_score` to compute it.

> **Choosing a metric:** F1 favors a balance between precision and recall. That balance may not match the application: sometimes false alarms are more costly, while in other cases missed detections matter more.

## The precision-recall trade-off

A classifier often produces a **decision score** before converting that score into a class label. A threshold determines which scores count as positive predictions:

$$
\hat{y} =
\begin{cases}
1, & s(x) \geq t, \\
0, & s(x) < t,
\end{cases}
$$

where $s(x)$ is the score for instance $x$ and $t$ is the decision threshold.

Lowering the threshold usually catches more positives, but also admits more false positives. Raising it makes positive predictions more selective, at the cost of missing more positives.

![Handwritten digits ordered by decision score, showing how three thresholds change which digits are predicted positive and their precision and recall.](/images/notes/hands-on-ml/classification/decision-thresholds.png)

*Moving the threshold changes the set of positive predictions.*

### Lower versus higher thresholds

| Lower threshold | Higher threshold |
| --- | --- |
| Casts a wider net | Makes more conservative predictions |
| Flags more instances as positive | Flags only the highest-scoring instances |
| Usually increases recall | Usually increases precision |
| Can generate more false alarms | Can miss borderline positive instances |

For probability scores, a threshold such as $0.2$ is more permissive than $0.8$. Raw decision scores can use a different scale, as in the plot below.

![Precision and recall plotted against the decision threshold: recall falls as the threshold increases, while precision generally rises but fluctuates.](/images/notes/hands-on-ml/classification/precision-recall-threshold.png)

*Precision and recall as functions of the decision threshold.*

### Why precision can be bumpy

- **Precision is not necessarily monotonic.** Removing some positive predictions can increase or decrease precision, depending on whether the removed instances were true or false positives.
- **Recall cannot increase as the threshold rises** for a fixed set of scores. Its denominator, $\mathrm{TP} + \mathrm{FN}$, is the fixed number of actual positives, while the number of detected positives can only stay the same or decrease.

The recall curve is therefore non-increasing. On a finite dataset, it changes in steps; a plotted line may make those steps look smooth.

## Precision-recall curves

A **precision-recall (PR) curve** plots precision on the vertical axis against recall on the horizontal axis as the decision threshold changes.

![Precision-recall curve with a selected threshold marked and the original handwritten annotation pointing toward the ideal top-right corner.](/images/notes/hands-on-ml/classification/precision-recall-curve.png)

*The original annotation highlights the goal of keeping both precision and recall high.*

- Curves that remain closer to the top-right corner, $(1, 1)$, maintain high precision over a wider range of recall values.
- The marked point represents one threshold; selecting a different threshold moves the operating point along the curve.
- The area under a PR curve, often called **PR-AUC**, summarizes performance across thresholds. Higher values are generally better when comparing models on the same evaluation dataset.
- PR curves are particularly useful when the positive class is rare.

> **PR-AUC versus average precision:** These names are sometimes used loosely. Scikit-learn's `average_precision_score` computes a recall-weighted sum of precision values, not the trapezoidal area under the PR curve. State which summary you report when comparing results. See the [average precision documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

## ROC curves

The **receiver operating characteristic (ROC) curve** also varies the decision threshold, but plots the **true positive rate** against the **false positive rate**.

### True and false positive rates

The true positive rate is the same as recall:

$$
\mathrm{TPR} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}.
$$

The false positive rate measures the fraction of actual negatives incorrectly classified as positive:

$$
\mathrm{FPR} = \frac{\mathrm{FP}}{\mathrm{TN} + \mathrm{FP}}.
$$

The **true negative rate**, also called **specificity**, is:

$$
\mathrm{TNR} = \frac{\mathrm{TN}}{\mathrm{TN} + \mathrm{FP}}.
$$

Therefore:

$$
\mathrm{FPR} = 1 - \mathrm{TNR} = 1 - \operatorname{Specificity}.
$$

A ROC curve thus plots sensitivity against $1 - \operatorname{Specificity}$.

> **Different denominators:** TPR and FPR both count positive predictions, but TPR is measured among actual positives, while FPR is measured among actual negatives.

### Reading the curve

![ROC curves with the perfect classifier at the top-left corner, the random-classifier diagonal, and handwritten threshold annotations at the endpoints.](/images/notes/hands-on-ml/classification/roc-curve.png)

*Better ROC curves move toward the top-left corner, where TPR is high and FPR is low.*

- With a threshold above every score, the classifier predicts nothing as positive: $\mathrm{TPR} = \mathrm{FPR} = 0$.
- With a threshold at or below every score, it predicts everything as positive: $\mathrm{TPR} = \mathrm{FPR} = 1$.
- Lowering the threshold generally catches more positives, but also allows more false positives.

The original annotations use thresholds $1$ and $0$ to illustrate the endpoints for probability-like scores. The exact endpoint thresholds depend on the score range and the rule used for ties.

## ROC-AUC: a summary of ranking quality

The **area under the ROC curve**, or **ROC-AUC**, summarizes how well the model ranks positive instances above negative ones across thresholds.

### A probabilistic interpretation

Choose one positive instance and one negative instance at random. Ignoring ties, ROC-AUC is the probability that the positive instance receives the higher score. With ties given half credit:

$$
\operatorname{ROC\text{-}AUC}
= \Pr\bigl(s(X^+) > s(X^-)\bigr)
+ \frac{1}{2}\Pr\bigl(s(X^+) = s(X^-)\bigr).
$$

Here, $X^+$ and $X^-$ are randomly selected positive and negative instances, respectively.

![Positive instances shown as purple squares and negative instances as orange circles across model scores, with handwritten notes identifying the two groups.](/images/notes/hands-on-ml/classification/roc-auc-ranking.png)

*AUC measures how often a randomly selected positive is ranked above a randomly selected negative.*

For example, a spam classifier with ROC-AUC equal to $1$ ranks every spam email above every legitimate email in the evaluation set. The final positive or negative prediction still depends on the chosen threshold.

### Interpreting AUC values

- **$\operatorname{AUC} = 1$:** perfect ranking on the evaluation set; the classes can be separated by a threshold, corresponding to the ROC point $(0, 1)$.
- **$0.8 \leq \operatorname{AUC} \leq 0.9$:** often described informally as good discrimination, but whether it is sufficient depends on the application and dataset.
- **$\operatorname{AUC} = 0.5$:** no overall ranking advantage over random scoring. Random scoring has an expected ROC curve along the diagonal $y = x$.
- **$\operatorname{AUC} < 0.5$:** ranking is worse than random in the chosen score direction.

> **Ranking is not a threshold choice:** A high AUC does not guarantee that the current threshold gives acceptable precision or recall. Inspect the operating point that the application will actually use.

Compute this summary with [`sklearn.metrics.roc_auc_score`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), using prediction scores rather than already-thresholded class labels.

## Choosing between PR and ROC curves

When the positive class is rare, a PR curve often makes the practical effect of false positives easier to see.

The difference comes from the denominators:

$$
\mathrm{FPR} = \frac{\mathrm{FP}}{\mathrm{TN} + \mathrm{FP}},
\qquad
\operatorname{Precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}.
$$

- FPR is normalized by the total number of actual negatives. When there are many negatives, a small FPR can still correspond to a large number of false alarms.
- Precision directly describes the quality of the positive predictions. If false positives are numerous relative to true positives, precision will be low.
- PR curves are especially useful when positives are rare or when the reliability of positive predictions is central to the task.
- ROC curves remain useful for examining the trade-off between the fraction of positives detected and the fraction of negatives incorrectly flagged.

> **Practical takeaway:** Choose the curve and metric according to the errors that matter. For an imbalanced problem, do not rely on a favorable ROC-AUC alone; also inspect precision, recall, and the number of false positives at the selected threshold.

## Multiclass classification

In **multiclass classification**, each instance belongs to one of more than two classes. A binary classifier can be extended to this setting in two common ways.

### One-versus-the-rest (OvR)

- Train one binary classifier per class: that class versus all remaining classes.
- For $K$ classes, train $K$ classifiers.
- At prediction time, choose the class whose classifier produces the highest decision score.

### One-versus-one (OvO)

- Train one binary classifier for every pair of classes.
- For $K$ classes, the number of classifiers is:

$$
\binom{K}{2} = \frac{K(K - 1)}{2}.
$$

- At prediction time, choose the class that wins the most pairwise comparisons, with a tie-breaking rule if necessary.
- Each binary classifier only needs the training instances belonging to the two classes it distinguishes.

## Multilabel classification

In **multilabel classification**, one instance can receive several binary labels at the same time. Evaluation can focus on complete predictions for each instance, on individual label assignments, or on aggregated per-label metrics.

Let $n$ be the number of instances and $L$ the number of possible labels. Write $y_{ij} \in \{0, 1\}$ for the true value of label $j$ on instance $i$, and $\hat{y}_{ij}$ for its prediction. The indicator $\mathbb{1}[A]$ equals $1$ when condition $A$ is true and $0$ otherwise.

### Subset accuracy

**Subset accuracy** counts an instance as correct only if its entire predicted label vector matches the true vector:

$$
\operatorname{SubsetAccuracy}
= \frac{1}{n}\sum_{i=1}^{n}
\mathbb{1}\bigl[\hat{\mathbf{y}}_i = \mathbf{y}_i\bigr].
$$

This is a strict measure: one incorrect label makes the whole prediction incorrect. It can be particularly harsh when there are many possible labels.

### Hamming loss

**Hamming loss** is the fraction of individual label assignments that are incorrect, across all instances and labels:

$$
\operatorname{HammingLoss}
= \frac{1}{nL}\sum_{i=1}^{n}\sum_{j=1}^{L}
\mathbb{1}\bigl[y_{ij} \neq \hat{y}_{ij}\bigr].
$$

Lower values are better. Unlike subset accuracy, this measure gives partial credit when some labels are correct.

### Jaccard similarity

Let $Y_i$ be the set of true positive labels for instance $i$, and $\hat{Y}_i$ its set of predicted positive labels. Their **Jaccard similarity** is:

$$
J(Y_i, \hat{Y}_i)
= \frac{|Y_i \cap \hat{Y}_i|}{|Y_i \cup \hat{Y}_i|}.
$$

It compares the labels shared by the two sets with all labels present in either set. A score of $1$ means the nonempty sets match exactly. If both sets are empty, the denominator is zero, so an evaluation must specify how that case is handled.

### Label-based evaluation

Another approach is to compute a binary metric, such as F1 or ROC-AUC, for each label and then aggregate the results:

- **Macro averaging:** average the per-label scores with equal weight for each label.
- **Weighted averaging:** weight each label's score by its support, or number of true occurrences.
- **Micro averaging:** pool the relevant counts across labels before computing a count-based metric such as precision, recall, or F1.

### Modeling dependencies with classifier chains

For a model that does not natively support multilabel classification, such as an SVC, one strategy is to train a separate binary classifier per label.

To capture dependencies between labels, arrange the classifiers in a **chain**. At prediction time, each classifier receives:

- The original input features.
- The predictions from classifiers earlier in the chain.

Scikit-learn provides this approach through [`sklearn.multioutput.ClassifierChain`](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html).

## Multioutput classification

**Multioutput classification** generalizes multilabel classification by allowing each output to have more than two possible classes.

- In multilabel classification, each output is binary: a label is present or absent.
- In multiclass-multioutput classification, there are several outputs, and each output can take one of multiple class values.

The distinction is therefore both **how many outputs** are predicted and **how many classes each output can take**.
