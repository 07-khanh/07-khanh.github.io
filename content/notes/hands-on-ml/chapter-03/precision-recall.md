---
title: "The Precision/Recall Tradeoff: Decision Thresholds and Geometry"
description: "Understanding how decision thresholds affect false positives and false negatives geometrically."
---

When evaluating a binary classifier, standard accuracy is almost always misleading on imbalanced datasets. Instead, we dissect predictions through the lens of precision and recall.

### The Mathematical Formulation

Let $\hat{y} \in \{0, 1\}$ be the predicted class and $y \in \{0, 1\}$ be the true label. The decision function assigns class $1$ if the score $s(x) \ge \tau$:

$$ \text{Precision} = \frac{TP}{TP + FP} = \frac{\sum \mathbb{I}(\hat{y}=1, y=1)}{\sum \mathbb{I}(\hat{y}=1)} $$

$$ \text{Recall} = \frac{TP}{TP + FN} = \frac{\sum \mathbb{I}(\hat{y}=1, y=1)}{\sum \mathbb{I}(y=1)} $$

Raising the threshold $\tau$ shifts the decision boundary rightward. As a result, false positives plummet (so precision increases), but marginal true positives are excluded (so recall decreases).

### Intuition & Code Experiment

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

def compute_threshold_metrics(y_true, scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    
    return thresholds[best_idx], f1_scores[best_idx]
