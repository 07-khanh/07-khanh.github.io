---
title: "Training Linear Models"
date: "2026-08-16"
chapter: 4
chapterTitle: "Training Linear Models"
bookSlug: "hands-on-ml"
tags: ["machine-learning", "linear-regression", "gradient-descent", "regularization", "logistic-regression", "scikit-learn"]
description: "My personal handwritten notes covering linear regression, gradient descent, polynomial regression, learning curves, regularized linear models, and logistic/softmax regression."
---

### LINEAR REGRESSION

* A linear model makes a prediction by computing a weighted sum of input features, plus a constant called the bias term (intercept term).

$$ \hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n $$

* The formula can be written in vectorized form:

$$ \hat{y} = h_\theta(\mathbf{x}) = \theta \cdot \mathbf{x} $$

where $\hat{y}$ is a scalar value and $\theta \cdot \mathbf{x}$ is the dot product.

Note: In ML, vectors are often written as column vectors. If $\vec{\theta}$ & $\vec{x}$ are column vectors (with $x_0 = 1$ as the dummy feature), then the prediction is:

$$ \hat{y} = \theta^T x $$

This is a single-cell matrix resulting from a matrix multiplication. In matrix form for the whole dataset:

$$ \hat{y} = X\theta $$

where $X \in \mathbb{R}^{m \times (n+1)}$, $\theta \in \mathbb{R}^{n+1}$, $\hat{y} \in \mathbb{R}^m$. In this note, I will use this notation.

* To train, we can use RMSE, or rather MSE for simplicity, as it gives the same result.

$$ \text{MSE}(X, y, h_\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( \theta^T x^{(i)} - y^{(i)} \right)^2 = \frac{1}{m} \| X\theta - y \|^2 $$

Note:
* Often, we use a different loss function during training than the performance measure used to evaluate the final model. This is because the one used in training is simpler & has extra terms like regularization.
  * Good performance metric: close to the business objective.
  * Good training loss: easy to optimize & strongly correlated with the metric.

#### ① The Normal Equation

* The normal equation is a closed-form solution to find the value of $\theta$ that minimizes the MSE.

$$ \hat{\theta} = (X^T X)^{-1} X^T y $$

where $\hat{\theta}$ is the value of $\theta$ that minimizes the cost function, and $y$ is the vector of target values containing $y^{(1)}$ to $y^{(m)}$.

**Proof:** We define the cost function $J(\theta)$ as the squared $l_2$ norm of the error vector.

$$ J(\theta) = \frac{1}{2} \| X\theta - y \|^2 \quad \text{(differs from MSE by a constant } \frac{2}{m}\text{)} $$

$$ = \frac{1}{2} (X\theta - y)^T (X\theta - y) $$

$$ = \frac{1}{2} (\theta^T X^T - y^T)(X\theta - y) $$

$$ = \frac{1}{2} \left( \theta^T X^T X \theta - \theta^T X^T y - y^T X \theta + y^T y \right) $$

As $\theta^T X^T y$ is a scalar & the transpose of a scalar is equal to itself:

$$ \theta^T X^T y = (\theta^T X^T y)^T = y^T X \theta $$

Combine those two (symmetric) terms:

$$ J(\theta) = \frac{1}{2} \theta^T X^T X \theta - \theta^T X^T y + \frac{1}{2} y^T y $$

To find the minimum, we find $\nabla_\theta J(\theta)$:

$$ \nabla_\theta J(\theta) = X^T X \theta - X^T y $$

Set $\nabla_\theta J(\theta) = 0$:

$$ X^T X \theta = X^T y \implies \theta = (X^T X)^{-1} X^T y \quad \blacksquare $$

* As $X^T X$ is not always invertible, we usually replace the normal equation with:

$$ \hat{\theta} = X^{+} y $$

where $X^{+}$ is the pseudoinverse of $X$, calculated using SVD.

> For more about SVD: [SVD - Single Value Decomposition]

* **Computational Complexity:**
  * Training: slow when the number of features is large.
    * Normal Equation: $\approx O(mn^2 + n^3)$
    * SVD: $\approx O(m \cdot n^2)$
  * Inference: $O(m_{\text{test}} \cdot n) \Rightarrow$ fast.

---

### GRADIENT DESCENT

* Gradient Descent (GD): iteratively finds weights that decrease loss by moving in the direction that reduces the loss the most (opposite of the gradient vector).

* **Challenges:**
  * Choose a good learning rate.
  * Choose a good starting point (especially when there are many local minima).

* Have to ensure all features have a similar scale, or else it will take longer to converge. (With feature scaling, GD's path toward the minimum is fairly direct; without feature scaling, the cost surface is elongated and GD takes a much longer, indirect path.)

#### ① Batch Gradient Descent

$$ \nabla_\theta \text{MSE}(\theta) = \begin{pmatrix} \frac{\partial}{\partial \theta_0} \text{MSE}(\theta) \\ \vdots \\ \frac{\partial}{\partial \theta_n} \text{MSE}(\theta) \end{pmatrix} = \frac{2}{m} X^T (X\theta - y) $$

$\Rightarrow$ Uses the entire training set $\rightarrow$ slow for large sets but scales well with the number of features.

* Gradient Descent update rule:

$$ \theta^{(\text{next step})} = \theta - \eta \nabla_\theta \text{MSE}(\theta) $$

* Can find a good learning rate $\eta$ using grid search... & limit the number of epochs.
* Find a good number of epochs by setting a very large number of epochs but interrupting the algorithm when the gradient vector becomes tiny — its norm $<\ \epsilon$ (tolerance).

#### ② Stochastic Gradient Descent

* SGD picks a random instance in the training set at every step to compute the gradients. $\Rightarrow$ fast but not regular / stable.
* The randomness can sometimes help SGD have a better chance of finding the global minimum than BGD.
* For SGD to settle at the minimum, we gradually reduce the learning rate $\rightarrow$ a learning schedule.

#### ③ Mini-Batch Gradient Descent

* MBGD computes the gradients on small random sets called mini-batches.

(Comparing the three: Stochastic bounces around erratically, Mini-batch is less erratic than SGD but still noisier than Batch, and Batch moves smoothly and directly to the minimum.)

---

### POLYNOMIAL REGRESSION

* Can use polynomial regression to express more complicated relationships.
* Use `sklearn.preprocessing.PolynomialFeatures` to generate a new feature matrix consisting of all polynomial combinations of the features with degree $\le$ a specific degree. Then we apply Linear Regression to the new feature matrix.
  For example, if an input sample is two-dimensional & of the form $[a, b]$, the degree-2 polynomial features are $[1, a, b, a^2, ab, b^2]$ (the leading 1 can be omitted; the cross-term $ab$ helps find the relationship between features).
* Note: Beware of the combinatorial explosion of the number of features.

---

### LEARNING CURVE

* A way to know whether your model is underfitting or overfitting is to look at the learning curve, which are plots of the model's training error & validation error as a function of the training iteration: just evaluate the model at regular intervals during training on both the training set & the validation set, & plot the results.
* If the model cannot be trained incrementally (i.e., if it does not support `partial_fit()` or `warm_start`), then you train it several times on gradually larger subsets of the training set.
* Use `sklearn.model_selection.learning_curve`.

* **Underfitting case:** both curves reach a plateau; they are close & fairly high.
* **Overfitting case:** the error on the training data is lower; there is a gap between the curves.

---

### BIAS / VARIANCE TRADE-OFF

* A model's generalization error can be expressed as the sum of three very different errors:

$$ E\left[(y_0 - h(x_0))^2\right] = \underbrace{\text{Var}(h(x_0)) + \text{Bias}(h(x_0))^2}_{\text{reducible}} + \underbrace{\text{Var}(\epsilon)}_{\text{irreducible}} $$

* **Bias:** due to wrong assumptions, such as assuming the data is linear while it is actually quadratic. A high-bias model is most likely to underfit.
* **Variance:** due to the model's excessive sensitivity to small variations in the training data. A model with many degrees of freedom is likely to have high variance $\rightarrow$ overfit.
* **Irreducible error:** due to the noisiness of the data itself.
* Increasing a model's complexity $\rightarrow$ increases variance, decreases bias.
  Reducing a model's complexity (regularization) $\rightarrow$ increases bias, decreases variance.

Usually, as flexibility increases:
* Train MSE: decreases monotonically.
* Test MSE: U-shaped.

---

### REGULARIZED LINEAR MODELS

* Regularization helps decrease overfitting.
* Regularization can stabilize linear models & make them more accurate. For example, when inputs are close to colinear $\rightarrow$ small differences in the training set can have a big impact on the trained model.
* Usually done by constraining the model's weights.

#### ① Ridge Regression

$$ J(\theta) = \text{MSE}(\theta) + \frac{\alpha}{m} \sum_{i=1}^{n} \theta_i^2 $$

* $\alpha \uparrow \rightarrow$ pulls weights closer to zero.
* The bias term $\theta_0$ is not regularized.
* The regularization term can be written as: $\alpha (\|w\|)^2 / m$.
* Have to scale data before performing regularization.
* Use `sklearn.linear_model.Ridge` or `penalty="l2"` parameter in `SGDRegressor`.
* For correlated features, Ridge shrinks their weights equally.
  For example, suppose $\hat{y} = w_1 x_1 + w_2 x_2$ & $x_1 \approx x_2 \approx x$, then $\hat{y} \approx (w_1 + w_2) x$. Therefore, the model only cares about $w_1 + w_2$.
  Suppose the optimum is when $w_1 + w_2 = 10$. Ridge uses $w_1^2 + w_2^2$, thus it prefers distributing the weights $5^2 + 5^2 = 50$ to $10^2 + 0^2 = 100$ or $8^2 + 2^2 = 68$, as the squared penalty makes large individual coefficients expensive.

*(Figure 4-18: Linear (left) and polynomial (right) models, both with various levels of ridge regularization.)*

#### ② Lasso Regression

$$ J(\theta) = \text{MSE}(\theta) + 2\alpha \sum_{i=1}^{n} |\theta_i| $$

* Uses the $l_1$ norm instead of the $l_2$ norm like in Ridge.
* Lasso can do feature selection, as it can drive the weights of the least important features to zero.

* Ridge: $R(\theta_j) = \lambda \theta_j^2 \Rightarrow \frac{\partial R}{\partial \theta_j} = 2\lambda \theta_j$ (proportional to the weights) $\Rightarrow \theta_j$ can never hit 0.
* Lasso: $R(\theta_j) = \lambda |\theta_j| \Rightarrow \frac{\partial R}{\partial \theta_j} = \lambda \,\text{sign}(\theta_j) = \begin{cases} -\lambda & \text{if } \theta_j < 0 \\ [-\lambda, \lambda] & \text{if } \theta_j = 0 \\ \lambda & \text{if } \theta_j > 0 \end{cases}$ (subgradient vector)

$\Rightarrow$ Using certain algorithms, Lasso can drive some weights close to 0 & make them stuck at 0.

* For correlated features, Lasso tends to keep one & eliminate the others.
  Using the same example as with Ridge, $w_1 + w_2 = 10$. Lasso cares about $|w_1| + |w_2|$, thus $|10| + |0|$ or $|5| + |5|$ or $|7| + |2|$ are all the same. Therefore, Lasso doesn't get a penalty advantage from splitting the weight. This makes sparse solutions such as $(10, 0)$ possible.
  $\Rightarrow$ Lasso encourages sparsity.
  And even though the updates for correlated weights are quite similar, the noise in the data can make some weights go to 0 first, then the one feature left will carry more predicting power when it stands alone (expressed in the MSE term), and if it can overpower the regularization term, it can stay nonzero.
* Note: Lasso may behave erratically when the number of features is greater than the number of training instances, or when several features are strongly correlated. This is because in those cases there are many nearly equivalent sparse solutions, & Lasso just arbitrarily picks one (e.g. $(10,0)$ & $(0,10)$ are the same). Therefore, small changes in the data can change the feature selection $\rightarrow$ unstable.

*(Figure 4-20: Lasso versus ridge regularization — contour plots comparing the $l_1$ penalty vs the $l_2$ penalty and the resulting GD paths.)* To avoid Lasso bouncing around as it approaches the global minimum, we can gradually decrease the learning rate.

* Use `sklearn.linear_model.Lasso` or `penalty="l1"` with `SGDRegressor`.

#### ③ Elastic Net Regression

* Elastic Net's regularization term is a weighted sum of both Ridge's & Lasso's regularization terms.

$$ J(\theta) = \text{MSE}(\theta) + r\left(2\alpha \sum_{i=1}^{n} |\theta_i|\right) + (1-r)\left(\frac{\alpha}{m} \sum_{i=1}^{n} \theta_i^2\right) $$

* Elastic Net tries to get the strength of both Ridge & Lasso:
  * The Ridge component discourages it from arbitrarily picking just one of several highly correlated features.
  * The Lasso component still gives it a corner at zero, so it can still make weights 0.
* Elastic Net can have a grouping effect: for correlated features, it can give a sparse-but-grouped solution.
  For example, suppose $x_1 \approx x_2 \approx x_3$:
  * Ridge can give $(1.7, 1.7, 1.6)$
  * Lasso can give $(5, 5, 0)$
  * Elastic Net can give $(2.5, 2.5, 0)$
* Use `sklearn.linear_model.ElasticNet`.

**! Which regularization method to choose:**
* Always better to have some regularization.
* If you think most features are useful $\rightarrow$ Ridge.
* If you suspect only some features are useful $\rightarrow$ Lasso / Elastic Net.
* In general, Elastic Net is preferred over Lasso, as Lasso can be unstable in the cases mentioned above.

(Geometric intuition: the $l_1$ norm (diamond) is sparsity-inducing, the $l_2$ norm (circle) leads to weight sharing, and $l_1 + l_2$ is a compromise between the two.)

#### ④ Early Stopping

* Another way to regularize an iterative learning algorithm such as GD is to stop training as soon as the validation error reaches a minimum, or when you think it cannot get any better (Mini-Batch GD, SGD).

---

### LOGISTIC REGRESSION

* Logistic Regression estimates the probability that an instance belongs to a specific class. $\rightarrow$ Classification.
* Logistic regression model estimated probability (vectorized form):

$$ \hat{p} = h_\theta(\mathbf{x}) = \sigma(\theta^T x) $$

* The logistic — denoted $\sigma(\cdot)$ — is a sigmoid function that outputs a number between 0 & 1.

$$ \sigma(t) = \frac{1}{1 + e^{-t}} $$

* Make predictions (using a 50% threshold):

$$ \hat{y} = \begin{cases} 0 & \text{if } \hat{p} < 0.5 \\ 1 & \text{if } \hat{p} \ge 0.5 \end{cases} $$

* $t$ is also called logits.
* The sigmoid function is the inverse of the logit function: the logit function maps $p \in [0,1] \rightarrow t \in (-\infty, +\infty)$, and the sigmoid does the opposite.
  * We have odds: the ratio of an event happening vs not happening.

    $$ \text{odds} = \frac{p}{1-p} $$

  * Log-odds (logit function): $[0,1] \rightarrow (-\infty, +\infty)$

    $$ \log\left(\frac{p}{1-p}\right) = \theta^T x = t $$

  * Invert the logit function: $(-\infty, +\infty) \mapsto [0,1]$

    $$ \frac{p}{1-p} = e^t \implies p = e^t(1-p) \implies p(1+e^t) = e^t $$

    $$ \implies p = \frac{e^t}{1+e^t} = \frac{1}{1+e^{-t}} = \sigma(t) $$

**Training & Cost Function**

* Cost function of a single instance:

$$ c(\theta) = \begin{cases} -\log(\hat{p}) & \text{if } y = 1 \\ -\log(1-\hat{p}) & \text{if } y = 0 \end{cases} $$

Makes sense, as when $t \rightarrow 0$, $-\log(t) \rightarrow +\infty$ (and $-\log(t) \rightarrow 0$ as $t \rightarrow 1$).

* Cost function for the entire set (log loss):

$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{p}^{(i)}) + (1-y^{(i)}) \log(1-\hat{p}^{(i)}) \right] $$

$$ = -\frac{1}{m} \left[ y^T \log(\hat{p}) + (1-y)^T \log(1-\hat{p}) \right] $$

* No closed-form solution, but the cost function is convex $\rightarrow$ GD is guaranteed to find the global minimum.

$$ \frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left[ \sigma(\theta^T x^{(i)}) - y^{(i)} \right] x_j^{(i)} $$

$$ \frac{\partial J}{\partial \theta} = \frac{1}{m} X^T \left[ \sigma(X\theta) - y \right] $$

**Decision Boundaries**

* (Illustrated with the Iris dataset: plotting "Not Iris virginica" & "Iris virginica" probability curves against petal width, with the decision boundary at the point where the two curves cross ($\hat{p} = 0.5$). A 2D version plots petal length vs. petal width with probability contour lines and a linear decision boundary separating "Not Iris virginica" from "Iris virginica.")

---

### SOFTMAX REGRESSION

* Softmax Regression extends Logistic Regression to support multi-class classification.
* First, compute the softmax score for class $k$:

$$ s_k(x) = (\theta^{(k)})^T x $$

Each class has its own parameter vector $\theta^{(k)}$. All $\theta^{(k)}$ are stored in matrix $\Theta$.

* Then we have the softmax function:

$$ \hat{p}_k = \sigma(s(x))_k = \frac{e^{s_k(x)}}{\sum_{j=1}^{K} e^{s_j(x)}} $$

where $K$ is the number of classes, and $\sigma(s(x))_k$ is the probability that $x$ belongs to class $k$, given the scores of each class for that instance $s(x)$.

* Predictions:

$$ \hat{y} = \operatorname*{argmax}_k \sigma(s(x))_k = \operatorname*{argmax}_k \big(s_k(x)\big) = \operatorname*{argmax}_k \big((\theta^{(k)})^T x\big) $$

* Cross entropy:

$$ J(\Theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{p}_k^{(i)}) $$

where $y_k^{(i)} \in \{0, 1\}$ is the probability that the $i$-th instance belongs to class $k$.

> Great explanation about entropy & cross-entropy: [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8)

* Cross entropy gradient vector for class $k$:

$$ \nabla_\theta^{(k)} J(\Theta) = \frac{1}{m} \sum_{i=1}^{m} \left( \hat{p}_k^{(i)} - y_k^{(i)} \right) x^{(i)} $$

* `LogisticRegression` uses softmax regression automatically when there are more than 2 classes; it also has regularization controlled by $C$ (lower $C$ = more regularization).

* (Illustrated with the Iris dataset: petal length vs. petal width, showing the three decision regions — Iris setosa, Iris versicolor, Iris virginica — separated by softmax probability contours.)
