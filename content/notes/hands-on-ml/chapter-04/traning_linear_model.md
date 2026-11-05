---
title: "Training Linear Models"
date: "2026-08-16"
chapter: 4
chapterTitle: "Training Linear Models"
bookSlug: "hands-on-ml"
tags: ["machine-learning", "linear-regression", "gradient-descent", "regularization", "logistic-regression", "scikit-learn"]
description: "Linear regression and the normal equation, gradient descent, polynomial features, learning curves, regularization, and logistic and softmax regression."
---

These notes follow the progression from fitting a linear model to controlling its complexity, then adapting linear scores for classification. The emphasis is on the mathematics behind the training objectives and on interpreting the accompanying diagrams.

## Linear regression

A linear model predicts a target by taking a weighted sum of the input features and adding an intercept, also called the bias term:

$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n.
$$

### Vector and matrix notation

Include a dummy feature $x_0 = 1$ to absorb the intercept into the parameter vector. With column vectors:

$$
\mathbf{x} = \begin{pmatrix}1 & x_1 & \cdots & x_n\end{pmatrix}^{\mathsf T},
\qquad
\hat{y} = h_{\boldsymbol{\theta}}(\mathbf{x})
= \boldsymbol{\theta}^{\mathsf T}\mathbf{x}.
$$

The prediction for one instance is a scalar. Stack the feature vectors as rows of the design matrix $X$ to predict an entire dataset:

$$
\hat{\mathbf{y}} = X\boldsymbol{\theta},
\qquad
X \in \mathbb{R}^{m\times(n+1)},
\quad
\boldsymbol{\theta} \in \mathbb{R}^{n+1},
\quad
\hat{\mathbf{y}} \in \mathbb{R}^{m}.
$$

Here, $m$ is the number of training instances and $n$ is the number of input features, excluding the dummy feature.

### Training with mean squared error

The mean squared error (MSE) is:

$$
\operatorname{MSE}(\boldsymbol{\theta})
= \frac{1}{m}\sum_{i=1}^{m}
\left(\boldsymbol{\theta}^{\mathsf T}\mathbf{x}^{(i)} - y^{(i)}\right)^2
= \frac{1}{m}\left\|X\boldsymbol{\theta} - \mathbf{y}\right\|_2^2.
$$

Minimizing MSE gives the same parameter minimizers as minimizing RMSE, because the square root is increasing. MSE is simpler to differentiate.

> **Training loss versus evaluation metric:** The evaluation metric should reflect the application objective. The training loss should be practical to optimize and aligned with that metric; it may also include regularization terms that are absent from the reported evaluation score.

## The normal equation

When $X^{\mathsf T}X$ is invertible, a closed-form least-squares solution is:

$$
\hat{\boldsymbol{\theta}}
= \left(X^{\mathsf T}X\right)^{-1}X^{\mathsf T}\mathbf{y}.
$$

### Deriving the solution

Define an equivalent objective:

$$
J(\boldsymbol{\theta})
= \frac{1}{2}\left\|X\boldsymbol{\theta}-\mathbf{y}\right\|_2^2.
$$

Since $\operatorname{MSE} = \frac{2}{m}J$, the positive constant factor does not change the minimizers. Expanding the quadratic form gives:

$$
\begin{aligned}
J(\boldsymbol{\theta})
&= \frac{1}{2}(X\boldsymbol{\theta}-\mathbf{y})^{\mathsf T}
(X\boldsymbol{\theta}-\mathbf{y}) \\
&= \frac{1}{2}\left(
\boldsymbol{\theta}^{\mathsf T}X^{\mathsf T}X\boldsymbol{\theta}
- \boldsymbol{\theta}^{\mathsf T}X^{\mathsf T}\mathbf{y}
- \mathbf{y}^{\mathsf T}X\boldsymbol{\theta}
+ \mathbf{y}^{\mathsf T}\mathbf{y}\right).
\end{aligned}
$$

The two cross terms are equal because they are transposes of the same scalar:

$$
\boldsymbol{\theta}^{\mathsf T}X^{\mathsf T}\mathbf{y}
= \left(\boldsymbol{\theta}^{\mathsf T}X^{\mathsf T}\mathbf{y}\right)^{\mathsf T}
= \mathbf{y}^{\mathsf T}X\boldsymbol{\theta}.
$$

Therefore:

$$
J(\boldsymbol{\theta})
= \frac{1}{2}\boldsymbol{\theta}^{\mathsf T}X^{\mathsf T}X\boldsymbol{\theta}
- \boldsymbol{\theta}^{\mathsf T}X^{\mathsf T}\mathbf{y}
+ \frac{1}{2}\mathbf{y}^{\mathsf T}\mathbf{y}.
$$

Taking the gradient and setting it to zero yields the normal equations:

$$
\nabla_{\boldsymbol{\theta}}J
= X^{\mathsf T}X\boldsymbol{\theta} - X^{\mathsf T}\mathbf{y} = \mathbf{0},
$$

$$
X^{\mathsf T}X\hat{\boldsymbol{\theta}} = X^{\mathsf T}\mathbf{y}.
$$

If the matrix is invertible, multiplying by its inverse gives the solution above.

### Pseudoinverse and SVD

$X^{\mathsf T}X$ is not always invertible. A more general solution uses the Moore-Penrose pseudoinverse:

$$
\hat{\boldsymbol{\theta}} = X^+\mathbf{y}.
$$

The pseudoinverse can be computed using **Singular Value Decomposition (SVD)**. It provides a least-squares solution even when the ordinary inverse does not exist; when several solutions minimize the error, it selects the one with minimum Euclidean norm.

> **Numerical practice:** The inverse formula explains the mathematics. In code, prefer a least-squares solver based on an appropriate matrix factorization rather than explicitly computing the inverse.

Let $d=n+1$ be the number of parameters. For a dense matrix with $m\geq d$:

- Forming and solving the normal equations costs roughly $O(md^2+d^3)$.
- A thin SVD costs roughly $O(md^2)$.
- Predicting $m_{\text{test}}$ instances costs $O(m_{\text{test}}d)$.

These direct training methods become expensive when the number of features is large, motivating iterative optimization.

## Gradient descent

**Gradient descent (GD)** iteratively adjusts the parameters in the direction opposite the gradient. Locally, this is the direction of steepest decrease under the usual Euclidean geometry.

![Gradient descent taking successive steps down a cost curve toward a minimum.](/images/notes/hands-on-ml/training-linear-models/gradient-descent.png)

The main choices are the learning rate, the stopping criterion, and, for nonconvex objectives, the initial parameters.

### Feature scaling and the learning rate

Features on very different scales can produce elongated cost contours. Gradient descent then zigzags through a narrow valley rather than moving directly toward the minimum.

![Cost contours and gradient-descent paths with feature scaling on the left and without scaling on the right.](/images/notes/hands-on-ml/training-linear-models/feature-scaling.png)

- A learning rate that is too small makes progress slow.
- A learning rate that is too large can overshoot or cause divergence.
- Scaling the features often makes convergence faster and learning-rate selection easier.

> **Convex versus nonconvex objectives:** Ordinary linear regression with MSE is convex, so it has no spurious local minima. Remarks about escaping local minima apply to nonconvex models, not to this least-squares objective.

### Batch gradient descent

Batch GD uses the entire training set to compute each gradient:

$$
\nabla_{\boldsymbol{\theta}}\operatorname{MSE}
= \begin{pmatrix}
\frac{\partial\operatorname{MSE}}{\partial\theta_0} \\
\vdots \\
\frac{\partial\operatorname{MSE}}{\partial\theta_n}
\end{pmatrix}
= \frac{2}{m}X^{\mathsf T}(X\boldsymbol{\theta}-\mathbf{y}).
$$

The update rule is:

$$
\boldsymbol{\theta}_{t+1}
= \boldsymbol{\theta}_t
- \eta_t\nabla_{\boldsymbol{\theta}}\operatorname{MSE}(\boldsymbol{\theta}_t),
$$

where $\eta_t$ is the learning rate at step $t$.

- Each update can be expensive on a large training set, but avoids forming and inverting a feature-by-feature matrix.
- Tune the learning rate using validation performance and a limited training budget.
- One possible stopping criterion is $\|\nabla J\|_2<\varepsilon$, together with a maximum number of iterations.

### Stochastic gradient descent

**Stochastic gradient descent (SGD)** estimates the gradient using one training instance at a time. Updates are inexpensive but noisy.

![A stochastic gradient-descent path moving irregularly around a cost minimum.](/images/notes/hands-on-ml/training-linear-models/stochastic-gradient-descent.png)

The noise can help optimization explore nonconvex landscapes, but for convex least squares its main benefit is computational efficiency. With a fixed learning rate, SGD may keep moving around the minimum. A **learning schedule** gradually reduces the learning rate to help it settle.

### Mini-batch gradient descent

**Mini-batch GD** estimates gradients from small batches of training instances, usually after shuffling the data. It balances inexpensive updates with less noise than single-instance SGD, and can take advantage of efficient matrix operations.

![Parameter trajectories for stochastic, mini-batch, and batch gradient descent, showing progressively smoother paths.](/images/notes/hands-on-ml/training-linear-models/gradient-descent-comparison.png)

*Stochastic updates are the noisiest, mini-batch updates are intermediate, and full-batch updates are the smoothest in this example.*

## Polynomial regression

A linear model can represent nonlinear relationships by using transformed features. **Polynomial regression** first constructs polynomial combinations of the inputs, then fits a linear model to those features.

For two input features $a$ and $b$, a degree-two expansion is:

$$
\phi(a,b) = \begin{pmatrix}1 & a & b & a^2 & ab & b^2\end{pmatrix}.
$$

- The constant feature can be omitted when the estimator already fits an intercept.
- The interaction term $ab$ lets the model represent relationships involving both inputs.
- Use `sklearn.preprocessing.PolynomialFeatures`, followed by a linear regression estimator.

> **Still linear in the parameters:** The prediction can be nonlinear in the original inputs while remaining linear in the learned coefficients. Watch for the combinatorial growth in the number of generated features as the degree and input dimension increase.

## Learning curves

Learning curves compare training and validation errors to diagnose underfitting and overfitting. Two related plots answer different questions:

- **Error versus training-set size:** refit on progressively larger subsets to see whether more data helps. This is the plot produced by `sklearn.model_selection.learning_curve`.
- **Error versus training iteration:** evaluate periodically during iterative training to see whether further optimization helps or starts to overfit.

For estimators that support incremental training, `partial_fit()` or a suitable `warm_start=True` setting can reuse learned parameters. Otherwise, fitting on several training subsets requires retraining the model.

### Underfitting

![Training and validation RMSE becoming close and plateauing at a relatively high error as the training-set size grows.](/images/notes/hands-on-ml/training-linear-models/learning-curve-underfitting.png)

When both curves plateau at similar, relatively high errors, the model may be too simple to capture the relationship. More data alone is unlikely to resolve substantial model bias.

### Overfitting

![Training RMSE below validation RMSE, with a persistent gap as the training-set size grows.](/images/notes/hands-on-ml/training-linear-models/learning-curve-overfitting.png)

A gap between low training error and higher validation error suggests overfitting. Consider more training data, a simpler model, or stronger regularization; compare the effect using validation data.

## The bias-variance trade-off

Under squared-error loss, the expected prediction error can be decomposed into squared bias, variance, and irreducible noise. For a fixed input $x_0$, with $y_0=f(x_0)+\varepsilon$ and independent, zero-mean noise:

$$
\mathbb{E}\left[(y_0-\hat{f}(x_0))^2\right]
= \underbrace{\left(\mathbb{E}[\hat{f}(x_0)]-f(x_0)\right)^2}_{\text{squared bias}}
+ \underbrace{\operatorname{Var}(\hat{f}(x_0))}_{\text{variance}}
+ \underbrace{\operatorname{Var}(\varepsilon)}_{\text{irreducible noise}}.
$$

- **Bias:** error from restrictive assumptions, such as fitting a line to a quadratic relationship. High bias is associated with underfitting.
- **Variance:** sensitivity to changes in the sampled training set. A highly flexible model may fit those changes and overfit.
- **Irreducible error:** randomness in the target that cannot be removed by changing the fitted model alone.

![Target diagrams illustrating high-bias clustered predictions and lower-bias predictions with increasing variance.](/images/notes/hands-on-ml/training-linear-models/bias-variance-targets.png)

Increasing model flexibility often reduces bias and increases variance. Stronger regularization often has the opposite effect.

![Models with different flexibility and their training and test MSE, with the original handwritten train and test labels.](/images/notes/hands-on-ml/training-linear-models/model-flexibility.png)

*Training error typically decreases as flexibility increases. Test error can first decrease and then increase; the U-shaped pattern is a useful intuition rather than a universal law.*

## Regularized linear models

Regularization constrains the coefficients to reduce overfitting and can stabilize a model when features are nearly collinear. Let $\mathbf{w}=(\theta_1,\ldots,\theta_n)^{\mathsf T}$ contain the feature weights, excluding the intercept $\theta_0$.

Scale the features before comparing coefficient penalties, and fit the scaler on the training data only. The intercept is usually left unpenalized.

> **Penalty conventions:** Factors such as $m$ and $1/2$ change the numerical meaning of the regularization parameter. The objectives below use the documented conventions of scikit-learn's estimators; equal numerical values of `alpha` are not directly comparable across all estimators.

### Ridge regression

Ridge adds an $\ell_2$ penalty. Dividing its sum-of-squares objective by $m$ gives the equivalent form:

$$
J_{\text{ridge}}(\boldsymbol{\theta})
= \operatorname{MSE}(\boldsymbol{\theta})
+ \frac{\alpha}{m}\|\mathbf{w}\|_2^2
= \operatorname{MSE}(\boldsymbol{\theta})
+ \frac{\alpha}{m}\sum_{j=1}^{n}\theta_j^2.
$$

Larger $\alpha$ puts more pressure on large coefficients. Use [`sklearn.linear_model.Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), or an $\ell_2$ penalty with `SGDRegressor` under that estimator's own scaling convention.

For two identical, equally scaled features, the fit may depend mainly on $w_1+w_2$. If the sum is fixed at $10$, compare:

$$
5^2+5^2=50,\qquad 8^2+2^2=68,\qquad 10^2+0^2=100.
$$

The squared penalty favors sharing the weight in this symmetric example. With highly correlated but nonidentical features, the coefficients need not be exactly equal.

![Linear and polynomial regression fits with different amounts of ridge regularization.](/images/notes/hands-on-ml/training-linear-models/ridge-regression.png)

### Lasso regression

Lasso uses an $\ell_1$ penalty. Its documented objective is:

$$
J_{\text{lasso}}(\boldsymbol{\theta})
= \frac{1}{2m}\|X\boldsymbol{\theta}-\mathbf{y}\|_2^2
+ \alpha\|\mathbf{w}\|_1.
$$

Multiplying the entire objective by $2$ gives the equivalent expression from the handwritten notes:

$$
2J_{\text{lasso}}(\boldsymbol{\theta})
= \operatorname{MSE}(\boldsymbol{\theta})
+ 2\alpha\sum_{j=1}^{n}|\theta_j|.
$$

Unlike ridge, lasso can produce **exactly zero coefficients**, so it can perform feature selection. Use [`sklearn.linear_model.Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html), or an $\ell_1$ penalty with `SGDRegressor`.

For a single weight, the ridge penalty $R(w)=\lambda w^2$ has derivative:

$$
R'(w)=2\lambda w.
$$

Its shrinkage pressure becomes smaller near zero. The lasso penalty $R(w)=\lambda|w|$ has a subgradient:

$$
\partial R(w)=
\begin{cases}
\{-\lambda\}, & w<0, \\
[-\lambda,\lambda], & w=0, \\
\{\lambda\}, & w>0.
\end{cases}
$$

The corner at zero allows sparse solutions. Appropriate algorithms, such as coordinate descent or proximal methods, can set coefficients exactly to zero. Ridge does not generally encourage exact zeros, although a coefficient can be zero in a particular solution.

For correlated features whose nonnegative weights sum to $10$:

$$
|10|+|0|=|5|+|5|=|8|+|2|=10.
$$

There is no $\ell_1$ penalty advantage to spreading the weight. A sparse solution is therefore possible, and small differences in the data can determine which correlated feature is retained. Once one coefficient is zero, the remaining coefficient may carry the useful predictive signal.

This selection can be unstable when many features are correlated or when there are more features than training instances.

![L1 and L2 penalty contours and optimization paths for lasso and ridge, preserving handwritten annotations about bouncing near the lasso minimum.](/images/notes/hands-on-ml/training-linear-models/lasso-ridge-contours.png)

*With a simple fixed-step subgradient method, the lasso path can oscillate near the optimum. Reducing the learning rate or using an optimizer suited to the nonsmooth penalty helps.*

### Elastic net regression

Elastic net combines the two penalties. Using scikit-learn's convention:

$$
J_{\text{elastic}}(\boldsymbol{\theta})
= \frac{1}{2m}\|X\boldsymbol{\theta}-\mathbf{y}\|_2^2
+ \alpha r\|\mathbf{w}\|_1
+ \frac{\alpha(1-r)}{2}\|\mathbf{w}\|_2^2,
$$

where $r\in[0,1]$ is the mixing ratio, called `l1_ratio` in [`sklearn.linear_model.ElasticNet`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html).

- $r=1$ gives the lasso penalty.
- $r=0$ gives an $\ell_2$ penalty, with a different `alpha` scaling from `Ridge`.
- Intermediate values combine sparsity with weight sharing among correlated features.

The $\ell_1$ component retains a corner at zero; the $\ell_2$ component discourages arbitrarily concentrating weight on one of several similar features. Elastic net can therefore retain useful correlated groups while removing other features.

> **Normalization correction:** To write this objective with MSE as its first term, multiply every term by $2$. The result has an $\ell_2$ coefficient of $\alpha(1-r)$, not $\alpha(1-r)/m$. Mixing the ridge estimator's normalization with elastic net's normalization changes the optimization problem.

### Choosing a regularizer

- **Ridge:** a useful candidate when many features contribute and coefficient stability matters.
- **Lasso:** a useful candidate when a sparse model is desired, with caution around correlated features.
- **Elastic net:** a useful candidate when sparsity and retention of correlated groups are both important.

![Geometric comparison of L1, L2, and combined L1 plus L2 constraints: sparsity, weight sharing, and a compromise.](/images/notes/hands-on-ml/training-linear-models/regularization-geometry.png)

Regularization is not automatically beneficial at every strength. Compare candidates, including an unregularized baseline, using validation data rather than assuming one method is always best.

### Early stopping

An iterative learning algorithm can also be regularized by stopping before it starts fitting noise in the training set.

- Monitor validation error during training.
- Keep the parameters from the best validation checkpoint.
- With noisy SGD or mini-batch updates, allow a reasonable period without improvement rather than stopping at the first upward fluctuation.

The aim is to retain the model with the best validation behavior, not necessarily the lowest training error.

## Logistic regression

Despite its name, **logistic regression** is a classification model. It applies a sigmoid to a linear score to estimate the probability of the positive class:

$$
\hat{p}=h_{\boldsymbol{\theta}}(\mathbf{x})
=\sigma(\boldsymbol{\theta}^{\mathsf T}\mathbf{x}),
\qquad
\sigma(t)=\frac{1}{1+e^{-t}}.
$$

![The sigmoid function rising smoothly from values near zero to values near one and crossing one-half at zero.](/images/notes/hands-on-ml/training-linear-models/sigmoid.png)

### From probability to prediction

Using a threshold of $0.5$:

$$
\hat{y}=
\begin{cases}
0, & \hat{p}<0.5, \\
1, & \hat{p}\geq 0.5.
\end{cases}
$$

The threshold can be changed according to the precision-recall trade-off discussed in the [Classification notes](/notes/hands-on-ml/chapter-03/classification/).

### Odds, logits, and the inverse sigmoid

For $p\in(0,1)$, the **odds** of an event are:

$$
\operatorname{Odds}(p)=\frac{p}{1-p}.
$$

Taking the logarithm gives the **logit**, or log-odds:

$$
t=\operatorname{logit}(p)=\log\left(\frac{p}{1-p}\right).
$$

The logit maps $(0,1)$ to $\mathbb{R}$. In logistic regression, the linear score $\boldsymbol{\theta}^{\mathsf T}\mathbf{x}$ models those log-odds. Solving for $p$:

$$
\frac{p}{1-p}=e^t
\;\Longrightarrow\;
p=e^t(1-p)
\;\Longrightarrow\;
p(1+e^t)=e^t,
$$

$$
p=\frac{e^t}{1+e^t}=\frac{1}{1+e^{-t}}=\sigma(t).
$$

Thus the sigmoid is the inverse of the logit. For finite real scores, its output is strictly between $0$ and $1$; the endpoints are limits.

### Binary cross-entropy loss

For one instance with true label $y\in\{0,1\}$, the loss is:

$$
\ell(y,\hat{p})=
\begin{cases}
-\log(\hat{p}), & y=1, \\
-\log(1-\hat{p}), & y=0.
\end{cases}
$$

The branch depends on the **true label**, not the predicted class. Assigning a tiny probability to the true class incurs a large loss, since $-\log(u)\to+\infty$ as $u\to0^+$.

![Original handwritten sketch of negative log t, tending to infinity near zero and crossing zero at t equal to one.](/images/notes/hands-on-ml/training-linear-models/negative-log-loss.png)

*The original sketch shows the general function $-\log(t)$. For probability-based log loss, the relevant range is $0<t\leq1$.*

Average over the training set to obtain **log loss**, also called binary cross-entropy:

$$
J(\boldsymbol{\theta})
=-\frac{1}{m}\sum_{i=1}^{m}
\left[y^{(i)}\log\hat{p}^{(i)}
+(1-y^{(i)})\log(1-\hat{p}^{(i)})\right].
$$

In vector form, with logarithms applied elementwise:

$$
J(\boldsymbol{\theta})
=-\frac{1}{m}\left[
\mathbf{y}^{\mathsf T}\log\hat{\mathbf{p}}
+(\mathbf{1}-\mathbf{y})^{\mathsf T}\log(\mathbf{1}-\hat{\mathbf{p}})
\right].
$$

Its gradient is:

$$
\frac{\partial J}{\partial\theta_j}
=\frac{1}{m}\sum_{i=1}^{m}
\left[\sigma(\boldsymbol{\theta}^{\mathsf T}\mathbf{x}^{(i)})-y^{(i)}\right]x_j^{(i)},
$$

$$
\nabla_{\boldsymbol{\theta}}J
=\frac{1}{m}X^{\mathsf T}\left[\sigma(X\boldsymbol{\theta})-\mathbf{y}\right].
$$

There is generally no closed-form solution. The unregularized objective is convex, so suitable optimization methods avoid spurious local minima. This does not guarantee convergence with an arbitrary learning rate; for perfectly separable data, an unregularized finite parameter minimizer may not exist.

### Decision boundaries

At a $0.5$ probability threshold, the boundary satisfies:

$$
\boldsymbol{\theta}^{\mathsf T}\mathbf{x}=0.
$$

![Binary logistic-regression probabilities for Iris virginica versus other flowers as a function of petal width, with the decision boundary marked.](/images/notes/hands-on-ml/training-linear-models/logistic-boundary-1d.png)

*In one dimension, the boundary occurs where the positive-class probability reaches $0.5$.*

![Iris petal length and width with linear logistic-regression probability contours and a dashed decision boundary.](/images/notes/hands-on-ml/training-linear-models/logistic-boundary-2d.png)

*With two input features, the same linear score defines a line. The surrounding contours show different predicted probabilities.*

## Softmax regression

**Softmax regression** extends logistic regression to a single target with more than two mutually exclusive classes.

### Class scores and probabilities

Each class $k$ has its own parameter vector $\boldsymbol{\theta}^{(k)}$. Store the vectors as columns of $\Theta$, and compute:

$$
s_k(\mathbf{x})=\left(\boldsymbol{\theta}^{(k)}\right)^{\mathsf T}\mathbf{x}.
$$

The softmax converts the $K$ scores into probabilities:

$$
\hat{p}_k
=\frac{\exp(s_k(\mathbf{x}))}{\sum_{j=1}^{K}\exp(s_j(\mathbf{x}))},
\qquad
\sum_{k=1}^{K}\hat{p}_k=1.
$$

Choose the class with the largest probability, equivalently the largest score:

$$
\hat{y}=\operatorname*{argmax}_k\hat{p}_k
=\operatorname*{argmax}_k s_k(\mathbf{x}).
$$

> **Multiclass, not multilabel:** Softmax assigns one class per instance. It is not the appropriate output model when several independent labels may be present simultaneously.

### Multiclass cross-entropy

For one-hot targets, $y_k^{(i)}=1$ if instance $i$ belongs to class $k$ and $0$ otherwise. These are target indicators, not estimated probabilities.

$$
J(\Theta)
=-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K}
y_k^{(i)}\log\hat{p}_k^{(i)}.
$$

The gradient for class $k$ is:

$$
\nabla_{\boldsymbol{\theta}^{(k)}}J(\Theta)
=\frac{1}{m}\sum_{i=1}^{m}
\left(\hat{p}_k^{(i)}-y_k^{(i)}\right)\mathbf{x}^{(i)}.
$$

The original notes recommend [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8) as a companion explanation:

[![Original video thumbnail for A Short Introduction to Entropy, Cross-Entropy and KL-Divergence, including its play overlay.](/images/notes/hands-on-ml/training-linear-models/entropy-video.png)](https://www.youtube.com/watch?v=ErfnhcEV1O8)

### Implementation and decision regions

With a multinomial-capable solver such as `lbfgs`, [`sklearn.linear_model.LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) fits the softmax model for multiclass targets. The solver must support the selected penalty and target type. Its `C` parameter is an inverse regularization strength: smaller values mean stronger regularization.

![Softmax decision regions and probability contours for Iris setosa, Iris versicolor, and Iris virginica using petal length and width.](/images/notes/hands-on-ml/training-linear-models/softmax-regions.png)

*The three classes occupy different decision regions, while the contours show how the predicted class probabilities vary across the feature space.*
