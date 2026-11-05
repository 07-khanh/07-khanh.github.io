---
title: "End-to-End Machine Learning Project"
date: "2026-08-20"
chapter: 2
chapterTitle: "End-to-End Machine Learning Project"
bookSlug: "hands-on-ml"
tags: ["machine-learning", "preprocessing", "scikit-learn", "cross-validation"]
description: "The complete ML project workflow: framing the problem, splitting and exploring data, building preprocessing pipelines, evaluating and tuning models, and monitoring deployment."
---

An end-to-end machine learning project involves much more than fitting a model. These notes follow seven stages: frame the problem, get the data, explore it, prepare it, train models, fine-tune the system, and maintain it after deployment.

## 1. Look at the big picture

### Frame the problem

Before selecting an algorithm, clarify:

- **Objective:** What should the system achieve?
- **Current solution:** How is the problem handled now, and what provides a useful baseline?
- **Supervision:** Is the task supervised, unsupervised, or semi-supervised?
- **Prediction task:** Is it classification, regression, or another type of problem?

Check assumptions about how the downstream system will use the model's output. The intended use should guide the task definition and evaluation measure.

### Select a performance measure

For a regression dataset with $m$ observations, let $h(\mathbf{x}^{(i)})$ be the prediction and $y^{(i)}$ the target.

**Root mean square error (RMSE)** squares the errors before averaging, so large errors receive more weight:

$$
\operatorname{RMSE}(\mathbf{X},\mathbf{y},h)
=\sqrt{\frac{1}{m}\sum_{i=1}^{m}
\left(h(\mathbf{x}^{(i)})-y^{(i)}\right)^2}.
$$

**Mean absolute error (MAE)** averages absolute errors and is less sensitive to extreme errors than RMSE:

$$
\operatorname{MAE}(\mathbf{X},\mathbf{y},h)
=\frac{1}{m}\sum_{i=1}^{m}
\left|h(\mathbf{x}^{(i)})-y^{(i)}\right|.
$$

For the residual vector $\mathbf{e}$, these are $\lVert\mathbf{e}\rVert_2/\sqrt{m}$ and $\lVert\mathbf{e}\rVert_1/m$, respectively. Higher-order norms put more emphasis on the largest errors.

## 2. Get the data

### Inspect the dataset

Understand how the values were collected and computed, alongside their types and distributions.

| Method | What it reveals |
| --- | --- |
| `info()` | Column names, data types, and non-null counts |
| `value_counts()` | Frequencies of values or categories in a column |
| `describe()` | Summary statistics such as mean, standard deviation, and quantiles |
| `hist()` | A histogram for each numerical attribute |

### Create a test set

Set aside a test set, for example 20% of the data, before exploring the data in depth.

- **Random split with a fixed seed:** Shuffle observations and select a subset. This is reproducible for the same input, but assignments can change when the dataset is updated or reordered.
- **Stable identifier and hash:** Hash each observation's stable, unique identifier. Assign it to the test set when its hash falls below the test ratio times the hash range. A suitably uniform hash keeps assignments stable as new observations arrive.
- **`train_test_split`:** A convenient way to create a random split.
- **Stratified sampling:** Divide the population into strata and sample proportionally from each, so important groups retain their representation in the split.

Use `StratifiedShuffleSplit` for stratified random splits, or pass the stratification labels through `train_test_split(..., stratify=...)` for a single split. Small random samples can misrepresent important groups by chance; stratification controls their proportions.

## 3. Explore and visualize the data

Explore the **training data**. Make a copy before experimenting, and use a smaller exploration subset if the training set is very large.

### Geographical data

When the data includes longitude and latitude, use a scatter plot to inspect geographical patterns.

### Look for correlations

The **standard correlation coefficient**, or **Pearson's $r$**, measures the direction and strength of a linear relationship between two numerical attributes.

- Values near $+1$ indicate a strong positive linear relationship.
- Values near $-1$ indicate a strong negative linear relationship.
- Values near $0$ indicate little linear association; a nonlinear relationship may still exist.

Calculate pairwise correlations with `corr()`. Use Pandas' `scatter_matrix()` to inspect the relationships visually rather than relying on coefficients alone.

### Experiment with attribute combinations

Combined attributes may expose patterns that individual features hide. Check whether a proposed feature adds useful information: a simple weighted sum of existing attributes can introduce linear redundancy.

## 4. Prepare the data for machine learning

Write reusable preprocessing functions and transformers. Work on copies, and separate the input features from the target.

> **Avoid data leakage:** Fit preprocessing steps on the training data only. Apply the fitted transformations to validation and test data. During cross-validation, preprocessing must be fitted separately within each training fold.

### Clean the data

There are three basic ways to handle missing values:

1. Remove observations with missing values using `dropna()`.
2. Remove an attribute using `drop()`.
3. Fill missing values with a constant or a statistic such as the mean or median using `fillna()` or an imputer.

Scikit-Learn provides several imputers:

| Transformer | Approach |
| --- | --- |
| `SimpleImputer` | Fill each column using a selected strategy, such as the median |
| `KNNImputer` | Estimate missing values from neighboring observations |
| `IterativeImputer` | Repeatedly model each feature with missing values using the other features |

The latter two approaches are more computationally involved; greater complexity does not guarantee better results. Compare their effect on validation performance.

Inspect outliers as well. `IsolationForest` can help identify unusual observations; investigate whether they are errors or meaningful cases before removing them.

### Handle text and categorical attributes

**`OrdinalEncoder`** maps categories to integers from $0$ to $n_{\text{categories}}-1$. If categories have a meaningful order, such as bad, neutral, and good, specify that order explicitly. Integer encoding can make a model treat categories as ordered or numerically spaced.

**`OneHotEncoder`** creates a binary attribute for each category and can return a SciPy sparse matrix. Many categories can produce many additional columns and increase training cost. Useful numerical attributes describing the categories may provide an alternative representation.

### Scale numerical features

**Min-max scaling** maps a feature into a chosen range $[a,b]$ using its training minimum and maximum:

$$
x_{\mathrm{unit}}=\frac{x-x_{\min}}{x_{\max}-x_{\min}},
\qquad
x_{\mathrm{scaled}}=a+(b-a)x_{\mathrm{unit}}.
$$

Use `MinMaxScaler`. New observations outside the training range can map outside $[a,b]$; `clip=True` clips transformed values to the selected range.

**Standardization** subtracts the training mean and divides by the training standard deviation:

$$
x_{\mathrm{std}}=\frac{x-\mu}{\sigma}.
$$

Use `StandardScaler`. A nonconstant training feature then has mean $0$ and standard deviation $1$, with no fixed output range. Standardization remains sensitive to outliers because both the mean and standard deviation are affected by them.

### Transform heavy-tailed and multimodal features

Scaling alone does not change a distribution's shape. For positive features with long right tails, consider a power transformation $x^p$ with $0<p<1$, or a logarithm for a very long tail.

![Handwritten sketch of a long right-tailed distribution becoming more symmetric after a logarithmic transformation.](/images/notes/hands-on-ml/end-to-end-project/log-transform.png)

*A log transformation can compress a long right tail; it does not guarantee a Gaussian distribution.*

Another option is **bucketizing**: divide a feature's range into intervals and replace each value with its bucket index. For a multimodal feature, treating the buckets as categories and one-hot encoding them allows different behavior across ranges.

Alternatively, create a similarity feature for each important mode. A **radial basis function (RBF)** depends only on distance from a fixed reference point. A Gaussian RBF has the form:

$$
\phi(\mathbf{x},\mathbf{c})
=\exp\!\left(-\gamma\lVert\mathbf{x}-\mathbf{c}\rVert_2^2\right),
\qquad \gamma>0.
$$

Here $\mathbf{c}$ is the reference point and $\gamma$ controls how quickly similarity decays. A larger $\gamma$ creates a narrower region of high similarity.

![Handwritten Gaussian radial basis function sketch showing high similarity near the reference point and decay with distance.](/images/notes/hands-on-ml/end-to-end-project/gaussian-rbf.png)

These features can help when membership near a particular mode is related to the target.

The **target** can also be transformed. `TransformedTargetRegressor` combines a target transformation with inverse transformation of predictions.

### Build custom transformers

- Use `FunctionTransformer` to wrap a transformation function.
- For a custom transformer class, implement `fit()` and `transform()`.
- Inherit from `TransformerMixin` to obtain `fit_transform()`.
- Inherit from `BaseEstimator` for parameter handling and integration with Scikit-Learn tools.

### Assemble transformation pipelines

Use `Pipeline` or `make_pipeline` to chain processing steps. Use `ColumnTransformer` to apply different transformations to selected columns, such as numerical and categorical attributes.

Combining preprocessing and the estimator in a pipeline keeps their fitting and application together, including during cross-validation.

## 5. Select and train a model

Use **cross-validation** to compare models more reliably than a single validation split.

In $k$-fold cross-validation:

1. Divide the training set into $k$ non-overlapping folds.
2. Train and evaluate $k$ times, holding out a different fold each time.
3. Fit on the remaining $k-1$ folds and evaluate on the held-out fold.
4. Compare validation scores and shortlist promising models.

![Ten-fold cross-validation diagram with each split holding out a different blue validation fold while the remaining yellow folds are used for training.](/images/notes/hands-on-ml/end-to-end-project/cross-validation.png)

*Each observation serves as validation data once in a complete pass through the folds.*

## 6. Fine-tune the model

### Search hyperparameters

- **Grid search:** `GridSearchCV` evaluates the specified combinations of hyperparameter values using cross-validation.
- **Randomized search:** `RandomizedSearchCV` samples hyperparameter configurations. The iteration count controls the search budget, which is useful for large or continuous search spaces.

### Analyze errors and evaluate the final system

Inspect feature importances and error patterns. These can suggest changes to feature engineering, preprocessing, or model choice. Compare performance across relevant subsets to identify uneven errors or potential bias.

Once the system and hyperparameters are selected, evaluate the final tuned system **once on the held-out test set**. Its score may be worse than the selected validation score because repeated model selection can overfit validation results. A confidence interval can help express uncertainty in the performance estimate.

Before launch, prepare reports and presentations, clean up the code, and record the execution environment.

## 7. Launch, monitor, and maintain

- **Save the system:** Persist the fitted model, its preprocessing steps, and evaluation results. `joblib` can be used for serialization.
- **Deploy:** Integrate the system into production, for example through a REST API.
- **Monitor:** Check model performance, data drift, and stale or invalid inputs.
- **Retrain:** Automate collection of new data and labels, including human ratings when needed. Evaluate updated models before redeployment.

Deployment begins an ongoing cycle of monitoring, evaluation, and maintenance.
