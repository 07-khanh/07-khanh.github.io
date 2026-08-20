---
title: "End-to-End Machine Learning Project"
date: "2026-08-20"
chapter: 2
chapterTitle: "End-to-End Machine Learning Project"
bookSlug: "hands-on-ml"
tags: ["machine-learning", "preprocessing", "scikit-learn"]
description: "My personal handwritten notes covering the full end-to-end ML project pipeline."
---

### Look at the BIG PICTURE

**2. Frame the problem**
* What is the objective?
* How is the current solution?
* What kind of training supervision? (supervised, unsupervised, semi-supervised)
* Classification / Regression...

**Select Performance Measure**
* Typical choice for regression model: Root Mean Square Error (RMSE).
$$ \text{RMSE}(\mathbf{X}, h) = \sqrt{\frac{1}{m} \sum_{i=1}^{m} \left( h(\mathbf{x}^{(i)}) - y^{(i)} \right)^2} $$
* $\text{RMSE}(\mathbf{X}, h)$ penalizes heavily large errors, using the $l_2$ (Euclidean) Norm.
* Also have other functions, e.g., Mean Absolute Error (MAE). 
$$ \text{MAE}(\mathbf{X}, h) = \frac{1}{m} \sum_{i=1}^{m} \left| h(\mathbf{x}^{(i)}) - y^{(i)} \right| $$
* $\text{MAE}(\mathbf{X}, h)$ is less sensitive to outliers, using the $l_1$ (Manhattan) Norm.
* The higher the norm index, the more focus on large errors.

**Check the assumptions**
* For instance, verify how the downstream system will use the model output.

---

### GET THE DATA

**Look at the data**
* `info()` method is useful (attributes' names, types, non-null values...).
* `value_counts()` look at categories of one attribute.
* `describe()` if numerical attributes (mean, standard deviation...).
* `hist()` on the entire dataframe, draw histograms for each numerical attribute. Should know how the values are computed.

**Create test set (e.g., 20%)**
* **Method 1:** Use `np.random.seed` to shuffle and select indices -> breaks when you update the dataset.
* **Method 2:** Use the data's unique identifiers. Compute hash of the identifiers, select hash value smaller than test ratio $\times$ max hash value (as hash ensures uniform distribution).
* **Method 3:** Scikit-Learn `train_test_split` (model selection) based on a specific trait.
* **Stratified Sampling:** the population is divided into subgroups (strata), and the right number of instances is selected from each stratum. The test set is representative of the overall population. Random selection easily leads to bias.
* **Method 4:** Scikit-Learn's `StratifiedShuffleSplit`. The test set's distribution (based on the category) resembles the overall population's distribution. Can do the same (proportional split) with the training set.

---

### EXPLORE & Visualize DATA

* Explore only on training data.
* If training set is large, create a smaller exploration set.
* Should make a copy of the data.
* **Geography:** when there are longitude, latitude -> scatter plot.
* **Look for Correlations:** 
  * Standard Correlation Coefficient (Pearson's $r$).
  * Calculate score using `corr()`.
  * Using Pandas' `scatter_matrix()` function.
* **Attribute Combination:** When creating new combined features, make sure they are not too linearly correlated with existing features (e.g., avoid doing simple weighted sum).

---

### PREPARE DATA FOR ML ALGORITHM
* Write in functions.
* Make copies of the data, separate features & target.

**4.1) Clean the data**
* **Missing values:** 3 treating options
  1. Get rid of those samples -> `dropna()`
  2. Drop the whole attribute -> `drop()`
  3. Set the missing values to some value (zero, mean, median) -> `fillna()`
* **Imputation:**
  * For computation (least destruction), use `SimpleImputer` (`sklearn.impute`).
  * Impute -> `IterativeImputer` more time-consuming.
  * `KNNImputer`.
  * Scikit-Learn also has more powerful imputers.
* **Outliers:** may want to drop, can use `IsolationForest`.

**4.2) Handling Text & Categorical Attributes**
* **`OrdinalEncoder`:** features are converted to ordinal integers $\in [0, n_{\text{categories}} - 1]$. Having "notion of distance" -> used for categories that can be ordered (e.g., bad, neutral, good).
* **`OneHotEncoder`:** one-hot encoding, creates new binary attributes per attribute, returns SciPy sparse matrix. If there are many categories, this creates a lot of additional columns -> slows down training. May want to replace with useful numerical features related to the categories.

**4.3) Feature Scaling & Transformation**
* **Feature Scaling:**
  * **Min-max scaling (Normalization):** scale data into a given range $[min, max]$.
    $$ X_{norm} = \frac{X - X_{min}}{X_{max} - X_{min}} $$
    (min, max along axis=0). Use `MinMaxScaler` + min $[min, max]$. If the test set has outliers, they can be out of range. Can fix by setting `clip=True`.
  * **Standardization:**
    $$ X_{std} = \frac{X - \mu}{\sigma} $$
    Result has $0$ mean, standard deviation = $1$, not restricted in given range -> less affected by outliers. Use `StandardScaler`.
* **Handling heavy tails:** When the feature distribution has a heavy tail, have to transform it to shrink the tail first before scaling.
  * If values are positive, can try raise them to a power between $0$ & $1$. If the feature has a really long, heavy tail (e.g., power law distribution), replace it with its logarithm might help.
  * Can also try bucketizing the feature, chopping the distribution into equal-sized buckets & replace each feature value with the index of the bucket it belongs to.
  * For multimodal (having many modes) feature, can try bucketizing but treat bucket index as category (one-hot encoding) rules for different ranges of this feature value.
  * Another approach for multimodal feature is to add a new feature for each mode (or main ones), representing the similarities between feature values & that particular mode.
  * This measure is typically computed using a radial basis function (RBF) - any function that only depends on the distance between the input value & a fixed point:
    $$ \phi(x, c) = \exp(-\gamma ||x - c||^2) $$
  * Commonly used is Gaussian RBF. $\gamma$ dictates how fast the similarity measure decays. Useful if this particular group is well correlated with the target.
* Sometimes, we also may want to scale the target -> use `TransformedTargetRegressor` for combined transform & inverse transform.

**4.4) Custom Transformers**
* Use `FunctionTransformer`.
* Thanks to scikit-learn duck typing, custom classes only need to implement `fit`, `transform` and `fit_transform` (obtained for free by inheriting `TransformerMixin`).
* Can also inherit from `BaseEstimator` for other methods.

**4.5) Transformation Pipelines**
* Processing steps can be chained together using `Pipeline` or `make_pipeline`.
* Pipelines can be nested together using `ColumnTransformer` -> apply transformations on selected columns.

---

### SELECT & TRAIN A MODEL
* Better evaluation using Cross-Validation.
* Split the training set into $k$ non-overlapping subsets (folds).
* Trains & evaluates $k$ times, picking a different fold for evaluation each time & using the remaining $k-1$ folds for training.
* Shortlist a few promising models.

---

### FINE-TUNE YOUR MODEL
* **Grid Search:** `GridSearchCV` is used to automate the evaluation of hyperparameter combinations.
* **Randomized Search:** `RandomizedSearchCV`, used when hyperparameter search space is large or continuous as it scales to any number of iterations.
* **Analyze & Evaluate:** Inspect feature importances & error patterns -> go back to preprocessing step.
* Perform bias analysis to ensure the model remains fair across different subsets.
* The final tuned system is evaluated once on the test set (can have higher loss as the model learns from validation sets). Can check confidence interval.
* **Prelaunch:** create reports, presentations, clean code, define environments....

---

### Launch, MONITOR & maintain
* **Save the model:** models & cross-validation scores are saved (can use `joblib`).
* **Deployment:** The model is prepared for production & typically deployed to a production environment (e.g., via a REST API).
* **Monitoring & MLOps:** System must be monitored for data drift & stale inputs. Retraining pipelines should be automated using scripts that regularly collect new data, obtain human-rated labels, & evaluate updated models before re-deployment....
