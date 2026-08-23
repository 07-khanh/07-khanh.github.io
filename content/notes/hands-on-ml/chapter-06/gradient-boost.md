---
title: "Gradient Boosting"
date: "2026-08-20"
chapter: 6
chapterTitle: "Ensemble Learning and Random Forests"
bookSlug: "hands-on-ml"
tags: ["machine-learning", "ensemble", "scikit-learn"]
description: "My detailed explanation about gradient boost."
---

# Gradient Boosting

## 1. Big Picture and Intuition

Gradient Boosting is an ensemble learning method that builds a strong model by combining many weak decision trees.

The most important idea to keep in mind is:

> **Gradient Boosting builds a model one small tree at a time, where each new tree tries to correct the mistakes made by the current collection of trees.**

There is also a deeper mathematical interpretation:

> **Each new tree approximates the direction in which the current model should move to reduce the loss function.**

That second statement is where the word **"gradient"** comes from.

---

# 2. What problem is Gradient Boosting trying to solve?

Suppose we have training data

$$
(x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n)
$$

and we want to learn a function

$$
F(x)
$$

that makes good predictions.

We define a loss function measuring how bad our predictions are:

$$
L(y_i,F(x_i))
$$

and our overall objective is

$$
\mathcal{L}(F)=\sum_{i=1}^{n}L(y_i,F(x_i)).
$$

So conceptually, machine learning is:

> **Find a function $F$ that makes the loss as small as possible.**

Gradient Boosting solves this by starting with a very simple model and **gradually adding trees that improve it**.

---

# 3. Before Gradient Boosting: understand Gradient Descent

This is extremely important because the name "Gradient Boosting" comes from the same basic idea.

Suppose we want to minimize a function

$$
J(\theta).
$$

For example:

$$
J(\theta)=(\theta-5)^2.
$$

We want to find the value of $\theta$ that minimizes $J$.

The derivative

$$
\frac{dJ}{d\theta}
$$

tells us the slope.

If the slope is positive, we're going uphill as $\theta$ increases, so we should move left.

If the slope is negative, we're going downhill as $\theta$ increases, so we should move right.

Therefore:

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta
\frac{dJ}{d\theta}
$$

where $\eta$ is the learning rate.

The key idea is:

> **Move in the opposite direction of the gradient because that is the direction that decreases the loss.**

---

# 4. So what does Gradient Boosting do?

Here is the conceptual leap.

In ordinary gradient descent, we have parameters:

$$
\theta_1,\theta_2,\ldots,\theta_p
$$

and we adjust those parameters.

But Gradient Boosting isn't primarily adjusting a fixed set of numerical parameters.

Instead, it is constructing a **function**:

$$
F(x).
$$

We construct that function by adding trees:

$$
F_0(x)
$$

then

$$
F_1(x)=F_0(x)+T_1(x)
$$

then

$$
F_2(x)=F_1(x)+T_2(x)
$$

and eventually

$$
F_M(x)
=
F_0(x)
+
\sum_{m=1}^{M}\eta T_m(x).
$$

So instead of saying:

> "I'll change parameter $\theta$ in the direction of the gradient."

we say:

> "I'll add a new function—a decision tree—that moves my predictions in the direction of the negative gradient."

This is often called **gradient descent in function space**.

That's the fundamental idea behind Gradient Boosting.

---

# 5. Why trees?

Why use decision trees?

Because decision trees are excellent at learning complicated nonlinear relationships.

A small regression tree can essentially say:

> If $x_1 < 10$, make this correction.  
> Otherwise, make that correction.

Another tree can then make another correction.

Another tree can correct what the previous trees couldn't handle.

So the final model is a **sum of many small trees**.

---

# 6. Gradient Boosting vs Random Forest

This distinction is extremely important.

## Random Forest

Build many trees **independently**:

$$
T_1,T_2,T_3,\ldots,T_M
$$

Then average their predictions for regression or vote for classification.

The trees don't depend on one another.

---

## Gradient Boosting

Build trees **sequentially**:

$$
T_1\rightarrow T_2\rightarrow T_3\rightarrow\cdots
$$

where each new tree depends on the current model.

The new tree is specifically designed to correct what the previous trees got wrong.

Therefore:

> **Random Forest = many independent trees working in parallel.**

> **Gradient Boosting = many sequential trees correcting one another.**

---

# 7. The intuition for regression

Suppose we're predicting people's weight.

We have:

| Person | Height | Weight |
|---|---:|---:|
| A | 160 | 60 |
| B | 170 | 65 |
| C | 180 | 80 |
| D | 190 | 90 |

We need a starting prediction.

What is a reasonable prediction if we know nothing about the features?

The average:

$$
\frac{60+65+80+90}{4}=73.75.
$$

So initially:

$$
F_0(x)=73.75.
$$

Every person initially gets predicted as $73.75$.

This prediction is poor, but it gives us a starting point.

---

# 8. The basic regression process

We now repeatedly do the following:

1. Make predictions using the current model.
2. Calculate how wrong those predictions are.
3. Train a tree to predict those errors.
4. Use the tree to make a small correction.
5. Add that correction to the existing model.
6. Repeat.

The important idea is:

> **The next tree is not trying to predict the original target directly. It is trying to predict how the current model needs to change.**

---

# 9. A simple mental simulation

Imagine our initial model says:

> Everyone weighs 70 kg.

### Tree 1

Looks at the features and says:

> Short people are generally lighter.  
> Tall people are generally heavier.

So we make a small correction.

---

### Tree 2

Looks at the remaining errors.

It discovers:

> Some tall people are still underpredicted.

So it makes another correction.

---

### Tree 3

Looks at what remains.

It discovers another systematic pattern.

So it makes another correction.

---

Eventually:

$$
\text{Prediction}
=
\text{initial prediction}
+
\text{correction}_1
+
\text{correction}_2
+
\cdots
$$

That is the intuitive essence of Gradient Boosting.

---

# 10. Why is this called "boosting"?

Because we're combining many **weak learners** to produce a strong learner.

A small tree might be a mediocre predictor.

But:

$$
T_1+T_2+T_3+\cdots+T_M
$$

can become extremely powerful.

The trees "boost" one another because each new tree improves the ensemble produced by the previous trees.

---

# 11. Gradient Descent vs Gradient Boosting

This distinction is worth making extremely clear.

| Gradient Descent | Gradient Boosting |
|---|---|
| Optimizes numerical parameters | Builds an additive function |
| Updates parameters | Adds trees |
| Uses gradient | New tree approximates negative gradient |
| Example: neural network weights | Example: decision trees |
| $\theta\leftarrow\theta-\eta\nabla J$ | $F\leftarrow F+\eta T$ |
| Usually same model structure | Model grows by adding learners |

The **idea of moving downhill is the same**.

The object being modified is different.

Gradient descent says:

> "Change my parameters."

Gradient Boosting says:

> "Add another function/tree that moves my predictions downhill."

---

# 12. Why the learning rate matters

Suppose a tree says:

> "The prediction should increase by 20."

If we immediately add 20, we may overshoot.

Instead, suppose:

$$
\eta=0.1.
$$

Then:

$$
\text{correction}=0.1(20)=2.
$$

So we take a small step.

This is directly analogous to gradient descent:

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla J.
$$

Gradient Boosting similarly takes a small step:

$$
F_{\text{new}}
=
F_{\text{old}}
+
\eta(\text{negative-gradient tree}).
$$

A smaller learning rate generally means that more trees are needed.

---

# 13. Why are the trees usually small?

Suppose each tree is extremely powerful.

The first tree might almost perfectly fit the training data.

Then:

$$
T_1(x)
$$

could already memorize everything.

There wouldn't be much useful work left for subsequent trees.

That makes overfitting easier.

Instead, Gradient Boosting commonly uses **shallow trees**.

Think of each tree as saying:

> "I can only make a small, relatively simple correction."

Then another tree makes another correction.

Then another.

This creates a gradual refinement process.

---

# 14. The deeper connection to gradients

Everything so far has been intuition.

Now we can derive exactly why the algorithm works.

Suppose the current model is:

$$
F_{m-1}(x).
$$

We calculate how the loss changes if we change the prediction.

That's the gradient:

$$
\frac{\partial L}{\partial F}.
$$

We want the **negative gradient**:

$$
-\frac{\partial L}{\partial F}.
$$

Then we train a tree to approximate that negative gradient:

$$
T_m(x)\approx
-\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}.
$$

Then take a small step:

$$
F_m(x)
=
F_{m-1}(x)+\eta T_m(x).
$$

Repeat.

That's Gradient Boosting.

---

# 15. The general Gradient Boosting algorithm

The entire algorithm can be summarized as follows.

### Step 0 — choose a loss

Choose:

$$
L(y,F(x)).
$$

For example:

- squared error for regression
- log loss for binary classification

---

### Step 1 — initialize

Find the constant prediction:

$$
F_0(x)=
\arg\min_c
\sum_i L(y_i,c).
$$

---

### Step 2 — repeat for $m=1,\ldots,M$

#### 2A. Calculate the negative gradient

For every training sample:

$$
r_{im}
=
-
\left[
\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}
\right]_{F=F_{m-1}}.
$$

This is called the **pseudo-residual**.

---

#### 2B. Fit a tree

Train a regression tree:

$$
T_m(x)
$$

to predict the pseudo-residuals $r_{im}$.

---

#### 2C. Determine leaf values

Find the output value for each leaf that minimizes the loss.

---

#### 2D. Update the model

$$
F_m(x)
=
F_{m-1}(x)
+
\eta T_m(x).
$$

---

### Step 3 — final prediction

For regression:

$$
\hat y=F_M(x).
$$

For binary classification:

$$
p=
\frac{1}{1+e^{-F_M(x)}}.
$$

Then convert $p$ to a class if needed.

---

# 16. Regression: Mathematical Derivation

Now let's derive every step mathematically.

---

## 16.1 Regression objective

For regression, suppose we use squared-error loss:

$$
L(y,F(x))
=
\frac12(y-F(x))^2.
$$

Therefore the total loss is:

$$
\mathcal L(F)
=
\sum_{i=1}^{n}
\frac12
(y_i-F(x_i))^2.
$$

Our goal is to find a function $F$ that minimizes this quantity.

---

# 17. Regression Step 1: derive the initialization

Before building any trees, we need $F_0(x)$.

We start with a constant prediction:

$$
F_0(x)=\gamma.
$$

Why a constant?

Because at the beginning we don't want the model to depend on any features. We want the **best possible constant prediction**.

So choose $\gamma$ that minimizes the loss:

$$
\gamma^*
=
\arg\min_\gamma
\sum_{i=1}^{n}
\frac12(y_i-\gamma)^2.
$$

Differentiate with respect to $\gamma$:

$$
\frac{d}{d\gamma}
\sum_i
\frac12(y_i-\gamma)^2
=
\sum_i(\gamma-y_i).
$$

Set the derivative equal to zero:

$$
\sum_i(\gamma-y_i)=0.
$$

Expand:

$$
n\gamma-\sum_i y_i=0.
$$

Therefore:

$$
n\gamma=\sum_i y_i.
$$

Thus:

$$
\boxed{
\gamma=\frac1n\sum_i y_i
}
$$

Therefore:

$$
\boxed{
F_0(x)=\bar y
}
$$

The initial prediction is the **mean of the target values**.

The mean is therefore not an arbitrary choice. It is the constant that minimizes squared-error loss.

---

# 18. Regression Step 2: derive the gradient

Now we have $F_0$.

For each observation, calculate:

$$
\frac{\partial L(y_i,F(x_i))}
{\partial F(x_i)}.
$$

Our loss is:

$$
L_i=
\frac12(y_i-F(x_i))^2.
$$

Differentiate:

$$
\frac{\partial L_i}{\partial F(x_i)}
=
\frac12
\cdot2(y_i-F(x_i))
\cdot(-1).
$$

Therefore:

$$
\frac{\partial L_i}{\partial F(x_i)}
=
F(x_i)-y_i.
$$

But gradient descent moves in the **negative gradient** direction:

$$
-\frac{\partial L_i}{\partial F(x_i)}
=
y_i-F(x_i).
$$

Therefore:

$$
\boxed{
r_i=y_i-F(x_i)
}
$$

This is exactly the ordinary regression residual.

So:

> **For squared-error regression, the ordinary residual is the negative gradient of the loss with respect to the model's prediction.**

This is why the algorithm can use residuals.

---

# 19. Regression Step 3: fit a tree to the gradients

Suppose after initialization:

$$
F_0(x_i)=\bar y.
$$

We calculate:

$$
r_{i1}=y_i-F_0(x_i).
$$

Now we have a new supervised-learning dataset:

$$
(x_i,r_{i1}).
$$

We train a regression tree on it.

The tree tries to approximate:

$$
r_{i1}\approx T_1(x_i).
$$

Therefore:

$$
\boxed{
T_1(x)
\approx
-\frac{\partial L}{\partial F}
}
$$

The tree is therefore learning:

> **How much should the current prediction be corrected, based on the features?**

---

# 20. Regression Step 4: derive the leaf values

Suppose our tree creates a leaf $R$ containing several observations.

Every observation in this leaf will receive the same correction $\gamma$.

We therefore want:

$$
\gamma^*
=
\arg\min_\gamma
\sum_{x_i\in R}
\frac12
(y_i-[F_0(x_i)+\gamma])^2.
$$

Differentiate:

$$
\frac{d}{d\gamma}
\sum_{x_i\in R}
\frac12
(y_i-F_0(x_i)-\gamma)^2
=
\sum_{x_i\in R}
(F_0(x_i)+\gamma-y_i).
$$

Set equal to zero:

$$
\sum_{x_i\in R}
(F_0(x_i)+\gamma-y_i)=0.
$$

Rearrange:

$$
\sum_{x_i\in R}
\gamma
=
\sum_{x_i\in R}
(y_i-F_0(x_i)).
$$

If there are $|R|$ observations:

$$
|R|\gamma
=
\sum_{x_i\in R}
(y_i-F_0(x_i)).
$$

Therefore:

$$
\boxed{
\gamma
=
\frac1{|R|}
\sum_{x_i\in R}
(y_i-F_0(x_i))
}
$$

But:

$$
y_i-F_0(x_i)=r_i.
$$

Therefore:

$$
\boxed{
\gamma
=
\text{mean residual in the leaf}
}
$$

That's why the leaf values are averages of residuals for squared-error Gradient Boosting.

---

# 21. Regression Step 5: derive the update

Suppose the tree produces $T_1(x)$.

We want to move our model toward lower loss.

Since $T_1$ approximates the negative gradient:

$$
T_1
\approx
-\nabla_F\mathcal L,
$$

we update:

$$
\boxed{
F_1(x)
=
F_0(x)+\eta T_1(x)
}
$$

Why the plus sign?

Because $T_1$ is already approximating the **negative** gradient.

Ordinary gradient descent is:

$$
\theta_{\text{new}}
=
\theta_{\text{old}}
-
\eta\nabla J.
$$

Gradient Boosting is:

$$
F_{\text{new}}
=
F_{\text{old}}
+
\eta(\text{negative gradient}).
$$

These are the same idea.

---

# 22. Regression: repeat the process

At iteration $m$:

### Current model

$$
F_{m-1}(x)
$$

### Calculate negative gradient

$$
r_{im}
=
y_i-F_{m-1}(x_i)
$$

### Fit regression tree

$$
T_m(x)\approx r_{im}
$$

### Find optimal leaf values

For squared loss:

$$
\gamma_{jm}
=
\frac{
\sum_{x_i\in R_{jm}}
r_{im}
}{
|R_{jm}|
}
$$

### Update

$$
\boxed{
F_m(x)
=
F_{m-1}(x)+
\eta T_m(x)
}
$$

Eventually:

$$
\boxed{
F_M(x)
=
\bar y+
\eta\sum_{m=1}^{M}T_m(x)
}
$$

---

# 23. What is each regression tree actually doing?

This is the most useful mental model.

### Tree 1

> "Here's my best correction to the original prediction."

### Tree 2

> "After Tree 1 corrected things, here's what's still wrong."

### Tree 3

> "After Trees 1 and 2, here's what's still wrong."

### Tree 4

> "Here's another correction."

And so on.

Therefore:

> **Gradient Boosting doesn't build many trees to independently solve the original problem. It builds trees to successively solve the remaining error.**

---

# 24. Classification: Mathematical Derivation

Now suppose:

$$
y_i\in\{0,1\}.
$$

We cannot simply use the regression setup because the target is categorical.

Instead, we use **log loss** and represent the model output as **log-odds**.

---

# 25. Classification: define the model output

Let:

$$
p(x)=P(y=1\mid x).
$$

The model output is:

$$
F(x)
=
\log
\left(
\frac{p(x)}{1-p(x)}
\right).
$$

This is the **log-odds**.

Why use log-odds instead of probability?

Probability is constrained:

$$
0\le p\le1.
$$

But log-odds can take any real value:

$$
-\infty<F<\infty.
$$

That makes it natural to add tree corrections:

$$
F_{\text{new}}
=
F_{\text{old}}
+
\text{correction}.
$$

---

# 26. Convert log-odds to probability

Starting with:

$$
F=\log\frac{p}{1-p}.
$$

Exponentiate:

$$
e^F=\frac{p}{1-p}.
$$

Multiply:

$$
e^F(1-p)=p.
$$

Expand:

$$
e^F-e^Fp=p.
$$

Move terms:

$$
e^F=p+e^Fp.
$$

Factor:

$$
e^F=p(1+e^F).
$$

Therefore:

$$
p=\frac{e^F}{1+e^F}.
$$

Divide numerator and denominator by $e^F$:

$$
\boxed{
p=\frac1{1+e^{-F}}
}
$$

This is the logistic function.

---

# 27. Classification Step 1: choose the loss

For binary classification, use log loss:

$$
L(y,p)
=
-
\left[
y\log p+(1-y)\log(1-p)
\right].
$$

The total loss is:

$$
\mathcal L
=
-\sum_{i=1}^{n}
\left[
y_i\log p_i
+
(1-y_i)\log(1-p_i)
\right].
$$

Our goal is to find the model $F(x)$ that minimizes this loss.

---

# 28. Classification Step 2: derive initialization

We start with a constant prediction:

$$
F_0(x)=\gamma.
$$

Since $F$ is log-odds, this means every sample initially has the same probability $p$.

We therefore find the probability $p$ that minimizes:

$$
\mathcal L(p)
=
-\sum_i
[
y_i\log p+
(1-y_i)\log(1-p)
].
$$

Let:

$$
S=\sum_i y_i.
$$

Since $y_i$ is either 0 or 1, $S$ is the number of positive examples.

Then:

$$
\mathcal L(p)
=
-
\left[
S\log p+
(n-S)\log(1-p)
\right].
$$

Differentiate:

$$
\frac{d\mathcal L}{dp}
=
-\frac{S}{p}
+
\frac{n-S}{1-p}.
$$

Set equal to zero:

$$
-\frac{S}{p}
+
\frac{n-S}{1-p}
=0.
$$

Therefore:

$$
\frac{n-S}{1-p}
=
\frac{S}{p}.
$$

Cross multiply:

$$
p(n-S)=S(1-p).
$$

Expand:

$$
pn-pS=S-Sp.
$$

The terms involving $pS$ cancel:

$$
pn=S.
$$

Therefore:

$$
\boxed{
p=\frac{S}{n}
}
$$

So:

$$
\boxed{
p_0=
\frac{\text{number of positives}}{n}
}
$$

The best constant probability prediction is simply the fraction of positive examples.

---

# 29. Convert the initial probability to log-odds

Our model output is $F$, not $p$.

Recall:

$$
F=
\log\frac{p}{1-p}.
$$

Substitute:

$$
p=\frac{S}{n}.
$$

Then:

$$
1-p
=
1-\frac Sn
=
\frac{n-S}{n}.
$$

Therefore:

$$
\frac{p}{1-p}
=
\frac{S/n}{(n-S)/n}
=
\frac{S}{n-S}.
$$

Thus:

$$
\boxed{
F_0
=
\log
\left(
\frac{S}{n-S}
\right)
}
$$

Equivalently:

$$
\boxed{
F_0=
\log
\left(
\frac{\text{number of positives}}
{\text{number of negatives}}
\right)
}
$$

This is why the initial model is based on the log-odds of the positive class.

---

# 30. Classification Step 3: derive the gradient

We have:

$$
F(x)
$$

and:

$$
p(x)=\frac1{1+e^{-F(x)}}.
$$

We need:

$$
\frac{\partial L}{\partial F}.
$$

We use the chain rule.

---

## 30.1 Calculate $\frac{dp}{dF}$

We have:

$$
p=\frac1{1+e^{-F}}.
$$

Rewrite:

$$
p=(1+e^{-F})^{-1}.
$$

Differentiate:

$$
\frac{dp}{dF}
=
-(1+e^{-F})^{-2}(-e^{-F}).
$$

Therefore:

$$
\frac{dp}{dF}
=
\frac{e^{-F}}{(1+e^{-F})^2}.
$$

This simplifies to:

$$
\boxed{
\frac{dp}{dF}=p(1-p)
}
$$

---

## 30.2 Calculate $\frac{\partial L}{\partial p}$

We have:

$$
L
=
-y\log p-(1-y)\log(1-p).
$$

Differentiate:

$$
\frac{\partial L}{\partial p}
=
-\frac yp
+
\frac{1-y}{1-p}.
$$

Now use the chain rule:

$$
\frac{\partial L}{\partial F}
=
\frac{\partial L}{\partial p}
\frac{dp}{dF}.
$$

Therefore:

$$
\frac{\partial L}{\partial F}
=
\left(
-\frac yp+
\frac{1-y}{1-p}
\right)
p(1-p).
$$

Distribute:

$$
=
-y(1-p)+(1-y)p.
$$

Expand:

$$
=
-y+yp+p-yp.
$$

The $yp$ terms cancel:

$$
\boxed{
\frac{\partial L}{\partial F}
=
p-y
}
$$

Therefore the **negative gradient** is:

$$
\boxed{
-\frac{\partial L}{\partial F}
=
y-p
}
$$

So the classification pseudo-residual is:

$$
\boxed{
r_i=y_i-p_i
}
$$

---

# 31. What does $y-p$ mean intuitively?

Suppose the true class is positive:

$$
y=1.
$$

If:

$$
p=0.9,
$$

then:

$$
y-p=1-0.9=0.1.
$$

The model is already doing well, so only a small positive correction is needed.

But if:

$$
p=0.2,
$$

then:

$$
y-p=1-0.2=0.8.
$$

The model strongly underestimates the probability of the positive class, so we need a much larger positive correction.

Now suppose:

$$
y=0.
$$

If:

$$
p=0.9,
$$

then:

$$
y-p=0-0.9=-0.9.
$$

The model is strongly overpredicting the positive class, so we need a large negative correction.

Thus:

> **The pseudo-residual tells us both the direction and magnitude of the correction needed by the current model.**

---

# 32. Classification Step 4: fit a regression tree

We now have:

$$
r_i=y_i-p_i.
$$

We create a new training target:

$$
(x_i,r_i).
$$

Then train a regression tree.

Why regression?

Because $r_i$ is continuous.

For example:

| $y$ | $p$ | $y-p$ |
|---:|---:|---:|
| 1 | 0.8 | 0.2 |
| 1 | 0.3 | 0.7 |
| 0 | 0.2 | -0.2 |
| 0 | 0.9 | -0.9 |

The tree predicts these continuous corrections.

So even though we're solving a classification problem, the individual boosting trees are typically **regression trees** because they're predicting corrections rather than class labels.

---

# 33. Classification Step 5: derive the leaf value

For regression, we can use the average residual in a leaf.

For classification with log loss, the optimal leaf value requires additional mathematics.

Suppose leaf $R_j$ gives a constant correction $\gamma_j$.

Our updated model inside this leaf is:

$$
F_{\text{new}}(x_i)
=
F_{\text{old}}(x_i)+\gamma_j.
$$

We want:

$$
\gamma_j^*
=
\arg\min_\gamma
\sum_{x_i\in R_j}
L(y_i,F_i+\gamma).
$$

For log loss, we use a **second-order Taylor approximation**.

---

# 34. Taylor expansion

For a small change $\gamma$:

$$
L_i(F_i+\gamma)
\approx
L_i(F_i)
+
g_i\gamma
+
\frac12h_i\gamma^2,
$$

where:

$$
g_i
=
\frac{\partial L_i}{\partial F_i}
$$

and:

$$
h_i
=
\frac{\partial^2L_i}{\partial F_i^2}.
$$

We already derived:

$$
g_i=p_i-y_i.
$$

Now differentiate again.

Since:

$$
g_i=p_i-y_i,
$$

and $y_i$ is constant:

$$
h_i=\frac{dp_i}{dF_i}.
$$

But:

$$
\frac{dp_i}{dF_i}=p_i(1-p_i).
$$

Therefore:

$$
\boxed{
h_i=p_i(1-p_i)
}
$$

---

# 35. Minimize the Taylor approximation

For a leaf $R_j$, the approximate loss is:

$$
\sum_{i\in R_j}
\left[
L_i
+
g_i\gamma
+
\frac12h_i\gamma^2
\right].
$$

The $L_i$ terms don't depend on $\gamma$, so we minimize:

$$
\sum_{i\in R_j}
\left[
g_i\gamma
+
\frac12h_i\gamma^2
\right].
$$

Differentiate:

$$
\sum_{i\in R_j}
\left[
g_i+h_i\gamma
\right]
=0.
$$

Therefore:

$$
\sum_i g_i
+
\gamma\sum_i h_i
=0.
$$

Rearrange:

$$
\gamma\sum_i h_i
=
-\sum_i g_i.
$$

Therefore:

$$
\gamma
=
-\frac{\sum_i g_i}
{\sum_i h_i}.
$$

Substitute:

$$
g_i=p_i-y_i
$$

and:

$$
h_i=p_i(1-p_i).
$$

Thus:

$$
\gamma
=
-\frac{
\sum_i(p_i-y_i)
}{
\sum_i p_i(1-p_i)
}.
$$

Therefore:

$$
\boxed{
\gamma
=
\frac{
\sum_i(y_i-p_i)
}{
\sum_i p_i(1-p_i)
}
}
$$

This is the classification leaf-value formula.

---

# 36. Why does $p(1-p)$ appear?

For logistic regression:

$$
p=\frac1{1+e^{-F}}.
$$

We derived:

$$
\frac{dp}{dF}=p(1-p).
$$

Therefore:

$$
p(1-p)
$$

appears naturally as the second derivative of the log-loss with respect to the model's log-odds output.

It represents the **curvature** of the loss.

So the classification leaf value is essentially:

$$
\boxed{
\text{step size}
=
\frac{\text{negative gradient}}
{\text{curvature}}
}
$$

This is closely related to **Newton's method**.

---

# 37. Classification Step 6: update the model

Suppose a sample belongs to leaf $R_j$.

Its current log-odds are:

$$
F_{m-1}(x_i).
$$

The tree gives correction:

$$
\gamma_{jm}.
$$

Apply the learning rate:

$$
\eta\gamma_{jm}.
$$

Then:

$$
\boxed{
F_m(x_i)
=
F_{m-1}(x_i)
+
\eta\gamma_{jm}
}
$$

Then convert log-odds to probability:

$$
\boxed{
p_m(x_i)
=
\frac1{1+e^{-F_m(x_i)}}
}
$$

Then calculate new pseudo-residuals:

$$
r_{im+1}=y_i-p_m(x_i).
$$

And repeat.

---

# 38. Complete classification algorithm

### Initialization

Calculate:

$$
p_0
=
\frac{\text{number of positives}}{n}.
$$

Then:

$$
F_0
=
\log
\frac{p_0}{1-p_0}.
$$

---

### Iteration $m$

Calculate probabilities:

$$
p_{i,m-1}
=
\frac1{1+e^{-F_{m-1}(x_i)}}.
$$

Calculate negative gradients:

$$
\boxed{
r_{im}
=
y_i-p_{i,m-1}
}
$$

Fit a regression tree to $r_{im}$.

For each leaf $R_{jm}$, calculate:

$$
\boxed{
\gamma_{jm}
=
\frac{
\sum_{x_i\in R_{jm}}
(y_i-p_i)
}{
\sum_{x_i\in R_{jm}}
p_i(1-p_i)
}
}
$$

Update:

$$
\boxed{
F_m(x)
=
F_{m-1}(x)+
\eta\gamma_{jm}
}
$$

Then:

$$
p_m(x)
=
\frac1{1+e^{-F_m(x)}}.
$$

Repeat.

---

# 39. Regression vs Classification

| | Regression | Classification |
|---|---|---|
| Target | Continuous $y$ | $y\in\{0,1\}$ |
| Loss | $\frac12(y-F)^2$ | $-[y\log p+(1-y)\log(1-p)]$ |
| Model output | $F$ | $F=\log\frac p{1-p}$ |
| Probability | — | $p=\frac1{1+e^{-F}}$ |
| Initialization | $\bar y$ | $\log\frac{\#positive}{\#negative}$ |
| Gradient | $F-y$ | $p-y$ |
| Negative gradient | $y-F$ | $y-p$ |
| Tree target | $y-F$ | $y-p$ |
| Second derivative | $1$ | $p(1-p)$ |
| Leaf value | $\frac{\sum(y-F)}{|R|}$ | $\frac{\sum(y-p)}{\sum p(1-p)}$ |
| Update | $F_m=F_{m-1}+\eta\gamma$ | $F_m=F_{m-1}+\eta\gamma$ |
| Final output | $F_M(x)$ | $\frac1{1+e^{-F_M(x)}}$ |

---

# 40. The common structure

Despite looking different, both algorithms follow exactly the same philosophy.

## Regression

$$
\boxed{
\begin{aligned}
F_0&=\arg\min_F\mathcal L\\
r_i&=-\frac{\partial L_i}{\partial F_i}\\
T_m&\approx r_i\\
\gamma&=\arg\min_\gamma\mathcal L\\
F_m&=F_{m-1}+\eta\gamma
\end{aligned}
}
$$

For squared loss, this becomes:

$$
F_0=\bar y
$$

$$
r_i=y_i-F_i
$$

and:

$$
\gamma=\text{mean residual}.
$$

---

## Classification

$$
\boxed{
\begin{aligned}
F_0&=\arg\min_F\mathcal L\\
r_i&=-\frac{\partial L_i}{\partial F_i}\\
T_m&\approx r_i\\
\gamma&=\arg\min_\gamma\mathcal L\\
F_m&=F_{m-1}+\eta\gamma
\end{aligned}
}
$$

For log loss:

$$
F_0
=
\log
\frac{\#positive}{\#negative}
$$

$$
r_i=y_i-p_i
$$

and:

$$
\gamma
=
\frac{\sum(y_i-p_i)}
{\sum p_i(1-p_i)}.
$$

---

# 41. Gradient Descent vs Gradient Boosting

Now we can connect everything back to ordinary Gradient Descent.

### Ordinary Gradient Descent

$$
\theta_m
=
\theta_{m-1}
-
\eta
\frac{\partial\mathcal L}
{\partial\theta}.
$$

### Gradient Boosting

$$
F_m
=
F_{m-1}
+
\eta T_m.
$$

where:

$$
T_m
\approx
-\frac{\partial\mathcal L}{\partial F}.
$$

Therefore:

$$
F_m
\approx
F_{m-1}
-
\eta
\frac{\partial\mathcal L}{\partial F}.
$$

So the fundamental optimization idea is the same.

The difference is **what is being updated**:

- Gradient Descent updates numerical parameters.
- Gradient Boosting adds a new function/tree.

---

# 42. Why Gradient Boosting is so powerful

Suppose the current model has a complicated error pattern:

$$
\text{Current model}
\rightarrow
\text{errors}
\rightarrow
T_1\text{ learns an error pattern}
\rightarrow
\text{remaining errors}
\rightarrow
T_2\text{ learns another pattern}
\rightarrow
\text{remaining errors}
\rightarrow
T_3\text{ learns another pattern}
\rightarrow\cdots
$$

Each tree doesn't have to solve the entire problem.

It only has to solve **part of the remaining problem**.

That's why a collection of relatively simple trees can create a highly sophisticated nonlinear model.

---

# 43. A complete mental simulation

Imagine a regression problem.

Our initial model says:

> Everyone weighs 70 kg.

### Tree 1

Looks at the features and says:

> Short people are generally underrepresented by the current prediction, while tall people are generally underrepresented in the opposite direction.

It makes small corrections.

---

### Tree 2

Looks at the remaining errors.

It discovers another feature-based pattern.

It makes another correction.

---

### Tree 3

Looks at what remains.

It discovers another pattern.

It makes another correction.

---

Eventually:

$$
\text{Prediction}
=
\text{initial guess}
+
\text{correction}_1
+
\text{correction}_2
+
\cdots
$$

The model has gradually moved toward a function that minimizes the loss.

---

# 44. The most important distinction between residuals and gradients

For regression with squared-error loss:

$$
\boxed{
\text{negative gradient}
=
y-\hat y
=
\text{ordinary residual}
}
$$

So we can simply say:

> "Fit the next tree to the residuals."

For classification with log loss:

$$
\boxed{
\text{negative gradient}
=
y-p
}
$$

So we use **pseudo-residuals**.

Therefore, when someone says:

> "Gradient Boosting fits trees to the residuals."

The more general and mathematically correct statement is:

> **Gradient Boosting fits each new tree to the negative gradient of the loss.**

Ordinary residuals are just a special case that occurs for squared-error regression.

---

# 45. The complete algorithm in one picture

The entire process can be remembered as:

$$
\boxed{
\begin{array}{c}
\text{Initialize a simple model }F_0\\
\downarrow\\
\text{Calculate current predictions}\\
\downarrow\\
\text{Calculate the negative gradient}\\
\downarrow\\
\text{Train a tree to approximate it}\\
\downarrow\\
\text{Determine the tree's optimal contribution}\\
\downarrow\\
\text{Multiply by learning rate }\eta\\
\downarrow\\
\text{Add the tree to the existing model}\\
\downarrow\\
\text{Repeat}
\end{array}
}
$$

For regression:

$$
\boxed{
r_i=y_i-\hat y_i
}
$$

For classification:

$$
\boxed{
r_i=y_i-p_i
}
$$

And for both:

$$
\boxed{
F_m(x)=F_{m-1}(x)+\eta T_m(x)
}
$$

---

# 46. Final mental model

If you remember only one thing, remember this:

> **Gradient Boosting starts with a simple prediction and repeatedly adds small decision trees that approximate the negative gradient of the loss, so that each new tree moves the overall model toward lower loss.**

The entire algorithm is therefore:

$$
\boxed{
\text{Current model}
\rightarrow
\text{loss}
\rightarrow
\text{gradient}
\rightarrow
\text{negative gradient}
\rightarrow
\text{tree}
\rightarrow
\text{tree contribution}
\rightarrow
\text{small update}
\rightarrow
\text{new model}
\rightarrow
\text{repeat}
}
$$

For regression:

$$
\boxed{
F_0=\bar y
\quad\rightarrow\quad
r_i=y_i-F(x_i)
\quad\rightarrow\quad
\text{fit tree}
\quad\rightarrow\quad
F_m=F_{m-1}+\eta T_m
}
$$

For binary classification:

$$
\boxed{
F_0=
\log\frac{\#positive}{\#negative}
\quad\rightarrow\quad
p_i=\frac1{1+e^{-F_i}}
\quad\rightarrow\quad
r_i=y_i-p_i
\quad\rightarrow\quad
\text{fit tree}
\quad\rightarrow\quad
F_m=F_{m-1}+\eta\gamma_m
}
$$

The central equation is:

$$
\boxed{
F_m(x)
=
F_{m-1}(x)
+
\eta T_m(x)
}
$$

because the new tree $T_m$ is an approximation of the direction that reduces the loss.
