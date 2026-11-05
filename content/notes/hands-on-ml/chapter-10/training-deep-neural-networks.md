---
title: "Training Deep Neural Networks"
date: "2026-10-01"
chapter: 10
chapterTitle: "Building Neural Networks with PyTorch"
bookSlug: "hands-on-ml"
tags: ["deep-learning", "pytorch", "initialization", "activation-functions", "normalization", "transfer-learning", "optimization", "learning-rate", "regularization"]
description: "Stable deep-network training with initialization, activations, normalization, clipping, transfer learning, adaptive optimizers, learning-rate schedules, and regularization."
---

Training deep neural networks requires controlling signal scales, choosing useful update rules, and limiting overfitting. These Chapter 10 notes build on [perceptrons and backpropagation](/notes/hands-on-ml/chapter-09/perceptrons-and-backpropagation/) and cover practical training techniques in PyTorch.

## Vanishing and exploding gradients

Backpropagation computes loss gradients from the output toward earlier layers. An optimizer then uses those gradients to update parameters. Across many layers, repeated products of derivative factors can make gradients shrink or grow dramatically:

- **Vanishing gradients:** Earlier layers receive very small updates and may learn extremely slowly.
- **Exploding gradients:** Large gradients can cause excessive parameter updates and unstable training.

Activation scales also matter in the forward pass. For example, sigmoid saturates near zero or one when its input has large magnitude. Its derivative, $\sigma'(z)=\sigma(z)(1-\sigma(z))$, then becomes very small.

![Sigmoid activation showing its approximately linear central region and saturated tails.](/images/notes/hands-on-ml/training-deep-networks/sigmoid-saturation.png)

Initialization and architecture choices aim to keep activations and gradients at useful scales. Exact variance preservation through every nonlinear layer is not generally possible; the calculations below provide useful approximations under stated assumptions.

## Xavier and He initialization

For a dense layer, **fan-in** is its number of input connections per neuron and **fan-out** its number of output connections per input. In the normal distributions below, the second argument is the **variance**.

### Why fan-in and fan-out matter

Consider a preactivation with zero bias:

$$
u=\sum_{i=1}^{n_{\mathrm{in}}}w_i x_i.
$$

Assume zero-centered inputs and weights, independence between weights and inputs, and sufficiently independent components with common variances. For independent random variables $A$ and $B$:

$$
\operatorname{Var}(AB)
=\operatorname{Var}(A)\operatorname{Var}(B)
+\mathbb{E}[A]^2\operatorname{Var}(B)
+\mathbb{E}[B]^2\operatorname{Var}(A).
$$

With zero means, the extra terms vanish, giving:

$$
\operatorname{Var}(u)
=n_{\mathrm{in}}\operatorname{Var}(w)\operatorname{Var}(x).
$$

For an approximately linear activation, matching input and output variance suggests $\operatorname{Var}(w)=1/n_{\mathrm{in}}$. A corresponding backward-pass calculation suggests $1/n_{\mathrm{out}}$.

### Xavier or Glorot initialization

Xavier initialization balances these forward and backward scales using:

$$
n_{\mathrm{avg}}=\frac{n_{\mathrm{in}}+n_{\mathrm{out}}}{2},
\qquad
\operatorname{Var}(w)=\frac{g^2}{n_{\mathrm{avg}}}
=\frac{2g^2}{n_{\mathrm{in}}+n_{\mathrm{out}}},
$$

where $g$ is an activation-dependent gain. Equivalent normal and uniform constructions are:

$$
w\sim\mathcal{N}\!\left(0,\frac{2g^2}{n_{\mathrm{in}}+n_{\mathrm{out}}}\right),
$$

$$
w\sim\mathcal{U}\!\left(
-g\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}},
g\sqrt{\frac{6}{n_{\mathrm{in}}+n_{\mathrm{out}}}}
\right).
$$

For example, PyTorch's recommended gains include $1$ for sigmoid and $5/3$ for tanh. Use `nn.init.xavier_normal_()` or `nn.init.xavier_uniform_()`, with `nn.init.calculate_gain()` when appropriate. See the [initialization reference](https://docs.pytorch.org/docs/stable/nn.init.html).

> **Keep variance and standard deviation distinct:** The standard deviation is $g/\sqrt{n_{\mathrm{avg}}}$; the variance is $g^2/n_{\mathrm{avg}}$. The average fan count is $(n_{\mathrm{in}}+n_{\mathrm{out}})/2$.

### He or Kaiming initialization

For a symmetric preactivation distribution, ReLU zeros its negative half. The useful scaling relationship is about the **second moment**:

$$
\mathbb{E}[\operatorname{ReLU}(u)^2]
=\frac12\mathbb{E}[u^2].
$$

This is not exactly a halving of variance, because ReLU's output generally has a positive mean. The relationship motivates He initialization for ReLU networks:

$$
\operatorname{Var}(w)=\frac{2}{n_{\mathrm{in}}},
$$

$$
w\sim\mathcal{N}\!\left(0,\frac{2}{n_{\mathrm{in}}}\right)
\quad\text{or}\quad
w\sim\mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\mathrm{in}}}},\sqrt{\frac{6}{n_{\mathrm{in}}}}\right).
$$

The standard-deviation gain is $\sqrt{2}$. For Leaky ReLU with negative slope $\alpha$, it becomes $\sqrt{2/(1+\alpha^2)}$. Choose `mode="fan_in"` to emphasize forward signal scaling or `mode="fan_out"` for backward scaling; Xavier initialization remains another valid family.

```python
import torch
from torch import nn

layer = nn.Linear(128, 64)
nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
nn.init.zeros_(layer.bias)
```

Initialization should match the actual activation and architecture. The ReLU derivation does not automatically determine the best scale for every smooth activation, such as GELU.

## Activation functions

### ReLU and inactive neurons

ReLU is fast and does not saturate for positive inputs. A neuron can nevertheless become inactive if its preactivation is negative for all relevant training examples: its output and local derivative are then zero.

![Original hand-drawn matrix sketch showing a column of negative preactivations becoming an all-zero neuron after ReLU.](/images/notes/hands-on-ml/training-deep-networks/dying-relu.png)

An inactive neuron can sometimes recover if earlier layers change its inputs. A negative-side activation slope provides another way for gradients to continue flowing.

### Leaky ReLU, RReLU, and PReLU

For $0<\alpha<1$:

$$
\operatorname{LeakyReLU}_{\alpha}(z)
=\begin{cases}\alpha z,&z<0,\\z,&z\ge0.\end{cases}
$$

![Leaky ReLU with the original annotation marking its nonzero negative slope.](/images/notes/hands-on-ml/training-deep-networks/leaky-relu.png)

- **RReLU:** Randomizes the negative slope during training and uses its mean during evaluation.
- **PReLU:** Learns the negative slope by backpropagation.

These piecewise-linear functions have a kink at zero unless their two slopes coincide. Implementations use a derivative convention there; this does not by itself make them ineffective for gradient-based training.

### ELU and SELU

The exponential linear unit is:

$$
\operatorname{ELU}_{\alpha}(z)
=\begin{cases}\alpha(e^z-1),&z<0,\\z,&z\ge0.\end{cases}
$$

SELU scales an ELU with particular constants:

$$
\operatorname{SELU}(z)=\lambda\operatorname{ELU}_{\alpha}(z),
\qquad \alpha\approx1.67326,\quad\lambda\approx1.05070.
$$

![ELU and SELU curves, including the original annotation near the origin.](/images/notes/hands-on-ml/training-deep-networks/elu-selu.png)

ELU has matching first derivatives at zero when $\alpha=1$; standard SELU retains a kink there. SELU's self-normalizing behavior requires suitable conditions, including a fan-in variance of $1/n_{\mathrm{in}}$. In PyTorch, the [SELU documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.SELU.html) recommends `nonlinearity="linear"` when using Kaiming initialization for that purpose. Self-normalizing architectures use compatible choices such as `AlphaDropout` when dropout is needed.

### GELU, SiLU, and Swish

**GELU** uses the standard Gaussian cumulative distribution function $\Phi$:

$$
\operatorname{GELU}(z)=z\Phi(z),
\qquad \Phi(z)=\Pr(Z\le z),\quad Z\sim\mathcal{N}(0,1).
$$

For large positive inputs it approaches the identity; for large negative inputs it approaches zero.

**SiLU** and the more general **Swish** are:

$$
\operatorname{SiLU}(z)=z\sigma(z),
\qquad
\operatorname{Swish}_{\beta}(z)=z\sigma(\beta z).
$$

SiLU corresponds to $\beta=1$. In a generalized Swish implementation, $\beta$ can be fixed or learned.

### SwiGLU, Mish, and squared ReLU

**SwiGLU** multiplies two projected branches elementwise:

$$
\operatorname{SwiGLU}(X)
=\operatorname{SiLU}(XW_g+\mathbf{b}_g)
\odot(XW_v+\mathbf{b}_v).
$$

The value branch proposes features; the gate branch modulates them. This gate can suppress, pass, or change the sign of a value. The [GLU variants paper](https://arxiv.org/abs/2002.05202) describes this family of gated feed-forward transformations.

**Mish** uses softplus inside tanh:

$$
\operatorname{Mish}(z)=z\tanh(\operatorname{softplus}(z)),
\qquad \operatorname{softplus}(z)=\ln(1+e^z).
$$

**Squared ReLU** is:

$$
\operatorname{ReLU}^2(z)=\max(0,z)^2.
$$

Its first derivative is continuous at zero. It still has an inactive negative region, and large positive inputs grow quadratically.

## Batch and layer normalization

### Batch normalization

For one feature or channel, let $M$ be the number of values contributing to its training statistics. Batch normalization computes:

$$
\mu_B=\frac1M\sum_{i=1}^{M}x_i,
\qquad
v_B=\frac1M\sum_{i=1}^{M}(x_i-\mu_B)^2,
$$

$$
\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{v_B+\varepsilon}},
\qquad
y_i=\gamma\hat{x}_i+\beta.
$$

The stabilizer $\varepsilon>0$ prevents division by zero. The scale $\gamma$ and shift $\beta$ are **learned parameters**. Normalization can improve optimization and reduce sensitivity to initialization, but does not guarantee that all gradient problems disappear.

A common block is a linear transformation followed by normalization and an activation; placement depends on the architecture. Adding batch normalization does not remove the need for appropriate input preprocessing.

### Training and evaluation statistics

During training, batch normalization normally uses batch statistics and updates running estimates. With an averaging coefficient $\rho$:

$$
\mu_{\mathrm{run}}\leftarrow(1-\rho)\mu_{\mathrm{run}}+\rho\mu_B,
$$

$$
v_{\mathrm{run}}\leftarrow(1-\rho)v_{\mathrm{run}}+\rho s_B^2.
$$

In PyTorch, the forward normalization uses the biased variance $v_B$, while the running variance update uses the unbiased estimate $s_B^2=Mv_B/(M-1)$ when $M>1$. Evaluation normally uses the stored estimates, which avoids dependence on the current inference batch. This describes the default running-statistics behavior; see [`BatchNorm1d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html).

PyTorch's batch-normalization variants keep separate statistics per channel:

| Module | Typical input shape | Statistics pooled over |
| --- | --- | --- |
| `BatchNorm1d` | $(N,C)$ or $(N,C,L)$ | $N$, or $N$ and $L$ |
| `BatchNorm2d` | $(N,C,H,W)$ | $N,H,W$ |
| `BatchNorm3d` | $(N,C,D,H,W)$ | $N,D,H,W$ |

Small batches can give noisy statistics, and normalization adds computation even when it reduces the number of training epochs needed.

### Layer normalization

Layer normalization computes statistics within each example over selected feature dimensions. In PyTorch, `normalized_shape` specifies the trailing dimensions to normalize. For example, `LayerNorm(C)` on an $(N,L,C)$ tensor normalizes each token's $C$ features.

It uses input statistics in both training and evaluation, without batch-normalization running averages. See [`LayerNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).

## Gradient clipping

Clip gradients after `loss.backward()` and before `optimizer.step()` to limit extreme updates. It is useful in recurrent networks and other settings with exploding gradients, and can complement normalization.

**Norm clipping** rescales a gradient vector $\mathbf{g}$ when its norm exceeds a limit $c>0$:

$$
\mathbf{g}'=\mathbf{g}\min\!\left(1,\frac{c}{\lVert\mathbf{g}\rVert_2}\right),
$$

with a zero vector left unchanged. PyTorch's [`clip_grad_norm_()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) treats the supplied parameter gradients as one concatenated vector for the total norm, rather than independently clipping each parameter tensor.

**Value clipping** acts on each component separately:

$$
g_i'=\operatorname{clamp}(g_i,-c,c).
$$

It can change the gradient's direction. Assuming a model, optimizer, loss function, and training batch are prepared:

```python
model.train()
optimizer.zero_grad(set_to_none=True)
loss = loss_fn(model(inputs), targets)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

## Reusing pretrained layers

### Transfer learning

Transfer learning reuses a model trained on a related task. A common strategy is to replace its task-specific output layers, initially freeze the reused backbone, and then unfreeze selected layers for fine-tuning.

For an already constructed `backbone`:

```python
for parameter in backbone.parameters():
    parameter.requires_grad_(False)
```

Freezing parameters does not freeze batch-normalization running statistics. Choose the reused modules' training or evaluation modes deliberately. The useful number of reused layers depends on how closely the source and target tasks are related.

### Unsupervised and auxiliary-task pretraining

With abundant unlabeled data, an autoencoder or another representation-learning objective can provide pretrained layers. Reuse the learned representation, add a task-specific head, and fine-tune with labeled examples.

An auxiliary task can also provide inexpensive supervision. The original example is face-verification pretraining, such as a VGGFace2-based same-identity task, before adapting learned features to a target recognition problem with fewer labeled examples.

## Faster optimization

Let $\theta_t$ be the parameters after update $t$, $\eta$ the learning rate, and $g_t=\nabla J_t(\theta_{t-1})$ the current batch's loss gradient. Vector products, divisions, maxima, and square roots below act elementwise unless specified otherwise. Moment accumulators start at zero; optional weight decay is omitted until AdamW.

### Momentum

Momentum remembers previous update directions:

$$
v_t=\beta v_{t-1}+g_t,
\qquad
\theta_t=\theta_{t-1}-\eta v_t.
$$

Consistent gradients accumulate velocity, while alternating directions tend to cancel. A coefficient such as $\beta=0.9$ retains much of the previous velocity. Excessive momentum can also delay reactions to a changing gradient and contribute to overshooting.

![Ball moving through a valley as an intuition for momentum accumulating and dissipating motion.](/images/notes/hands-on-ml/training-deep-networks/momentum-intuition.png)

### Nesterov accelerated gradients

The classical lookahead interpretation evaluates a gradient where existing momentum is about to move the parameters:

$$
\theta_{\mathrm{look}}=\theta_{t-1}-\eta\beta v_{t-1},
$$

$$
v_t=\beta v_{t-1}+\nabla J_t(\theta_{\mathrm{look}}),
\qquad
\theta_t=\theta_{t-1}-\eta v_t.
$$

The lookahead gradient can correct the next step earlier, reducing some oscillation. This motivates Nesterov momentum; PyTorch exposes it through `SGD(..., momentum=0.9, nesterov=True)` using its documented update convention.

![Ordinary momentum and Nesterov momentum compared on a curved objective.](/images/notes/hands-on-ml/training-deep-networks/nesterov-momentum.png)

### AdaGrad

AdaGrad accumulates squared gradients separately for each parameter:

$$
s_t=s_{t-1}+g_t\odot g_t,
\qquad
\theta_t=\theta_{t-1}-\frac{\eta}{\sqrt{s_t}+\varepsilon}\odot g_t.
$$

Coordinates with larger accumulated squared gradients receive smaller effective learning rates. This can help adjust for different coordinate scales, but does not guarantee reaching a global minimum.

![AdaGrad and ordinary gradient descent following different trajectories through an elongated loss contour.](/images/notes/hands-on-ml/training-deep-networks/adagrad.png)

Because $s_t$ only increases, effective learning rates can become too small during long runs. AdaGrad can still be useful, especially for sparse problems; its suitability depends on the task.

### RMSProp

RMSProp replaces the ever-growing accumulator with an exponential moving average:

$$
s_t=\rho s_{t-1}+(1-\rho)g_t\odot g_t,
$$

$$
\theta_t=\theta_{t-1}-\frac{\eta}{\sqrt{s_t}+\varepsilon}\odot g_t.
$$

Older gradients gradually lose influence, avoiding AdaGrad's permanently accumulated denominator. The decay coefficient $\rho$ controls the averaging timescale.

### Adam and bias correction

Adam combines a first-moment average with a second-moment average:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t\odot g_t.
$$

Starting both accumulators at zero reduces their early magnitudes. Bias correction removes this zero-initialization factor:

$$
\hat{m}_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat{v}_t=\frac{v_t}{1-\beta_2^t}.
$$

The parameter update is:

$$
\theta_t=\theta_{t-1}
-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}.
$$

For the original example, a constant gradient $g_t=10$ and $\beta_1=0.9$ give $m_1=1$, $m_2=1.9$, and $m_3=2.71$. Dividing by $1-0.9^t$ gives $\hat{m}_t=10$ at each step.

![Original worked example showing how a zero-initialized exponential average approaches a constant gradient.](/images/notes/hands-on-ml/training-deep-networks/adam-bias-correction.png)

As $t$ grows, the correction factors approach one. For changing gradients, corrected moments remain moving averages rather than exact copies of the current gradient.

### AdaMax

AdaMax replaces Adam's squared-gradient accumulator with a decaying infinity-norm accumulator. In PyTorch's numerical convention:

$$
u_t=\max\!\left(\beta_2u_{t-1},\,|g_t|+\varepsilon\right),
$$

$$
\theta_t=\theta_{t-1}
-\frac{\eta}{1-\beta_1^t}\frac{m_t}{u_t}.
$$

The first-moment accumulator $m_t$ is the same as in Adam, and still needs its bias correction. See [`Adamax`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adamax.html).

### NAdam

NAdam combines adaptive moment estimates with a Nesterov-style correction. A simplified constant-momentum expression illustrating the idea is:

$$
\bar{m}_t=\beta_1\hat{m}_t
+\frac{(1-\beta_1)g_t}{1-\beta_1^t},
\qquad
\theta_t=\theta_{t-1}
-\eta\frac{\bar{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}.
$$

PyTorch's [`NAdam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.NAdam.html) uses a momentum schedule and corresponding correction factors. The simplified expression is an intuition, not a reproduction of every implementation detail.

### AdamW

AdamW applies weight decay separately from the adaptive loss-gradient update:

$$
\theta_t=(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}.
$$

The moment estimates use the loss gradient, while $\lambda$ controls the separate shrinkage. This distinction matters because an L2 penalty inside Adam's gradient is scaled by its adaptive machinery. The [decoupled weight decay paper](https://arxiv.org/abs/1711.05101) develops the AdamW formulation.

![Original qualitative comparison chart for SGD, momentum, adaptive optimizers, and AdamW.](/images/notes/hands-on-ml/training-deep-networks/optimizer-comparison.png)

*The original chart is an informal study summary. Actual convergence speed and generalization depend on the model, data, tuning, and computational budget.*

## Learning-rate scheduling

A schedule changes the learning rate during training. In this section, $t$ counts scheduler steps; whether one step means a batch or an epoch must match the implementation.

### Exponential scheduling

Multiplying by a fixed decay factor $0<\gamma<1$ at each scheduler step gives:

$$
\eta_t=\eta_0\gamma^t.
$$

A factor closer to one decays more slowly. PyTorch provides `ExponentialLR`.

### Cosine annealing

Over one interval $0\le t\le T_{\max}$, cosine annealing moves from $\eta_{\max}$ toward $\eta_{\min}$:

$$
\eta_t=\eta_{\min}
+\frac12(\eta_{\max}-\eta_{\min})
\left[1+\cos\!\left(\frac{\pi t}{T_{\max}}\right)\right].
$$

![Cosine and exponential learning-rate schedules compared over the same training interval.](/images/notes/hands-on-ml/training-deep-networks/cosine-vs-exponential.png)

Use `CosineAnnealingLR` with a deliberate interval and minimum rate. Cosine and exponential schedules are alternatives to compare on validation performance; neither is universally better.

### Performance-based scheduling

`ReduceLROnPlateau` monitors a metric, such as validation loss, and reduces the rate after improvement stalls for a chosen patience interval. For loss minimization, pass the current validation loss to `scheduler.step(val_loss)` after measuring it.

### Warmup

Warmup begins with a small learning rate and increases it over an initial interval. It can help when early activations, gradients, and optimizer state are still settling. It complements Adam's bias correction rather than replacing it, and is not required for every training problem.

### Cosine annealing with warm restarts

Warm restarts repeatedly raise the learning rate and begin a new cosine-decay interval. The higher rate can encourage exploration of a different region of the loss landscape. It restarts the schedule, not the network's weights.

![Repeated cosine learning-rate cycles with progressively longer intervals.](/images/notes/hands-on-ml/training-deep-networks/cosine-restarts.png)

`CosineAnnealingWarmRestarts` supports a cycle-length multiplier. A multiplier of two, for example, doubles successive cycle lengths.

### One-cycle scheduling

The one-cycle policy raises the learning rate to a maximum and then lowers it to a value much smaller than its starting rate. Cosine and linear annealing are common choices for the phases.

![One-cycle schedule with a rising phase followed by cosine decay, retaining the original phase annotations.](/images/notes/hands-on-ml/training-deep-networks/one-cycle.png)

PyTorch's [`OneCycleLR`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) updates after every training batch. Call `optimizer.step()` before `scheduler.step()`, and configure the total number of updates consistently with the training loop. Other schedulers may use different calling intervals.

## Avoiding overfitting through regularization

### L1, L2, and weight decay

An explicit regularized objective can take the form:

$$
J_{\mathrm{reg}}(\theta)
=J(\theta)+\lambda_1\lVert\theta\rVert_1
+\frac{\lambda_2}{2}\lVert\theta\rVert_2^2.
$$

For plain SGD without momentum, the L2 term yields multiplicative shrinkage:

$$
\theta_t=(1-\eta\lambda_2)\theta_{t-1}-\eta g_t.
$$

With adaptive optimizers, adding an L2 term to the loss generally differs from decoupled weight decay. AdamW implements the latter. See also the [likelihood note's explanation of priors as regularization](/notes/hands-on-ml/chapter-08/likelihood-function/#priors-as-regularization).

### Dropout

Dropout randomly masks activations during training, encouraging the network to rely less on particular combinations of features. It masks intermediate values rather than setting their stored weights to zero.

![Neural network with a random subset of hidden and input activations dropped, retaining the original annotations.](/images/notes/hands-on-ml/training-deep-networks/dropout.png)

Let $p\in[0,1)$ be the drop probability and $\mathbf{h}$ an activation vector. Sample a mask:

$$
m_i\sim\operatorname{Bernoulli}(1-p),
\qquad
\tilde{\mathbf{h}}=\mathbf{h}\odot\mathbf{m}.
$$

**Inverted dropout** scales the retained activations during training:

$$
\mathbf{h}_{\mathrm{drop}}=\frac{\mathbf{h}\odot\mathbf{m}}{1-p}.
$$

For example, if $p=0.5$ and $\mathbf{m}=[1,0,1,0]$, the unscaled masked vector is $[h_1,0,h_3,0]$ and the inverted-dropout output is $[2h_1,0,2h_3,0]$.

The conditional expectation is preserved:

$$
\mathbb{E}[h_{\mathrm{drop},i}\mid h_i]
=\frac{h_i}{1-p}\mathbb{E}[m_i]
=h_i.
$$

During ordinary evaluation, dropout becomes the identity operation. This is the convention used by [`nn.Dropout`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html). Dropout may slow fitting, and its probability should be chosen using validation rather than assuming it always improves the model.

### Monte Carlo dropout

**Monte Carlo (MC) dropout** keeps dropout active for repeated inference passes of the same input. Average the predictions and inspect their variation as an approximate measure of model uncertainty.

![Original MC-dropout illustration showing several predictions for one image and their average.](/images/notes/hands-on-ml/training-deep-networks/mc-dropout.png)

For classification, convert each pass's logits to probabilities before averaging. Variation between these predictions can help identify inputs that merit inspection or additional labels, but it does not guarantee calibrated uncertainty. The [MC-dropout paper](https://arxiv.org/abs/1506.02142) develops its approximate Bayesian interpretation.

For a multiclass classifier returning class logits and using `nn.Dropout`, with `model` and `inputs` prepared:

```python
model.eval()
for module in model.modules():
    if isinstance(module, nn.Dropout):
        module.train()

with torch.no_grad():
    probabilities = torch.stack([
        model(inputs).softmax(dim=-1) for _ in range(30)
    ])
mean_probability = probabilities.mean(dim=0)
probability_std = probabilities.std(dim=0)
```

This example uses 30 stochastic passes. It leaves batch normalization in evaluation mode while reactivating dropout. Restore ordinary evaluation with `model.eval()` afterward; other dropout module types need to be enabled explicitly if used.

### Max-norm regularization

Max-norm regularization constrains each neuron's incoming weight vector after an optimizer update:

$$
\lVert\mathbf{w}\rVert_2\le r,
\qquad
\mathbf{w}\leftarrow\mathbf{w}
\min\!\left(1,\frac{r}{\lVert\mathbf{w}\rVert_2}\right),
$$

with zero vectors left unchanged. This constrains **weights**, whereas gradient clipping constrains **gradients** before the update. In a PyTorch `Linear` layer, incoming weights for one output neuron form one row of the stored weight matrix.

## Practical starting points

The original notes suggest the following training choices. Treat them as candidates to validate for the actual task:

| Choice | Starting point or comparison |
| --- | --- |
| Initialization | Match the activation: He for ReLU-like layers; Xavier for suitable tanh layers; appropriate fan-in initialization for self-normalizing SELU |
| Activation | Compare ReLU with smooth or gated alternatives when the architecture calls for them |
| Normalization | Add batch or layer normalization where suitable for the architecture and batch structure |
| Regularization | Monitor validation performance; consider early stopping, weight decay, or dropout |
| Optimizer | Compare momentum or Nesterov SGD with an adaptive option such as AdamW |
| Learning-rate schedule | Consider a performance-based schedule or a deliberately configured one-cycle policy |

![Original practical-guidelines table covering initialization, activation, normalization, regularization, optimizer, and learning-rate scheduling.](/images/notes/hands-on-ml/training-deep-networks/practical-guidelines.png)

*The preserved table records the original study recommendations. Framework defaults depend on the module, and effective choices depend on validation results.*
