---
title: "Perceptrons and Backpropagation"
date: "2026-09-29"
chapter: 9
chapterTitle: "Introduction to Artificial Neural Networks"
bookSlug: "hands-on-ml"
tags: ["machine-learning", "neural-networks", "perceptron", "multilayer-perceptron", "backpropagation", "activation-functions", "autodiff"]
description: "Threshold logic units, perceptron learning, dense-layer matrix shapes, multilayer networks, backpropagation, and activation functions, with the original diagrams."
---

An **artificial neural network (ANN)** combines numerical inputs, weighted connections, and activation functions. A perceptron is one of the simplest architectures; multilayer networks extend it by learning intermediate representations.

## The perceptron

### Threshold logic units

A **threshold logic unit (TLU)**, also called a **linear threshold unit (LTU)**, is an artificial neuron that:

- Receives numerical inputs.
- Assigns a weight to each input connection.
- Computes a weighted sum plus a bias, then applies a step function to produce a discrete output.

For input $\mathbf{x}\in\mathbb{R}^{d}$, weights $\mathbf{w}\in\mathbb{R}^{d}$, and bias $b$:

$$
z=\mathbf{w}^{\mathsf T}\mathbf{x}+b,
\qquad
h_{\mathbf{w},b}(\mathbf{x})=\operatorname{step}(z).
$$

The weighted sum plus bias is an affine function of the input. The step function turns that score into a threshold decision.

![Threshold logic unit combining weighted inputs and a bias before applying a step activation, with the original figure caption.](/images/notes/hands-on-ml/neural-networks/threshold-logic-unit.png)

### Common step functions

One convention for the **Heaviside step function** is:

$$
H(z)=\begin{cases}
0,&z<0,\\
1,&z\ge0.
\end{cases}
$$

The **sign function** is:

$$
\operatorname{sgn}(z)=\begin{cases}
-1,&z<0,\\
0,&z=0,\\
1,&z>0.
\end{cases}
$$

The value assigned exactly at the threshold is a convention. A binary classifier using sign-based labels must resolve the zero-score case consistently.

### A single dense layer

A perceptron can contain one layer of one or more TLUs, each connected to every input. Such a layer is **fully connected**, or **dense**.

![Perceptron architecture with two inputs connected to each of three output neurons.](/images/notes/hands-on-ml/neural-networks/perceptron-layer.png)

For a batch of $m$ instances, $d$ input features, and $n$ output neurons, use:

$$
X\in\mathbb{R}^{m\times d},
\qquad W\in\mathbb{R}^{d\times n},
\qquad \mathbf{b}\in\mathbb{R}^{n}.
$$

The layer's predictions are:

$$
\hat{Y}=\phi\!\left(XW+\mathbf{1}_m\mathbf{b}^{\mathsf T}\right)
\in\mathbb{R}^{m\times n},
$$

where $\phi$ acts elementwise and the bias is added to every row. For a TLU layer, $\phi$ is a step function.

> **Check matrix dimensions:** In this convention, the weight matrix has shape **input features × output neurons**. Its size does not depend on the batch's number of instances. PyTorch's [`nn.Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html) stores weights as **output features × input features** and computes the equivalent expression $XW_{\mathrm{torch}}^{\mathsf T}+\mathbf{b}$ using bias broadcasting.

## Perceptron learning

The perceptron uses a supervised error-correction rule related to Hebbian learning ideas. For one training example, let $y_j$ be the target of output neuron $j$ and $\hat{y}_j$ its prediction. Update the weight from input feature $r$ to output $j$ as:

$$
w_{rj}\leftarrow w_{rj}+\eta(y_j-\hat{y}_j)x_r,
\qquad
b_j\leftarrow b_j+\eta(y_j-\hat{y}_j),
$$

where $\eta>0$ is the learning rate. A correct prediction produces no update; an error moves the parameters in a direction that corrects that example.

Each TLU has a linear decision boundary. The classical perceptron convergence result guarantees finding a separating hyperplane in finitely many updates when the training data is linearly separable. For nonseparable data, a perfect separator does not exist in the original feature space, so that guarantee does not apply. See the [Cornell perceptron notes](https://www.cs.cornell.edu/courses/cs4780/2023sp/lectures/lecturenote03.html).

The classical threshold perceptron is mainly a foundation for understanding modern neural networks, rather than the usual architecture used for deep learning.

## Multilayer perceptrons

A **multilayer perceptron (MLP)** contains:

- An input layer representing the input features.
- One or more hidden layers that transform intermediate representations.
- An output layer producing the final predictions.

![Multilayer perceptron with two inputs, one hidden layer of four neurons, and an output layer of three neurons.](/images/notes/hands-on-ml/neural-networks/multilayer-perceptron.png)

Hidden layers with nonlinear activation functions allow the model to represent more complex relationships than a single linear threshold unit. Stacking affine transformations alone still gives an affine transformation; the nonlinear activations are essential.

The original diagram uses threshold-unit symbols to illustrate the connections. Networks trained using backpropagation generally replace those hard steps with activation functions suitable for gradient computation.

## Backpropagation and parameter updates

Training a neural network involves three distinct stages:

1. **Forward pass:** Compute the activations and predictions, then evaluate a loss $\mathcal{L}$ against the targets.
2. **Backward pass:** Use the chain rule to compute the loss gradients with respect to the trainable parameters. This is backpropagation, an application of reverse-mode automatic differentiation.
3. **Optimizer step:** Use those gradients to update the parameters. For gradient descent:

$$
\theta\leftarrow\theta-\eta\nabla_{\theta}\mathcal{L}.
$$

Here $\theta$ collects the weights and biases. The backward pass computes gradients; the optimizer uses them to change the parameters. PyTorch's [automatic differentiation tutorial](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) describes gradient computation through the recorded computation graph.

### The chain rule at one neuron

For a scalar neuron output $a=\phi(z)$ with $z=\mathbf{w}^{\mathsf T}\mathbf{x}+b$, the contribution of this path to a weight's loss gradient is:

$$
\frac{\partial\mathcal{L}}{\partial w_r}
=\frac{\partial\mathcal{L}}{\partial a}
\frac{\partial a}{\partial z}
\frac{\partial z}{\partial w_r}
=\frac{\partial\mathcal{L}}{\partial a}\,\phi'(z)\,x_r.
$$

Backpropagation passes the gradient from later computations through these local derivatives. When a value contributes through several paths, their gradient contributions are added.

> **Backpropagation and gradient descent have different roles:** Backpropagation computes derivatives. Gradient descent is one possible update rule that uses those derivatives.

## Activation functions and their derivatives

A hard step has derivative zero away from its threshold and is undefined at the jump. This makes it unsuitable for ordinary gradient-based learning through that activation.

Common alternatives include ReLU, sigmoid, and hyperbolic tangent:

| Activation | Function | Derivative |
| --- | --- | --- |
| ReLU | $\phi(z)=\max(0,z)$ | $0$ for $z<0$; $1$ for $z>0$ |
| Sigmoid | $\sigma(z)=1/(1+e^{-z})$ | $\sigma(z)(1-\sigma(z))$ |
| Tanh | $\tanh(z)=(e^z-e^{-z})/(e^z+e^{-z})$ | $1-\tanh^2(z)$ |

![Heaviside, ReLU, sigmoid, and tanh activations in the top row, with their derivatives below, retaining the original figure caption.](/images/notes/hands-on-ml/neural-networks/activation-functions.png)

*Read the activation curves in the top row and their derivatives in the bottom row.*

Sigmoid and tanh are smooth. ReLU is not differentiable at zero, so implementations use a derivative convention there. These choices provide useful gradients in parts of their input ranges, but do not guarantee nonzero gradients everywhere: ReLU has a flat negative region, while sigmoid and tanh can saturate.

## Further study

The original notes link to this [Micrograd notebook](https://github.com/chizkidd/Karpathy-Neural-Networks-Zero-to-Hero/blob/main/001_micrograd/micrograd.ipynb) as a companion reference for studying neural networks and backpropagation.
