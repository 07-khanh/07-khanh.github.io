---
title: "Orthogonal Matrix"
date: "2026-08-25"
description: "Orthogonal and orthonormal vectors, the geometry of orthogonal matrices, and their role in machine learning."
---

# Orthogonal Matrices

---

## 1. Fundamentals of Orthogonality

### Orthogonal vs. Orthonormal Vectors
Two vectors are **orthogonal** (perpendicular) when their inner product is zero:

$$
u = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad v = \begin{bmatrix} 0 \\ 1 \end{bmatrix} \implies u^T v = (1)(0) + (0)(1) = 0 \implies u \perp v
$$

A set of vectors is **orthonormal** if it satisfies two conditions:
1. **Orthogonality:** Every pair of distinct vectors is perpendicular ($u_i^T u_j = 0$ for $i \neq j$).
2. **Unit Length:** Every vector has a Euclidean norm of 1 ($\|u_i\| = \sqrt{u_i^T u_i} = 1$).

---

### The Definition of an Orthogonal Matrix
An **orthogonal matrix** $Q$ is a square matrix whose columns form an orthonormal set.

Let $Q = \begin{bmatrix} q_1 & q_2 & \cdots & q_n \end{bmatrix}$. Computing $Q^T Q$:

$$
Q^T Q = 
\begin{bmatrix}
q_1^T \\ q_2^T \\ \vdots \\ q_n^T
\end{bmatrix}
\begin{bmatrix}
q_1 & q_2 & \cdots & q_n
\end{bmatrix}
=
\begin{bmatrix}
q_1^T q_1 & q_1^T q_2 & \cdots & q_1^T q_n \\
q_2^T q_1 & q_2^T q_2 & \cdots & q_2^T q_n \\
\vdots & \vdots & \ddots & \vdots \\
q_n^T q_1 & q_n^T q_2 & \cdots & q_n^T q_n
\end{bmatrix}
$$

Because $q_i^T q_i = 1$ and $q_i^T q_j = 0$ ($i \neq j$), the product simplifies directly to the identity matrix:

$$
\boxed{Q^T Q = I \iff Q^{-1} = Q^T}
$$

> **Key Property:** Inverting an orthogonal matrix requires no row reduction or matrix inversion algorithms—simply take its transpose.

---

## 2. Geometric Properties: Rigid Transformations

An orthogonal matrix acts as an **isometry**—a transformation that preserves geometric structure without stretching, compressing, or skewing space.

### Preservation of Lengths (Norms)
Applying $Q$ to any vector $x$ leaves its length unchanged:

$$
\|Qx\|^2 = (Qx)^T (Qx) = x^T (Q^T Q) x = x^T I x = x^T x = \|x\|^2 \implies \boxed{\|Qx\| = \|x\|}
$$

### Preservation of Angles and Distances
The dot product between any two transformed vectors $Qx$ and $Qy$ is invariant:

$$
(Qx)^T (Qy) = x^T (Q^T Q) y = x^T y
$$

Because both lengths and inner products are preserved, the angle $\theta$ between vectors remains identical:

$$
\cos \theta = \frac{(Qx)^T(Qy)}{\|Qx\| \|Qy\|} = \frac{x^Ty}{\|x\| \|y\|}
$$

---

### Rotations vs. Reflections
Every orthogonal transformation represents a **rotation**, a **reflection**, or a composition of both:

* **Pure Rotation ($\det(Q) = +1$):**
  $$
  Q_{\text{rot}} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
  $$
* **Pure Reflection ($\det(Q) = -1$):**
  $$
  Q_{\text{ref}} = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix} \quad (\text{flips the } y\text{-axis})
  $$

---

## 3. Applications in Linear Algebra and Machine Learning

```
[Input Vector x]
               │
               ▼
Rotate / Reflect via Vᵀ  (Isometry: preserves lengths)
│
▼

Stretch / Compress via Σ (Scaling along principal axes)
│
▼

Rotate / Reflect via U   (Isometry: aligns to output space)
│
▼
[Output Vector Ax]
```

### Geometric Interpretation of SVD ($A = U \Sigma V^T$)
Singular Value Decomposition factorizes any linear transformation into three distinct steps:

$$
A = \underbrace{U}_{\text{Orthogonal}} \cdot \underbrace{\Sigma}_{\text{Diagonal}} \cdot \underbrace{V^T}_{\text{Orthogonal}}
$$

1. **$V^T$ (Rigid Change of Basis):** Rotates/reflects the input domain without distorting lengths.
2. **$\Sigma$ (Axis-Aligned Scaling):** Applies all stretching and shrinking via singular values $\sigma_i$.
3. **$U$ (Rigid Alignment):** Rotates/reflects the scaled coordinates into the final output space.

---

### Coordinate Systems in PCA
In Principal Component Analysis, the projection matrix $V = \begin{bmatrix} v_1 & v_2 & \cdots & v_k \end{bmatrix}$ is constructed from the orthonormal eigenvectors of the sample covariance matrix:

$$
V^T V = I_k
$$

* **Decoupled Components:** Orthogonality guarantees that the resulting principal components $z_i = Xv_i$ are completely uncorrelated ($\text{Cov}(z_i, z_j) = 0$ for $i \neq j$).
* **Variance Preservation:** Because $V$ preserves lengths and dot products, projecting data onto the principal axes maintains the maximum possible geometric variance of the original dataset.

---

## 4. Core Takeaway

$$
\boxed{Q^T Q = I} \implies \boxed{Q^{-1} = Q^T} \implies \boxed{\text{Preserves lengths, angles, and distances}}
$$

In data transformations, **orthogonal matrices change orientation (viewpoint), while diagonal matrices change scale (importance).**
