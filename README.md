# Introduction to Machine Learning

This document combines the key concepts from the lectures of **NYU CS-GY 6923 Machine Learning**, simplified with clear explanations, formulas, and intuitions.

## Lecture 1: Introduction to Machine Learning

### 1. What is Machine Learning?

Machine Learning (ML) is the process of **teaching computers to learn patterns from data** rather than writing explicit rules.

* **Traditional Programming:** Rules + Data → Output
* **Machine Learning:** Data + Output → Rules (learned automatically)

### 2. The Prediction Problem

We aim to predict an output ( y ) from an input ( x ).

|      Symbol     | Meaning                      | Example                              |
| :-------------: | ---------------------------- | ------------------------------------ |
|      ( x )      | Input feature(s)             | Size of a house                      |
|      ( y )      | Output / Target              | Price of the house                   |
| ( f_\theta(x) ) | Model mapping input → output | ( f_\theta(x) = \beta_0 + \beta_1x ) |
|    ( \theta )   | Parameters to learn          | ( \beta_0, \beta_1 )                 |

Goal: Find parameters ( \theta^* ) such that:
[
\theta^* = \arg\min_\theta L(\theta)
]
where ( L(\theta) ) is the **loss function** measuring prediction error.

### 3. Supervised vs Unsupervised Learning

| Type              | Description                                | Example              |
| ----------------- | ------------------------------------------ | -------------------- |
| **Supervised**    | Learn from labeled data (x, y)             | Predict house prices |
| **Unsupervised**  | Find structure in unlabeled data           | Customer clustering  |
| **Reinforcement** | Learn by interacting and receiving rewards | Robotics / Game AI   |

### 4. Linear Regression (Simple Case)

Model:
[
\hat{y} = \beta_0 + \beta_1x
]

**Loss Function (Mean Squared Error):**
[
L(\beta_0, \beta_1) = \frac{1}{n}\sum_{i=1}^{n}(y_i - (\beta_0 + \beta_1x_i))^2
]

**Closed-form Solution:**
[
\beta_1 = \frac{\sigma_{xy}}{\sigma_x^2}, \quad \beta_0 = \bar{y} - \beta_1\bar{x}
]

where:
[
\sigma_{xy} = \frac{1}{n}\sum_i (x_i - \bar{x})(y_i - \bar{y}), \quad \sigma_x^2 = \frac{1}{n}\sum_i (x_i - \bar{x})^2
]

### 5. Empirical Risk Minimization (ERM)

We minimize **average loss** over data:
[
L(\theta) = \frac{1}{n}\sum_{i=1}^{n}\ell(f_\theta(x_i), y_i)
]
and find:
[
\theta^* = \arg\min_\theta L(\theta)
]

This is the foundation of supervised learning.

### 6. Generalization, Overfitting & Underfitting

* **Underfitting:** Model too simple → misses patterns.
* **Overfitting:** Model too complex → memorizes training data.
* **Generalization:** Performs well on new, unseen data.

We use **train/test splits** or **cross-validation** to measure generalization.

---

## Lecture 2: Multiple Linear Regression & Feature Transformations

### 1. Multiple Linear Regression

When output ( y ) depends on several features:
[
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_dx_d
]

**Matrix Form:**
[
\hat{y} = X\beta
]
where
( X \in \mathbb{R}^{n \times (d+1)} ), ( \beta \in \mathbb{R}^{(d+1)} ), ( y \in \mathbb{R}^n ).

### 2. Least Squares Loss Function

[
L(\beta) = |y - X\beta|^2 = (y - X\beta)^T(y - X\beta)
]

**Derivative:**
[
\nabla_\beta L(\beta) = -2X^T(y - X\beta)
]

Set derivative to zero:
[
X^TX\beta = X^Ty
]

**Normal Equation Solution:**
[
\boxed{\beta^* = (X^TX)^{-1}X^Ty}
]

### 3. Feature Transformations

To capture non-linear relationships, add new features:
[
y \approx \beta_0 + \beta_1x + \beta_2x^2 + \beta_3x^3
]

or use general basis functions:
[
\phi_1(x)=1, ; \phi_2(x)=x, ; \phi_3(x)=x^2, ; \phi_4(x)=\sin(x), ; \phi_5(x)=\cos(x)
]
Then:
[
\hat{y} = \sum_j \beta_j\phi_j(x)
]

✅ Still a linear model in ( \beta ) → same normal equation applies.

### 4. Handling Categorical Data – One-Hot Encoding

Convert text variables (e.g., `Ford`, `BMW`, `Honda`) into binary indicators:

| Brand | Ford | BMW | Honda |
| ----- | ---- | --- | ----- |
| Ford  | 1    | 0   | 0     |
| BMW   | 0    | 1   | 0     |
| Honda | 0    | 0   | 1     |

This avoids false numerical relationships between categories.

### 5. Model Selection & Overfitting

Adding features increases flexibility but also risk of overfitting.
To select the right complexity:

* **Train/Test Split:** Train on 80%, test on 20%.
* **K-Fold Cross Validation:** Average performance over K splits.

Choose the model with lowest **validation loss**.

### 6. Bias–Variance Tradeoff

| Type          | Behavior                   | Fix                                 |
| ------------- | -------------------------- | ----------------------------------- |
| High Bias     | Misses patterns (underfit) | Add features / lower λ              |
| High Variance | Too sensitive (overfit)    | Simplify model / add regularization |

---

## Lecture 3: Regularization & Logistic Regression

### 1. Motivation for Regularization

Overfitting often occurs when model weights become large.
Regularization **penalizes large weights** to keep the model simpler.

### 2. Ridge Regression (L2 Regularization)

**Loss Function:**
[
L(w) = |y - Xw|^2 + \lambda|w|^2
]

**Derivative:**
[
\frac{\partial L}{\partial w} = -2X^Ty + 2X^TXw + 2\lambda w
]

Set derivative to zero:
[
(X^TX + \lambda I)w = X^Ty
]

**Solution:**
[
\boxed{w^* = (X^TX + \lambda I)^{-1}X^Ty}
]

* ( \lambda > 0 ): regularization strength
* Large ( \lambda ) → smaller weights → smoother model.

### 3. LASSO Regression (L1 Regularization)

**Loss Function:**
[
L(w) = |y - Xw|^2 + \lambda \sum_i |w_i|
]

* Encourages sparsity (some weights become exactly zero).
* Useful for **feature selection**.

### 4. Comparison Table

| Type        | Formula                     | Shape    | Effect                   |
| ----------- | --------------------------- | -------- | ------------------------ |
| L2 (Ridge)  | ( |y-Xw|^2 + \lambda|w|^2 ) | Circular | Shrinks weights smoothly |
| L1 (Lasso)  | ( |y-Xw|^2 + \lambda|w|_1 ) | Diamond  | Some weights = 0         |
| Elastic Net | Combination of L1 + L2      | Blend    | Mix of both              |

### 5. Logistic Regression – For Classification

For binary labels ( y_i \in {0,1} ).

Model:
[
\hat{y} = \sigma(\beta_0 + \beta_1x)
]
where the **sigmoid function** is:
[
\sigma(z) = \frac{1}{1 + e^{-z}}
]

Outputs ( \hat{y} \in (0,1) ) interpreted as ( P(y=1|x) ).

### 6. Logistic Loss (Negative Log-Likelihood)

We assume:
[
P(y_i = 1|x_i) = \sigma(x_i^T\beta), \quad P(y_i = 0|x_i) = 1 - \sigma(x_i^T\beta)
]

The total likelihood:
[
L(\beta) = \prod_i [\sigma(x_i^T\beta)]^{y_i}[1 - \sigma(x_i^T\beta)]^{1-y_i}
]

Take log and negate (to minimize):
[
J(\beta) = -\sum_i [y_i\log \sigma(x_i^T\beta) + (1 - y_i)\log(1 - \sigma(x_i^T\beta))]
]

**With regularization:**
[
J(\beta) = -\sum_i [y_i\log \sigma(x_i^T\beta) + (1 - y_i)\log(1 - \sigma(x_i^T\beta))] + \lambda|\beta|^2
]

### 7. Gradient for Logistic Regression

The derivative of the logistic loss is:
[
\nabla J(\beta) = X^T(\sigma(X\beta) - y)
]

Used in **gradient descent**:
[
\beta \leftarrow \beta - \eta, \nabla J(\beta)
]
where ( \eta ) is the learning rate.

### 8. Summary: Logistic Regression

| Concept        | Explanation                             |
| -------------- | --------------------------------------- |
| Output         | Probability that y = 1                  |
| Sigmoid        | Maps any real number → (0,1)            |
| Loss           | Cross-entropy / Negative log-likelihood |
| Training       | Gradient descent (iterative)            |
| Regularization | L2 or L1 to reduce overfitting          |

### 9. Expected Loss (General Form)

For any probabilistic model with parameters (\theta):
[
R(\theta) = \mathbb{E}*{(x,y) \sim \mathcal{D}}[\ell(f*\theta(x), y)]
]
In practice, we use the **empirical approximation**:
[
\hat{R}(\theta) = \frac{1}{n}\sum_{i=1}^{n}\ell(f_\theta(x_i), y_i)
]
and minimize ( \hat{R}(\theta) ) (Empirical Risk Minimization).

---

## Final Takeaways

* **Machine Learning = Data-driven pattern discovery**
* **Supervised Learning** = Fit functions to labeled examples
* **Linear Regression** = Fit continuous values
* **Regularization** = Simplicity control
* **Logistic Regression** = Probability-based classification
* **Gradient Descent** = Universal optimization method
