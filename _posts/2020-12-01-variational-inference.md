---
title: 'Variational Inference: Deriving the Lower Bound on Log Likelihood from the Kullback-Leibler Divergence'
date: 2020-12-01
excerpt: 'Variational inference'
permalink: /posts/2020/12/variational-inference/
tags:
  - bayesian-learning
  - machine-learning
  - probabilistic-modelling
---

# Variational Inference
u
In variational inference, the true but intractable distribution $p^*(x)$, defined by the posterior distribution $p(x|D)$ over a set of unobserved variables $x = \{x_1 ... x_n\}$ given data $D$, is approximated by a tractable variational distribution $q(x)$:

$$p^*(x) = p(x|D) \approx q(x)$$

# Derivation from the Kullback-Leibler Divergence

To approximate $Q(\theta)$ to $P(\theta|X)$, the Kullback-Leibler (KL) divergence can be employed as the cost function:

$$KL(p^*||q) = \sum_x p^*(x) log \frac{p^*(x)}{q(x)}$$

However, since $p^*(x)$ is intractable, the reverse KL divergence is employed:

$$KL(q||p^*) = \sum_x q(x) log \frac{q(x)}{p^*(x)}$$

Because evaluating $p^*(x)=p(x|D)$ pointwise is hard due to the intractable normalization constant $p(D)$, it is replaced with the unnormalized distribution:

$$\tilde{p}(x)=p(x,D)=p^*(x)p(D)$$

in the objective function to be minimised:

$$J(q) = KL(q||\tilde{p})$$

$$=\sum_x q(x) log\frac{q(x)}{\tilde{p}(x)}$$

$$=\sum_x q(x) log\frac{q(x)}{p^*(x)p(D)}$$

$$=\sum_x q(x) log\frac{q(x)}{p^*(x)} - log p(D)$$

$$=KL(q||p^*)-log p(D)$$

Since $p(D)$ is a constant, by minimizing $J(q)$, $q$ approximates to $p^*$.

# Deriving the Lower Bound on Log Likelihood

Since KL divergence is non-negative, $J(q)$ is an upper bound on the negative log likelihood to be minimised:

$$J(q) = KL(q||p^*) - log p(D) \geq - log p(D)$$

Equivalently, maximising the negative of $J(q)$ derives a lower bound on the log likelihood of the data:

$$L(q) = -J(q) = -KL(q||p^*) + log p(D) \leq log p(D)$$

When $q=p^*$, the lower bound is tight and variational inference is closely related to expectation maximisation.
