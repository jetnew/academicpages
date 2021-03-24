---
title: "Simkit"
excerpt: "Simkit, a framework for building generative and probabilistic models for training reinforcement learning agents, is my internship project at Grab in summer 2020, which I presented at Google Developer Space.<br/><img src='/images/simkit/preview.png'>"
date: 2020-08-01
collection: portfolio
---

# About Simkit

Simkit was my internship project at Grab in summer 2020, which I [presented](https://www.youtube.com/watch?v=wl_Z9URl6BU) at Google Developer Space. Read the full report [here](/files/sip-grab.pdf).

Simkit, or Environment Simulation Kit, is a generalized framework for building generative and probabilistic models which can be used to construct artificial environments for training reinforcement learning (RL) agents.

Simkit contains models, such as conditional generative feature models and probabilistic feature models, and utilities, such as metrics, visualisation and hyperparameter optimisation.

# Content

* Objective
* Models
  * Conditional Generative Feature Models
    * Gaussian Mixture Density Network
    * Conditional Generative Adversarial Network
  * Probabilistic Feature Models
    * Bayesian Neural Network
    * Monte Carlo Dropout
    * Deep Ensemble
* Utilities
  * Metrics
  * Visualisation
  * Hyperparameter Optimisation
* Evaluation
  * Synthetic Dataset: Gaussian
  * Industry Datasets: Wholesale & Supermarket

# Objective

The reinforcement learning (RL) framework consists of an agent learning from interactions with the environment. With every Action taken by the agent, the environment returns a State and a Reward.

Reinforcement learning is known to be unstable in training due to sample inefficiency and noisy observations/ reward functions, so Simkit proposes 2 types of models:

* Conditional generative feature models to model the distribution of States, and
* Probabilistic response models to model the distribution of Rewards, from the environment.

## How do the models improve RL?

The conditional generative feature models model the conditional dependencies among different features. For eg, a feature such as “Time taken before booking confirmation” could be dependent on another feature like “Age”, because people tend to be slower with age. In this case, we call the independent feature the parent feature, and the dependent feature the child feature.

The probabilistic response models model the distributional responses for stable learning of reinforcement learning agents. Previously, the agent may learn point estimates of responses which may be noisy. By learning the distributional responses, the model can learn the uncertainty that comes with each response.

# Models

## Conditional Generative Feature Models

In the Conditional Generative models, I implemented the Gaussian mixture density network and the conditional generative adversarial network.

* The GMDN is a model for representing normally distributed subpopulations within an overall population. 
  * A Gaussian mixture model is a universal approximator of densities, that any smooth density can be approximated by a Gaussian mixture model with enough components.
  * The GMM is parameterized by mixture component weights, means and covariances. For each input x, predict a probability density function of $P(Y=y\vert X=x)$ that is a probability weighted sum of smaller Gaussian distributions.
  * The conditional Gaussian mixture model can be implemented using Mixture Density Networks, consisting of a neural network to predict the parameters that define the Gaussian mixture model. For each input x, the MDN predicts a probability density function of $P(Y=y\vert X=x)$.
  * Each parameter pi, mu, sigma is approximated by a neural network as a function of input x, which represents the parent feature.

* The CGAN is a generative adversarial model that conditions on parent features for both generator and discriminator models during training.
  * The generative adversarial network is a deep generative model consisting of 2 neural networks, the generative network that generates candidates and the discriminative network that evaluates them, that contest with each other in optimisation.
  * The GAN framework has been popular in recent years for its performance in image generation, and conditioning features during training allows control over the generation process.
  * The conditional GAN is constructed by feeding the data to be conditioned on, parent features, to both the generator and discriminator.
  * GANs are known to be unstable in training and therefore a number of improvements are implemented. The Wasserstein loss is used for the models, label smoothing is performed, and a diversity penalty is attached to the discriminator model.

## Probabilistic Reponse Models

On the Probabilistic Response Model side of things:

* The Bayesian neural network are essentially neural networks but with weights that are assigned a probability distribution to estimate uncertainty, and trained via variational inference.
  * The Bayesian Neural Network, unlikely an ordinary neural network, has weights that are assigned a probability distribution to estimate uncertainty. 
  * Bayesian neural networks can be trained with variational inference by perturbing the weights, such as using flipout. Flipout decorrelates the gradients between different examples without biasing the gradient estimates.
  * The key focus of Bayesian neural networks is that it puts a prior distribution p(W) over the weights and approximates the posterior distribution $P(W\vert D) \propto p(W)p(D\vert W)$, where D denotes observed data.  The evidence lower bound (ELBO) is maximised in variational inference:

* Monte carlo dropout is a method of Bayesian approximation by performing T stochastic forward passes through the neural network.
  * The Monte Carlo (MC) Dropout has been shown to approximate to Bayesian inference in deep Gaussian processes. Dropout training with neural networks allows modelling of uncertainty without sacrificing either computational complexity or test accuracy.
  * During inference, MC Dropout performs T stochastic forward passes through the neural network with dropout applied.

* The Deep Ensemble is essentially an ensemble of randomly-initialised neural networks.
  * Deep Ensembles perform better than Bayesian neural networks in practice, particularly under dataset shift, because variational Bayesian methods tend to focus on a single mode, while deep ensembles tend to explore diverse modes in function space due to random initializations, mapping the loss landscape.
  * Outputs of different randomly-initialised models of the ensembles is combined by averaging in the implementation.

# Utilities

On the Utilities, I implemented performance metrics and visualisation which assess the results quantitatively and qualitatively.

## Performance Metrics

The performance metrics are computed empirically based on the data output.

* Kullback-Leibler Divergence
  * The Kullback-Leibler divergence is computed empirically by splitting the data into histogram bins before applying the formula. For multivariate data, data is splitted into multivariate bins.
* Jensen-Shannon Divergence
  * The Jensen-Shannon divergence is computed using empirical KL divergence.

The KL divergence is non-symmetrical while the JS divergence is symmetrical. One limitation of the metrics: Because the metrics are computed empirically by histograms, when there are a lot of non-overlapping histogram bins, the empirical values may be far from the actual divergences with values sometimes being 0 or even negative. So do check the overlapping bins before using the empirical metrics.

## Performance Visualisation
* The surface plot visualises the probability density function given axes of (X1, X2), and the probabilities at each point.
* The conditional probability plots the probability, given a particular X value, and a range of y values.
* The relative density plots the relative probability densities using a violin plot, by accepting data points (not probability).
* The conditional relative density plots relative densities between actual data and the fitted model, but specifies a fixed X value and a tolerance around it.
* The Grid violin plot is a visualisation of conditional relative densities.
  * In this case, X is 2-dimensional only, and y is 1-dimensional.
  * X and y axes of the grid represent how the 2D X variable changes, and in each plot, it represents the comparison of probability density between actual data and fitted model.

## Hyperparameter Optimization
For hyperparameter optimisation, the Ax framework by Facebook is used for its Bayesian optimisation process. Parameter & loss visualisation methods are also provided.

* Bayesian optimisation tunes parameters in relatively few iterations compared to grid search or global optimisation techniques.
* Bayesian optimization builds a smooth surrogate model of outcomes using Gaussian processes from the noisy observations from previous rounds of parameterizations. The surrogate model is used to make predictions at unobserved parameterizations and quantify uncertainty.

# Evaluation
I use a synthetic dataset based on normally distributed data for sanity testing, and the Wholesale Customers and Supermarket Sales dataset, which both involve predicting a price of some sort from parent features.

For each experimentation, I applied hyperparameter optimisation prior to testing.

* Synthetic Dataset
  * The synthetic dataset consists of 1000 samples, with 4 clusters of Gaussian-distributed data.
  * In summary, for the synthetic dataset, the GMDN performed the best, followed by CGAN and Deep Ensemble.
* Supermarket Sales Dataset
  * We use only 2 parent features, Supermarket Branch and Time of payment, and 1 child feature, Total spending.
  * In summary, the GMDN performed the best on the Supermarket dataset, with the Bayesian NN being reasonably well fit.
* Wholesale Customers Dataset
  * It has parent features of Channel and Region which are categorical variables, and child features which are continuous variables that seem to be exponentially distributed.
  * In summary, for the Wholesale Customers Dataset, the GMDN performed the best, followed by CGAN and Deep Ensemble.

# Summary

In summary, Simkit is a generalized framework for generative and probabilistic modelling for training reinforcement learning agents in TensorFlow 2.

I introduced the Gaussian mixture density network, Conditional generative adversarial network, Bayesian neural network, Monte carlo dropout and Deep ensemble, as well as some utility methods and evaluation experiments.

# References
1. Bishop, Christopher M. "Mixture density networks." (1994).
2. Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." arXiv preprint arXiv:1411.1784 (2014).
3. Blundell, Charles, et al. "Weight uncertainty in neural networks." arXiv preprint arXiv:1505.05424 (2015).
4. Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. 2016.
5. Lakshminarayanan, Balaji, Alexander Pritzel, and Charles Blundell. "Simple and scalable predictive uncertainty estimation using deep ensembles." Advances in neural information processing systems. 2017.
6. Fort, Stanislav, Huiyi Hu, and Balaji Lakshminarayanan. "Deep ensembles: A loss landscape perspective." arXiv preprint arXiv:1912.02757 (2019).
