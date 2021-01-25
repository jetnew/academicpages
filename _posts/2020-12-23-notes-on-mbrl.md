---
title: 'Notes on Model-Based Reinforcement Learning'
date: 2020-12-23
excerpt: 'Notes on Model-Based Reinforcement Learning (MBRL) as part of my Undergraduate Research Opportunities Programme (UROP) in NUS.'
permalink: /posts/2021/01/notes-on-mbrl/
tags:
  - research
  - reinforcement-learning
---

# Notes on Model-Based Reinforcement Learning

Model-based reinforcement learning (MBRL) has been rising in popularity for being the state-of-the-art in sample efficiency and asymptotic performance. However, MBRL still faces many open problems that are actively being researched. For a quick read on the survey, refer to [Model-based Reinforcement Learning: A Survey](https://arxiv.org/abs/2006.16712) and [Deep Model-Based Reinforcement Learning for High-Dimensional Problems, a Survey](https://arxiv.org/abs/2008.05598) for a deep learning treatment. For a comparison between MBRL and model-free reinforcement learning (MFRL) algorithms across various environments and settings, refer to [Benchmarking Model-Based Reinforcement Learning](https://arxiv.org/abs/1907.02057)

# Model-Based Reinforcement Learning (MBRL)

Model-based reinforcement learning learns a model of the environment that the agent interacts in, which can be used by the agent for generation of training data, or long-term planning.

# World Models - PlaNet - Dreamer

## World Models

Ha & Schmidhuber published [World Models](https://arxiv.org/abs/1803.10122) in 2018, which inspired interest in MBRL.

<img src="https://worldmodels.github.io/assets/world_model_schematic.svg">

World Models employ a model that learns the dynamics of the environment and an agent policy that learns the actions that maximises expected cumulative reward. The dynamics model is a combination of a variational autoencoder (VAE) that compresses the image-based observation into a lower-dimensional latent vector, and a mixture density network-recurrent neural network (MDN-RNN) that remembers past temporal information. The policy is a simple linear function that is optimised by evolutionary strategy so that an analytic solution to the optimisation problem is not needed. The environment model and the agent policy are trained separately, like in the Dyna set-up.

## PlaNet

WIP

## Dreamer

WIP

# Modelling Dynamics in MBRL

Modelling environment dynamics is a common approach in MBRL, which can be learnt either separately from the agent training (as in the Dyna set-up) or coupled in policy search. Dynamics modelling are modelled in 3 directions: Forward, backward and inverse models.

## Directional Dynamics Modelling

Directional dynamics modelling refers to the direction by which environment dynamics are being modelled. Given a trajectory consisting of tuples of transitions of the form $(s_t, a_t, s_{t+1})$, models of the environment can be learnt by predicting forward, backward or inverse. The forward model learns $P(s_{t+1}|s_t,a_t)$, predicting the next state. The backward model learns $P(s_t|s_{t+1},a_t)$ inferring the previous state given the next state. The inverse model learns $P(a_t|s_t,s_{t+1})$, inferring the action required to transition from current to next state.

## Planning: Shooting and Collocation

WIP

