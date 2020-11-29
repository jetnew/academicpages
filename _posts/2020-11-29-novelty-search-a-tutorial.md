---
title: 'Evolutionary Computation: Novelty Search (Part 3/3)'
date: 2020-11-29
excerpt: 'Novelty Search is an Evolutionary Strategy (ES) algorithm that optimises using a novelty function instead of a fitness function (like in a vanilla genetic algorithm), which has shown to produce competitive performance for exploration in reinforcement learning. 3rd of a 3-part series on evolutionary computation.'
permalink: /posts/2020/11/novelty-search/
tags:
  - evolutionary-computation
  - complex-systems
  - reinforcement-learning
  - tutorial
---

# Novelty Search

Novelty Search is an Evolutionary Strategy (ES) algorithm that optimises using a novelty function instead of a fitness function (like in a vanilla genetic algorithm), which has shown to produce competitive performance for exploration in reinforcement learning. The novelty of a solution is defined by how similar the solution's behaviour is as compared to the rest of the population. The novelty score is therefore computed by its average distance from the k-nearest neighbours in the population. 3rd of a 3-part series on evolutionary computation (Part 1 - [Genetic Algorithm](https://jetnew.io/posts/2020/11/genetic-algorithm/), Part 2 - [Neuroevolution](https://jetnew.io/posts/2020/11/neuroevolution/)).


```python
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
```

# Neuroevolution for Reinforcement Learning

Neuroevolution is the application of evolutionary strategies to neural networks. We use a simple neural network in PyTorch, with 2 linear layers and 2 non-linear activation functions tangent and sigmoid. In deep reinforcement learning, the neural network serves as function mapping from the observation of the environment to an action chosen by the agent. Over one episode, the agent performs an action and the state of the environment is observed by the agent, along with a reward at that particular timestep. The fitness of an individual neural network is therefore defined by the cumulative reward obtained by the agent over one episode of interacting with the environment. The environment used is [CartPole-v1](https://gym.openai.com/envs/CartPole-v1/), where the agent's goal is to balance the pole by pushing the cart. The state observed by the agent is defined as:

$$Observation = [Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity]$$

where the range of values are:

$$Cart Position = [-4.8,4.8]$$
$$Cart Velocity = [-Inf, Inf]$$
$$Pole Angle = [-24 degrees, 24 degrees]$$
$$Pole Angular Velocity = [-Inf, Inf]$$

and the action is a single scalar discrete value:

$$Action = [0, 1]$$

where $0$ and $1$ represents the action of pushing the cart to the left and right respectively.


```python
class Net(nn.Module):
  def __init__(self, input_size, output_size, n_hidden=16):
    super(Net, self).__init__()
    self.linear1 = nn.Linear(input_size, n_hidden, bias=True)
    self.tanh1 = nn.Tanh()
    self.linear2 = nn.Linear(n_hidden, output_size)
    self.sigmoid = nn.Sigmoid()
  def forward(self, x):
    x = self.linear1(x)
    x = self.tanh1(x)
    x = self.linear2(x)
    x = self.sigmoid(x)
    return x


def get_action(net, obs):
  return net(torch.from_numpy(obs.copy()).float()).detach().numpy().argmax()

def evaluate(net):
  obs = env.reset()
  done = False
  total_reward = 0
  while not done:
    action = get_action(net, obs)
    obs, reward, done, _ = env.step(action)
    total_reward += reward
  return total_reward

def fitness_function(net, episodes=1):
  return np.mean([evaluate(net) for _ in range(episodes)])

def compute_fitness(population):
  return np.array([fitness_function(individual) for individual in population])

def get_fittest(population, fitness_scores):
  return population[fitness_scores.argmax()]
```

# Novelty Selection

The main difference between neuroevolution and novelty search is the selection criterion, changed from the fitness score to the novelty score. Instead of selecting for the fittest individuals in a population, novelty search selects the most novel individuals by the novelty score with respect to the rest of the population. The novelty score indicates the novelty of an individual, defined as the average difference of an individual $\pi$ to $k$-nearest neighbours in the population, notably in terms of their behaviour. Therefore, the behaviour of an individual $b(\pi_i)$ must be defined. We employ a simple characterisation of a neural network's behaviour as the terminal (final) state $s_n$ in the sequence of states observed by the agent $S_{\pi_i} = [s_1, s_2, ..., s_n]$ in 1 evaluation:

$$Behaviour(\pi_i) = Terminal(S_{\pi_i}) = s_{n}$$

The similarity between 2 individuals' behaviours is simply the sum of squared difference between final observations:

$$Similarity(\pi_i, \pi_j) = ||Behaviour(\pi_i) - Behaviour(\pi_j)||$$

The novelty of an individual with respect to its $k$-nearest neighbours of the population $P$ is defined by:

$$Novelty(\pi_i, N_{\pi_i}) = \frac{1}{|N_{\pi_i}|} \sum_{\pi_k\in N_{\pi_i}}Similarity(\pi_i, \pi_k)$$

where $N_{\pi_i}$ refers to the $k$-nearest neighbours of $\pi_i$. The $k$-nearest neighbours $N_{\pi_i}$ are selected by the $k$ largest similarity scores between $\pi_i$ and $\pi_{k}\in N_{\pi_i}$.


```python
def behaviour(net):
  obs = env.reset()
  done = False
  while not done:
    action = get_action(net, obs)
    obs, reward, done, _ = env.step(action)
  return obs

def similarity(net1, net2):
    b1, b2 = behaviour(net1), behaviour(net2)
    return np.sum((b1 - b2)**2)

def compute_novelty(population, k=3):
    distances = []
    n = len(population)
    for i in range(n):
        distance_i = sorted([similarity(population[i], population[j]) for j in range(n) if i != j])[:k]
        distances.append(np.mean(distance_i))
    return distances

def get_novel_subpopulation(population, novelty_scores):
  return population[novelty_scores.argmax()]

def select_most_novel(population, novelty_scores, k=0.5):
  return population[np.argsort(novelty_scores)[-int(len(population) * k):]]
```

# Perform Reproduction

As with neuroevolution, reproduction among the novel parent individuals is performed by simply sampling then making a copy of the parent individual to form child individuals, replenishing the population. There is no change from the neuroevolution implementation in [Part 2](https://jetnew.io/posts/2020/11/neuroevolution/).


```python
import copy

def perform_reproduction(subpopulation):
  num_children = population_size - len(subpopulation)
  parents = np.random.choice(subpopulation, num_children)
  return np.append(subpopulation, [copy.deepcopy(p) for p in parents], axis=0)
```

# Perform Mutation

As with neuroevolution, mutation is performed by applying an additive Gaussian noise to the parameters of the neural networks. There is no change from the neuroevolution implementation in [Part 2](https://jetnew.io/posts/2020/11/neuroevolution/).


```python
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def get_params(net):
  return parameters_to_vector(net.parameters())

def mutate_params(net, sigma=0.1):
    mutated_params = get_params(net) + torch.normal(0, sigma, size=get_params(net).data.shape)
    vector_to_parameters(mutated_params, net.parameters())

def perform_mutation(population, sigma=0.1):
  for individual in population:
    mutate_params(individual, sigma=0.1)
  return population
```

# The Novelty Search Algorithm

By selecting the most novel individuals over generations, the individuals in the population will find its behavioural niche, improving exploration in the behaviour space.

Novelty Search:
1. Generate the initial population of individuals.
2. Repeat until convergence:
  1. Compute novelty of the population.
  2. Select the most novel individuals to form the parent subpopulation.
  3. Perform reproduction between parents to produce children.
  4. Perform mutation on the population.
    
3. Select the fittest individual of the population as the solution.


```python
# Novelty Search hyperparameters
population_size = 20
num_generations = 30
top_k = 0.2
mutation_sigma = 0.1
k_nearest = 3

# CartPole environment initialisation
env = gym.make('CartPole-v1')

# Neural network hyperparameters
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
n_hidden = 16

# Process 1: Generate the initial population.
population = np.array([Net(input_size, output_size, n_hidden) for _ in range(population_size)])

# Misc: Experimental tracking
scores = []
fittests = []

for i in range(num_generations):

  # Process 2: Compute the novelty of individuals with respect to closest neighbours in the population.
  novelty_scores = compute_novelty(population, k=k_nearest)
    
  # Process 3: Select the most novel individuals.
  novel_subpopulation = select_most_novel(population, novelty_scores, k=top_k)

  # Misc: Experimental tracking
  fitness_scores = compute_fitness(population)
  fittest = get_fittest(population, fitness_scores)
  fittests.append(fittest)
  scores.append(max(fitness_scores))

  # Process 4: Perform reproduction between parents.
  children = perform_reproduction(novel_subpopulation)
  population = perform_mutation(children, sigma=mutation_sigma)


# Misc: Experimental tracking
plt.plot(np.arange(num_generations), scores)
plt.show()
```


    
![png](/images/novelty-search/output_11_0.png)
    


# Experiment Results

Plotting the novelty score against fitness score for the final population, the novelty score defined by the $k$-nearest neighbour similarity of terminal states is not linearly correlated with a high fitness score. It is important that the novelty score is not linearly correlated with the fitness score because a linear combination of the fitness score would have been used for selection, defeating the purpose of novelty-based search.


```python
plt.xlabel("Novelty Score")
plt.ylabel("Fitness Score")
plt.scatter(novelty_scores, fitness_scores)
plt.show()
```


    
![png](/images/novelty-search/output_13_0.png)
    


Visualising an episode of the fittest individual shows that the novelty search algorithm has successfully achieved the goal of CartPole-v1 of balancing the pole. For an introductory treatment of the genetic algorithm, refer to [Part 1](https://jetnew.io/posts/2020/11/genetic-algorithm/). For an introductory treatment of neuroevolution, refer to [Part 2](https://jetnew.io/posts/2020/11/novelty-search/).


```python
%%capture
from matplotlib.animation import FuncAnimation

def get_frames(net, episodes=3):
  frames = []
  for i in range(episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
      action = get_action(net, obs)
      obs, reward, done, _ = env.step(action)
      frames.append(env.render(mode='rgb_array'))
  env.close()
  return frames

frames = get_frames(fittest, episodes=3)
fig, ax = plt.subplots()
screen = plt.imshow(frames[i])

def animate(i):
  screen.set_data(frames[i])

ani = FuncAnimation(fig, animate, frames=np.arange(0, len(frames)), interval=80, repeat=False)
```


```python
ani.save('/images/novelty-search/novelty_search.gif')
```

    MovieWriter ffmpeg unavailable; using Pillow instead.
    

<img src="/images/novelty-search/novelty_search.gif">
