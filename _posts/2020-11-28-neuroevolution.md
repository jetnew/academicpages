---
title: 'Evolutionary Computation: Neuroevolution (Part 2/3)'
date: 2020-11-28
excerpt: 'Neuroevolution is a method of applying evolutionary algorithms to optimise neural networks instead of using backpropagation. Neuroevolution therefore is a non-gradient (or derivation-free) optimisation, which can speed up training as backward passes are not computed. 2nd of a 3-part series on evolutionary computation.'
permalink: /posts/2020/11/neuroevolution/
tags:
  - evolutionary-computation
  - complex-systems
  - neural-networks
  - tutorial
---

# Neuroevolution

Neuroevolution is a method of applying evolutionary algorithms to optimise neural networks instead of using backpropagation. Neuroevolution therefore is a non-gradient (or derivation-free) optimisation, which can speed up training as backward passes are not computed. The neural network optimised by neuroevolution can be adapted in terms of parameters, hyperparameters or network architecture. Prominent examples of neuroevolution are NeuroEvolution of Augmenting Topologies (NEAT) and Covariance-Matrix Adaptation Evolution Strategy (CMA-ES). The evolutionary algorithm employed in this notebook is the vanilla genetic algorithm without crossing-over, applying only mutation over neural network parameters (weights). 2nd of a 3-part series on evolutionary computation (Part 1 - [Genetic Algorithm](https://jetnew.io/posts/2020/11/genetic-algorithm/), Part 3 - [Novelty Search](https://jetnew.io/posts/2020/11/novelty-search/)).


```python
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
```

# The Neural Network Model ("Neuro"-evolution)

The neural network, or a multi-layer perceptron, is a universal function approximator. The neural network in PyTorch with 2 hidden layers and non-linear activation functions hyperbolic tangent (tanh) and sigmoid is defined.


```python
net = nn.Sequential(
    nn.Linear(in_features=2, out_features=16, bias=True),
    nn.Tanh(),
    nn.Linear(in_features=16, out_features=1),
    nn.Sigmoid()
)

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

net = Net(2, 1)
```

# The Mutation Function (Neuro-"evolution")

As with the genetic algorithm, neuroevolution can be implemented by adding an additive Gaussian noise $\epsilon\sim N(0,\sigma)$ to all neural network weights to introduce variance in the "gene pool" of the population.


```python
from torch.nn.utils import parameters_to_vector, vector_to_parameters

def get_params(net):
  return parameters_to_vector(net.parameters())

def mutate_params(net, sigma=0.1):
    mutated_params = get_params(net) + torch.normal(0, sigma, size=get_params(net).data.shape)
    vector_to_parameters(mutated_params, net.parameters())

print(f"Before mutation:\n {get_params(net)}\n")
mutate_params(net, sigma=0.1)
print(f"After mutation:\n {get_params(net)}")
```

    Before mutation:
     tensor([-0.5949, -0.1591, -0.6217,  0.0710, -0.6687, -0.3964,  0.3319, -0.2988,
             0.6695,  0.4645,  0.2398, -0.2250, -0.5464,  0.2512,  0.0582,  0.0818,
             0.1810, -0.5316,  0.3275, -0.1162,  0.2542, -0.6751,  0.4344,  0.1846,
             0.4996, -0.1422,  0.3201, -0.0814, -0.1195,  0.1880, -0.2272, -0.4236,
            -0.0218, -0.6078, -0.0099,  0.1856, -0.4883, -0.2465,  0.0166, -0.1269,
             0.4119,  0.0229,  0.2381, -0.0007, -0.2959, -0.4865,  0.0240,  0.0228,
             0.2293, -0.0649, -0.1661,  0.0788,  0.2253, -0.1549, -0.2465, -0.0267,
            -0.1861, -0.2189,  0.0964,  0.0684, -0.0555,  0.0063,  0.1374, -0.1588,
             0.2334], grad_fn=<CatBackward>)
    
    After mutation:
     tensor([-0.6005, -0.2099, -0.5364,  0.1007, -0.5897, -0.5340,  0.3688, -0.3615,
             0.7335,  0.3900,  0.2519, -0.1144, -0.5839,  0.2251,  0.0043,  0.1630,
             0.1419, -0.6593,  0.3079, -0.1238,  0.3217, -0.7810,  0.4419,  0.3621,
             0.5246,  0.0064,  0.4284, -0.1177, -0.0700, -0.0537, -0.1281, -0.3613,
            -0.0873, -0.6996, -0.1507,  0.1944, -0.6326, -0.1384, -0.0384,  0.0323,
             0.3344,  0.0667,  0.1177, -0.1347, -0.3413, -0.5302, -0.1326,  0.3330,
             0.2282, -0.1485, -0.1944,  0.2058,  0.2997, -0.0631, -0.1202, -0.0973,
            -0.1269, -0.4766, -0.0509,  0.1725, -0.0470, -0.0562,  0.1357, -0.2274,
             0.3410], grad_fn=<CatBackward>)
    

# Optimization Problem: Circles Dataset

The optimization problem is the Circles dataset from Scikit-Learn, where the neural network model must learn to predict and discriminate between the inner circles (labelled 1) and outer circles (labelled 0). The Circles dataset is the reason that non-linear activation functions in the neural network architecture are needed. $X$ is 2-dimensional while $y$ is 1-dimensional.


```python
from sklearn.datasets import make_circles

def plot_data(X, y):
  X = X.detach().numpy()
  y = y.detach().numpy().flatten()
  plt.plot(X[y==0,0], X[y==0,1], '.', c='b', label='0')
  plt.plot(X[y==1,0], X[y==1,1], '.', c='r', label='1')

X, y = make_circles(n_samples=100)
X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float().view(-1, 1)

plot_data(X, y)
net(X[:5, :])
```




    tensor([[0.5009],
            [0.4840],
            [0.6110],
            [0.6143],
            [0.5136]], grad_fn=<SigmoidBackward>)




    
![png](/images/neuroevolution/output_7_1.png)
    


# Process 1: Generate the initial population of neural networks.

For illustration purposes, a small population size of 5 and 4 hidden units per neural network layer is used. Inspecting the first 2 neural networks in the population, neural network weights are randomly initialised. The specific initialisation method used for the weights is documented in the [PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) for interested readers.


```python
population_size = 5
initial_population = np.array([Net(2,1,n_hidden=4) for _ in range(population_size)])

for p in initial_population[:2]:
  print(get_params(p))
```

    tensor([ 0.5282,  0.4404, -0.6330, -0.1387, -0.2104,  0.5194,  0.3596,  0.2664,
             0.1044,  0.5380,  0.1583,  0.1221,  0.0324,  0.1239, -0.3698, -0.3658,
            -0.0879], grad_fn=<CatBackward>)
    tensor([ 0.3366, -0.6526, -0.1909, -0.4681, -0.3542,  0.2475,  0.1892, -0.1961,
            -0.2867, -0.4570, -0.3183, -0.2220,  0.1677,  0.2118,  0.1053,  0.3238,
             0.0737], grad_fn=<CatBackward>)
    

# Process 2: Compute the fitness of the population.

The fitness function measures the performance of an individual neural network. Because $y$ is a binary variable of values $\{0,1\}$, the negative binary cross entropy error (BCE) is employed, negated to reflect a higher value as more desirable.


```python
def fitness_function(net):
  return -nn.BCELoss()(net(X), y).detach().numpy().item()

def compute_fitness(population):
  return np.array([fitness_function(individual) for individual in population])

fitness_score = fitness_function(net)
fitness_scores = compute_fitness(initial_population)

fitness_score, fitness_scores
```




    (-0.7065134644508362,
     array([-0.69943392, -0.70006615, -0.70542192, -0.69766504, -0.76979303]))



# Process 3: Select the fittest neural networks.

Select the top $k$ percentage of neural networks with the highest fitness score to form the parent subpopulation.


```python
def solution(individual):
  return individual(X).view(-1).detach().numpy().round()

def get_fittest(population, fitness_scores):
  return population[fitness_scores.argmax()]

def select_fittest(population, fitness_scores, k=0.5):
  return population[np.argsort(fitness_scores)[-int(len(population) * k):]]

parent_subpopulation = select_fittest(initial_population, fitness_scores, k=0.4)
compute_fitness(parent_subpopulation)
```




    array([-0.69943392, -0.69766504])



# Process 4: Perform reproduction of the parents to replenish the population.

In contrast to common implementations of genetic algorithms, no crossing-over is performed. Parent neural networks are simply uniformly sampled with replacement to create an identical copy as the child.


```python
import copy

def perform_reproduction(subpopulation):
  num_children = population_size - len(subpopulation)
  parents = np.random.choice(subpopulation, num_children)
  return np.append(subpopulation, [copy.deepcopy(p) for p in parents], axis=0)

next_population = perform_reproduction(parent_subpopulation)
compute_fitness(next_population)
```




    array([-0.69943392, -0.69766504, -0.69943392, -0.69943392, -0.69766504])



# Process 5: Perform mutation on the population.

As explained previously, add a Gaussian noise perturbation to all parameters of the neural network.


```python
def get_population_parameter(population):
  return [get_params(net) for net in population]

def perform_mutation(population, sigma=0.1):
  for individual in population:
    mutate_params(individual, sigma=0.1)
  return population

print("Before mutation:")
print(get_population_parameter(next_population))

perform_mutation(next_population)

print("\nAfter mutation:")
print(get_population_parameter(next_population))
```

    Before mutation:
    [tensor([ 0.5282,  0.4404, -0.6330, -0.1387, -0.2104,  0.5194,  0.3596,  0.2664,
             0.1044,  0.5380,  0.1583,  0.1221,  0.0324,  0.1239, -0.3698, -0.3658,
            -0.0879], grad_fn=<CatBackward>), tensor([ 0.3565, -0.1857,  0.2187,  0.1788,  0.1412,  0.1778, -0.6750,  0.6518,
            -0.5023,  0.2402,  0.4160, -0.0343,  0.0818,  0.2978,  0.3582, -0.0858,
            -0.3332], grad_fn=<CatBackward>), tensor([ 0.5282,  0.4404, -0.6330, -0.1387, -0.2104,  0.5194,  0.3596,  0.2664,
             0.1044,  0.5380,  0.1583,  0.1221,  0.0324,  0.1239, -0.3698, -0.3658,
            -0.0879], grad_fn=<CatBackward>), tensor([ 0.5282,  0.4404, -0.6330, -0.1387, -0.2104,  0.5194,  0.3596,  0.2664,
             0.1044,  0.5380,  0.1583,  0.1221,  0.0324,  0.1239, -0.3698, -0.3658,
            -0.0879], grad_fn=<CatBackward>), tensor([ 0.3565, -0.1857,  0.2187,  0.1788,  0.1412,  0.1778, -0.6750,  0.6518,
            -0.5023,  0.2402,  0.4160, -0.0343,  0.0818,  0.2978,  0.3582, -0.0858,
            -0.3332], grad_fn=<CatBackward>)]
    
    After mutation:
    [tensor([ 0.6775,  0.5253, -0.6425,  0.0200, -0.2527,  0.5128,  0.4704,  0.1674,
             0.0913,  0.3902,  0.1255,  0.2492, -0.1159,  0.0734, -0.2054, -0.3601,
            -0.1615], grad_fn=<CatBackward>), tensor([ 0.4123, -0.2091,  0.3269,  0.2261,  0.2042,  0.2701, -0.7392,  0.5924,
            -0.5903,  0.0912,  0.2945, -0.2038,  0.0711,  0.1820,  0.3112, -0.0067,
            -0.3152], grad_fn=<CatBackward>), tensor([ 0.6215,  0.5592, -0.5993, -0.0947, -0.3291,  0.4888,  0.3596,  0.2436,
             0.2440,  0.4675, -0.0707,  0.0269,  0.0515,  0.0444, -0.3241, -0.3425,
            -0.1050], grad_fn=<CatBackward>), tensor([ 0.4444,  0.4725, -0.6944, -0.2502, -0.1153,  0.5679,  0.2696,  0.4509,
            -0.0502,  0.6541,  0.0673,  0.1718,  0.0901,  0.0092, -0.2689, -0.4209,
            -0.0223], grad_fn=<CatBackward>), tensor([ 0.2158, -0.1765,  0.0658,  0.1894,  0.0741, -0.1204, -0.7014,  0.5762,
            -0.3188,  0.3198,  0.5875, -0.0955,  0.0913,  0.2711,  0.3587, -0.0755,
            -0.4120], grad_fn=<CatBackward>)]
    

# The Neuroevolution Algorithm: All 5 Processes Together

By combining the 5 processes together, we construct the neuroevolution algorithm and run it to find a neural network solution that models the Circles dataset well.

Neuroevolution:
1. Generate the initial population of individuals.
2. Repeat until convergence:
	1. Compute fitness of the population.
	2. Select the fittest individuals (parent subpopulation).
	3. Perform reproduction between parents to produce children.
	4. Perform mutation on the population.
3. Select the fittest individual of the population as the solution.


```python
# Neuroevolution hyperparameters
population_size = 100
num_generations = 300
top_k = 0.1
mutation_sigma = 0.1
n_hidden = 16

# Process 1: Generate the initial population.
population = np.array([Net(2, 1, n_hidden) for _ in range(population_size)])

# Misc: Experimental tracking
scores = []
solutions = []
fittests = []

for i in range(num_generations):
  # Process 2: Compute fitness of the population.
  fitness_scores = compute_fitness(population)

  # Process 3: Select the fittest individuals.
  fittest_subpopulation = select_fittest(population, fitness_scores, k=top_k)
  
  # Misc: Experimental tracking
  fittest = get_fittest(population, fitness_scores)
  fittests.append(fittest)
  solutions.append(solution(fittest))
  scores.append(fitness_function(fittest))

  # Process 4: Perform reproduction between parents.
  children = perform_reproduction(fittest_subpopulation)

  # Process 5: Perform mutation on the population.
  population = perform_mutation(children, sigma=mutation_sigma)


# Misc: Experimental tracking
plt.plot(np.arange(num_generations), scores)
plt.show()
```


    
![png](/images/neuroevolution/output_19_0.png)
    


# Experiment Result

The background colours illustrate the neural network's decision boundary, while the individual data points are the original dataset. Looking at the fittest individual neural network of the final population, the non-linear decision boundary has been correctly and well-learnt by the fittest neural network in the final population.


```python
def plot_individual(net):
  x1 = np.arange(X[:,0].min()*1.2, X[:,0].max()*1.2, 0.01)
  x2 = np.arange(X[:,1].min()*1.2, X[:,1].max()*1.2, 0.01)
  X1, X2 = np.meshgrid(x1, x2)

  Y = np.zeros(X1.shape).flatten()
  for i, [x1, x2] in enumerate(zip(X1.flatten(), X2.flatten())):
    Y[i] = np.asarray(net(Variable(torch.Tensor([x1,x2])).float()).data)
  Y = Y.reshape(X1.shape)

  plt.xlim(min(X[:,0])*1.2, max(X[:,0])*1.2)
  plt.ylim(min(X[:,1])*1.2, max(X[:,1])*1.2)
  plt.contourf(X1, X2, Y, cmap='bwr', alpha=0.8)
  plt.colorbar()


fitness_score = fitness_function(fittest)
print(f"Fittest score: {fitness_score}")

plot_data(X, y)
plot_individual(fittest)
```

    Fittest score: -0.028863554820418358
    


    
![png](/images/neuroevolution/output_21_1.png)
    


By visualising the fittest model at each generation of neuroevolution, notice that the circular decision boundary is eventually found. For an evolutionary strategy based on novelty applied on reinforcement learning, refer to [Part 3](https://jetnew.io/posts/2020/11/novelty-search/) of the Evolutionary Computation series on Novelty Search. For an introductory treatment of the genetic algorithm, refer to [Part 1](https://jetnew.io/posts/2020/11/genetic-algorithm/).


```python
%%capture
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

plot_data(X, y)
ax.set_xlim(min(X[:,0]*1.2), max(X[:,0])*1.2)
ax.set_ylim(min(X[:,1]*1.2), max(X[:,1])*1.2)

x1 = np.arange(X[:,0].min()*1.2, X[:,0].max()*1.2, 0.01)
x2 = np.arange(X[:,1].min()*1.2, X[:,1].max()*1.2, 0.01)
X1, X2 = np.meshgrid(x1, x2)

def animate(i):
  net = fittests[i]
  Y = net(torch.Tensor(np.stack([X1.flatten(), X2.flatten()], axis=1))).detach().numpy().reshape(X1.shape)
  ax.contourf(X1, X2, Y, cmap='bwr', alpha=0.8)
  ax.set_xlabel(f'Gen {i+1}')

ani = FuncAnimation(fig, animate, frames=np.arange(0, num_generations), interval=80, repeat=False)
```


```python
ani.save('/images/neuroevolution/neuroevolution.gif')
```

    MovieWriter ffmpeg unavailable; using Pillow instead.
    

<img src="/images/neuroevolution/neuroevolution.gif">
