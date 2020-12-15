---
title: 'A Note on Causal Inference'
date: 2030-12-03
excerpt: 'Causal inference is the inference of the effect of any treatment, policy or intervention based on the causal structures of the underlying process.'
permalink: /posts/2020/12/causal-inference/
tags:
  - causal-inference
  - bayesian-learning
  - statistics
---

# Causal Inference

Causal inference is the inference of the effect of any treatment, policy or intervention, the effect of $X$ on $Y$, based on the causal structures of the underlying process, e.g. inferring the effect of a treatment on a disease.

Having read 4 weeks of Brady Neal's Causal Inference course, the purpose of this note is to consolidate my learning and provide an easy way to revise causal inference concepts.

## Correlation v.s. Causation

Correlation does not imply causation. The mantra-like statement is engrained in students of science since young, but what is causation then? Let's illustrate the statement with an example: Sleeping with shoes on is strongly correlated with waking up with a headache, but sleeping with shoes on (as we know it) do not cause a headache when waking up. The reason for the association is the existence of a confounder variable (a common cause), drinking the night before, that confounds the association of shoe-sleepers to headaches. Correlation = Causation is a cognitive bias that is due to the availability heuristic and motivated reasoning, where people believe a correlated event as causal because the thought of the event is more available to the person (availability heuristic) and because people produce justifications that are most desired (motivated reasoning) ([Blanco & Matute, 2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4488611/)).

# Terminology

* Potential outcome: The effect of a treatment on some outcome, $Y_i\vert_{do(T=1)}=Y_i(1)$.
* Causal effect: $Y_i(1)-Y_i(0)$
* Individual treatment effect (ITE): $Y_i(1)-Y_i(0)$
* Average treatment effect (ATE): $E[Y_i(1)-Y_i(0)]=E[Y(1)]-E[Y(0)]\neq E[Y\vert T=1]-E[Y\vert T=0]$

# The Fundamental Problem of Causal Inference

Given a treatment $T$, if we perform $do(T=1)$, the potential outcome of $do(T=0)$, $Y_i(0)$ cannot be observed (because we cannot go back in time to repeat the opposite treatment on the same individual), and vice versa.

# Randomized Control Trials (RCTs)

When confounding exists, the ATE:

$$E[Y(1)]-E[Y(0)]\neq E[Y|T=1]-E[Y|T=0]$$

When confounding does not exist, the ATE:

$$E[Y(1)]-E[Y(0)]=E[Y|T=1]-E[Y|T=0]$$

Randomized Control Trials (RCTs) is the process where the experimenter randomizes subjects into the treatment or control group, resulting in the treatment $T$ to not have any causal parents, removing the confounding association to isolate the causal association.

# Measuring Causal Effects in Observational Studies

The average treatment effect (ATE) is not equal to the associational difference:

$$E[Y(1)]-E[Y(0)]\neq E[Y|T=1]-E[Y|T=0]$$

because the groups may be uncomparable due to a difference in ratios of confounder characteristics between groups. To compute the ATE based on the associational difference, the ATE must be equal the associational difference, and 4 assumptions must hold.

# Assumptions for ATE to Equal Associational Difference

1. Ignorability (exchangeability): The potential outcomes are independent to the treatment, $(Y(1),Y(0))\perp T$ allowing the causal quantity $E[Y(t)]$ to be identifiable from the statistical quantity $E[Y\vert T]$.
2. Positivity: The probability of treatment given the covariate $0<P(T=1\vert X=x)<1$ for all covariates is positive (and non-zero).
3. No interference: The potential outcome is a function of only the treatment, $Y_i(t_1,...,t_i,t_{i+1},...,t_n)=Y_i(t_i)$.
4. Consistency: The same treatment implies the same potential outcome, $T=t\rightarrow Y=Y(t)$.

When all 4 assumptions hold,

$$E[Y(1)-Y(0)] \tag{no interference}$$

$$=E[Y(1)]-E[Y(0)] \tag{linearity of expectation}$$

$$=E_X[E[Y(1)|x]-E[Y(0)|X]] \tag{law of iterated expectations}$$

$$=E_X[E|Y(1)|T=1,X]-E[Y(0)|T=0,X]] \tag{unconfoundedness and positivity}$$

$$E_X[E[Y|T=1,X]-E[Y|T=0,X]] \tag{consistency}$$

# Terminology

* Estimand: Any quantity to estimate.
* Causal estimand: e.g. $E[Y(1)-Y(0)]$.
* Statistical estimand: e.g. $E_X[E[Y\vert T=1,X]-E[Y\vert T=0,X]]$.
* Estimate: Approximation of estimand using data.
* Estimation: Process to approximate from data and estimand to the estimate.'

# Identification-Estimation Flowchart

Causal Estimand $\overset{Identification}{\rightarrow}$ Statistical Estimand $\overset{Estimation}{\rightarrow}$ Estimate

1. Identification

$$E[Y(1)-Y(0)]=E_X[E[Y|T=1,X]-E[Y|T=0,X]]$$

2. Estimation

$$\frac{1}{n}\sum_x[E[Y|T=1,x]-E[Y|T=0,x]]$$

# Local Markov Assumption

Given its parents in a Directed Acyclic Graph (DAG), a node $X$ is independent of all its non-descendants.

Bayesian network factorization:

$$P(x_1,..,x_n)=\Pi_i P(x_i|pa_i)$$

Local Markov Assumption $\leftrightarrow$ Bayesian Network Factorization

# Minimality Assumption

1. Local Markov Assumption: Permits distributions where $P(x,y)=P(x)P(y\vert x)$ and where $P(x,y)=P(x)P(y)$.
2. Adjacent nodes in the DAG are dependent ($X\rightarrow Y$). Does not permit distributions where $P(x,y)=P(x)P(y)$.

# Causal Edges Assumption
In a directed graph, every parent is a direct cause of all its children.

# Assumptions Flowchart
$\overset{Markov Assumption}{\rightarrow}$ Statistical Independencies $\overset{Minimality Assumption}{\rightarrow}$ Statistical Dependencies $\overset{Causal Edges Assumption}{\rightarrow}$ Causal Dependencies

Causal Dependencies: DAG + Markov Assumption + Causal Edges Assumption

# Basic Building Blocks of Graphs
* Unconnected nodes: $X_1,X_2$
* Connected nodes: $X_1\rightarrow X_2$
* Chain: $X_1\rightarrow X_2\rightarrow X_3$
* Fork: $X_1\leftarrow X_2\rightarrow X_3$
* Immorality: $X_1\rightarrow X_2\leftarrow X_3$
* In the graph $X_1\rightarrow X_2$, the local Markov assumption states that $X_1$ and $X_2$ are associated.

# Chains and Forks: Dependence
* $X_1\rightarrow X_2\rightarrow X_3$: $X_1$ is associated with $X_3$ (statistical dependence).
* $X_1\leftarrow X_2\rightarrow X_3$: $X_1$ is associated with $X_3$ because they share a common cause $X_2$.
* Association flows in the paths between $X_1$ and $X_3$.

# Chains and Forks: Independence
In both chains and forks as above, when conditioned as $X_2$, association is blocked from $X_1$ to $X_3$, making them conditionall independent (blocked path).

# Immoralities
* $X_1\rightarrow X_2\leftarrow X_3$: $X_1$ and $X_3$ are not associated (blocked path).
* $X_2$ is a collider and blocks association flow between $X_1$ and $X_3$.
* By conditioning on the collider, the path is unblocked to allow association flow.

# D-Separation
* Two (sets of) nodes $X$ and $Y$ are d-separated by a set of nodes $Z$ if all of the paths between (any node in) $X$ and (any node in) $Y$ are blocked by $Z$.
* Theorem: Given that probability distribution $P$ is Markov with respect to graph $G$, $X\perp_G Y\vert Z\rightarrow X\perp_P Y\vert Z$.
* Global Markov assumption: D-separation implies conditional independencies in the distribution ($\leftrightarrow$ Local Markov Assumption $\leftrightarrow$ Markov Assumption)

# Structural Causal Model

A structural equation defines $A$ as a cause of $B$, $B:=f(A)$.

A causal mechanism for $X_iL=f(A,B,...)$ where $A,B,...$ are direct causes of $X_i$. 

A structural causal model (SCM) is a collection of structural equations.

$$M: (B:=f_B(A,U_B), C:=f_C(A,B,U_C), D:=f_D(A,C,U_D)）$$

Endogenous variables are endogenous to the model as their causal mechanisms are modelled, while exogenous variables are exogenous to the model as their causes are not modelled (and do not have parents).

A SCM is a tuple of the following sets:
* A set of endogenous variables
* A set of exogenous variables
* A set of functions to generate each endogenous variable as a function of other variables.

# Intervention

* SCM (Model) - $M: (T:=f_T(X,U_T), Y:=f_Y(X,T,U_Y))$
* Interventional SCM (Submodel) - $M_t: (T:=t, Y:=f_Y(X,T,U_Y))$
* Modularity assumption for SCMs: Consider an SCM $M$ and an interventional SCM $M_t$ obtained by performing the intervention $do(T=t)$. The modularity assumption states that $M$ and $M_t$ share all their structural equations except the structural equation for $T$, which is $T:=t$ in $M_t$.

# Collider Bias from Conditioning on Treatment Descendants

* By not conditioning on descendants of the treatment variable, the collider bias on post-treatment covariates can be avoided.
* However, collider bias can still be observed by inducing pre-treatment association, known as the M-bias.
* To avoid M-bias, do not condition on the relevant covariate by referencing the causal graph.

# References
* Brady Neal [Causal Inference Course](https://www.bradyneal.com/causal-inference-course)