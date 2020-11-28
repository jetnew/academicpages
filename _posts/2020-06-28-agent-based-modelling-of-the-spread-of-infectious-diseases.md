---
title: 'Agent-Based Modelling of the Spread of Infectious Diseases: Compliance on Mask-Wearing and its Consequences on Policies'
date: 2020-06-28
author: New Jun Jie, Lim Wei Liang, Gwendolyn Yong En Ling, Tan Pei Han
permalink: /posts/2020/06/agent-based-modelling-of-the-spread-of-infectious-diseases/
tags:
  - agent-based-modelling
  - simulation
  - complex-systems
---

Abstract
======
The objective of this study is to develop an agent-based modelling framework to simulate the spread of infectious diseases in order to investigate the extent to which mask-wearing is beneficial in the context of different policy interventions. The simulation is coded in Python 3 using NumPy, Pandas and Matplotlib. 6 different experiments are performed in the simulation: An experimental control, a mask effectiveness condition, an improper mask handling condition, an improved hygiene policy, a lockdown policy and an aggressive testing policy. The simulation framework and experimental data are made available: https://github.com/jetnew/COVID-Mask-Policy-Simulator. Regression is performed on the data, in which the difference in transmission rate from different percentages of mask wearers in the population is analysed against the extent of the policy intervention. The results show that the practice of mask-wearing allows mask wearers to benefit from a range of policy interventions to a greater extent than non-mask wearers.

Table of Content
======
1. Introduction and Motivation
2. Research Questions
3. Research Methodology
4. Data Analysis
5. Results and Discussion
6. Limitations and Future Directions
7. Conclusion
8. References
9. Presentation Deck

Introduction and Motivation
======
In light of COVID-19, countries around the world have taken various preventive measures, e.g. movement control orders and promotion of hand hygiene in hopes to stall the spread of the virus (BBC). As a unique strain of the coronavirus, its basic reproduction number (R0), a measure of the infectivity of the disease, appears to range from 2.28 (Zhang) to 5.7 (Sanche), much higher than other strains of coronavirus — the overall R0 of MERS-CoV is < 1(World Health Organisation), while that of SARS was pegged at approximately 3(World Health Organisation Department Of Communicable Disease Surveillance And Response). As of 26 April 2020, more than 2.9 million people have been infected across nearly the entire world, and the death toll stands at a staggering 200 thousand (Worldometer).
Amidst the flurry of COVID-19, mask-wearing has gained traction as a measure to prevent the virus’s spread. Even in Singapore, the policy on mask-wearing has taken root — from discouraging the panic-buying of masks to currently, the legal enforcement of mask-wearing in public (Toh). While mask-wearing is readily adopted in certain countries, especially those severely affected by the 2003 SARS outbreak (Wong), non-mask wearers cite political reasons for their refusal.
Existing literature on infectious diseases generally demonstrate the effectiveness of mask-wearing (Cowling). Homemade masks have also been shown to provide more protection than none at all, even if they possess less filtration efficacy than surgical masks (Davies). For COVID-19, given the high number of asymptomatic and pre-symptomatic cases, mask-wearing should be encouraged even for healthy individuals and implemented on a wide scale. Symptoms take up to 11.5 days to surface; mask-wearing prevents the spread of disease here when patients are unaware that they have contracted the virus.
However, opponents list two main arguments against the adoption of widespread mask-wearing: ineffectiveness and inefficient allocation (Li). This is not without cause; not all masks (homemade or surgical) fit well or have high filtration efficacy (Davies). This could be further hampered by improper usage. Furthermore, the widespread demand for surgical masks poses a threat to frontline healthcare workers who find themselves low on supplies (Lacina).
A third argument against the adoption of masks may prove more significant: mask-wearing promotes risky-behaviour. Mask-wearing may raise awareness of better hygiene, but it may similarly also “engender a false sense of security” (Brienen). One experiment compared two groups’ adherence to hand hygiene policies, one with face masks and one without; in the former, mask-wearing appeared to have slightly decreased the overall adherence to hand hygiene intervention (Cowling). This suggests that mask-wearing policies should be adopted in tandem with other policies to counterbalance potential risky behaviours. Proponents of mask-wearing argue that the effectiveness of masks is not only linked to its timely, consistent and correct use, but also when it is implemented simultaneously with other personal hygiene measures (Bin-Resa).
With limited resources for mask-supply, it is imperative to determine how and if mask-wearing should complement other management strategies, notably lockdowns, aggressive testing and hygiene promotion. To examine the complex behavioural patterns of mask wearers vis-a-vis non-mask wearers, agent-based modelling (ABM) can be employed where each person is modelled as an individual agent that makes decisions based on individual assessments (Bonabeau). ABMs are commonly used for modelling the effects of infectious diseases (Koo). Through ABM, we hope to replicate the dynamics of a real-world population and observe how a population’s compliance with mask wearing affects the spread of a disease in the context of other real-world policy interventions.

Research Questions
======
We utilized agent-based modeling to investigate the extent to which mask-wearing may be beneficial in the context of different policy interventions. While existing literature generally demonstrates that masks reduce dispersion of respiratory droplets, there is little focus on how mask-wearing works in tandem with other disease management policies. This research project aims to investigate whether the comparative benefits of mask-wearing are still retained when other disease-management policies are put in place. In order to do so, this research will delve into the following questions. Questions 1 and 2 deal with individual handling of masks, while questions 3 to 5 examine the compound effect of other policies on masks.
1. If masks are less effective, how much safer are mask-wearers than non mask-wearers?
Before contextualising mask-wearing in different disease-management landscapes, we should examine the relative effectiveness of masks. Different masks have different effectiveness, either due to the filtration efficacy of the material used or the fit of the mask; a randomized trial comparing the effect of medical and cloth masks on healthcare worker illness found that those wearing cloth masks were 13 times more likely to experience influenza-like illness than those wearing medical masks (MacIntyre).
2. If masks are improperly handled, how much safer/dangerous are mask-wearers?
The effectiveness of masks also depends on their correct use. Experts claim that most people are unaware of how to wear or use masks properly, for example by touching their faces after touching their masks, or by not properly covering their faces, which could result in higher rates of infections (Howard) Also, by improperly reusing masks, they may breathe in contaminated respiratory droplets from the masks (Lopez). Using our model, we seek to investigate if masks are improperly handled, how much safer or dangerous are mask wearers compared to non-mask wearers.
3. How does improved hygiene policy affect the effectiveness of mask wearing?
Beyond mask-wearing, other strategies such as hand hygiene, lockdowns and aggressive testing are measures put in place to help slow transmission rates. The World Health Organization (WHO) advocates for better hand hygiene to reduce the transmission of COVID-19 via respiratory droplets or contact (WHO). Countries all over the world have put in place hand hygiene campaigns (WHO).
4. How does the effectiveness of lockdown policies affect the effectiveness of mask-wearing?
Many countries around the world have implemented some form of movement control or lockdown to curb daily social interactions and thus the spread of COVID-19. Modelling studies have shown that these social distancing interventions are highly effective in flattening the epidemic curve (Milne). We may even quantify the extent to which these have reduced daily travel in cities — for example, movement in Singapore has fallen to 10% of the normal (Citymapper).
5. How does the effectiveness of aggressive testing policies affect the effectiveness of mask-wearing?
Aggressive testing methods ensure targeted isolation. However, its benefits may not be universally reaped; there has to be enough test kits available in a city and even then testing is usually conducted in areas that experienced a large spike in cases. For Singapore, a spike in COVID-19 cases occurred within foreign worker dormitories. This led to aggressive testing, even for healthy individuals (Cher), to detect asymptomatic cases. Other cities such as New York (Romo) and Mumbai (Basu) have also started aggressive testing.

Research Methodology
======
To investigate the aforementioned research questions, we created COVID Mask Policy Simulator, an agent-based model (ABM) simulation designed to inform policies about mask-wearing. We have made the code and data open-source for replication of experimental results and further work at https://github.com/jetnew/COVID-Mask-Policy-Simulator.
Image for post

<img src="https://miro.medium.com/max/290/1*6yz6Ma7rnv4H06qOfwFsnw.png">
Figure 1: Simulation Animation Representing Infection Spread

In Figure 1, the simulation represents the spread of the infectious disease. Each agent has 4 different states: masked-uninfected (white), unmasked-uninfected (yellow), masked-infected (orange) and unmasked-infected (red).

Simulation Design
------
The COVID Mask Policy Simulator used in our experiments is a 100x100, 2D grid world. Agent movement is modelled identically by a random walk in 8 directions (north, south, east, west, northeast, northwest, southeast, southwest) at every discrete time step. Transmission of infection occurs during a collision when an infected agent moves to the same coordinate as an uninfected agent at a transmission rate of 0.20 between non-mask wearers. All simulations are initialised with a total of 200 agents, 10 of which are infected mask wearers and 10 are infected non-mask wearers. The simulations run until the entire population has been infected with the diseases. The inverse of the time taken for 25%, 50%, 75% and 100% of the population to be infected represents the rate of transmission given a set of experimental parameters.

Case Study Experiment Design
------
We investigate 6 case study experiments, consisting of 1 control, 2 conditions and 3 policies. This section describes each case study and its corresponding objective. All simulations are run until all agents in the environment become infected. All case study experiments are performed over 30 runs to reduce noise in the results.

<img src="https://miro.medium.com/max/875/1*CHIwD7YUFFM3WgqmbIzaag.png">
Table 1: Table of Notations
<img src="https://miro.medium.com/max/711/1*cqH1TE46AzthsIw63WZA7Q.png">
Table 2: Rates of Infection

In Table 2, we formulate the rate of infection (ri) of different types of transmissions. The effectiveness of mask (em) is inversely proportional to the rate of infection (ri). We represent this relationship in the form ri = ri / em. As ri of the transmission between mask wearer to non-mask wearer should be higher than that between mask wearer to mask wearer, we represent a further decrease in ri by ri = ri / em2. The severity of improper handling of masks (su) should be directly proportional to ri, thus ri of the transmission between non-mask wearer to mask-wearer is represented by ri = ri / em * (1 + su), a percentage increase in ri. For all experiments, ri=0.20, em=5, su=0, except for the Mask Effectiveness Condition, Improper Mask Handling Condition and Improved Hygiene Policy.

Experiment 1: Experimental Control
------
The objectives of the experimental control are: 1. to compare the transmission rate between mask wearers and non-mask wearers; and 2. to analyse how the percentage of mask wearers (tM/tT) in the population affects transmission rate.

Experiment 2: Mask Effectiveness Condition
------
The objective of the mask effectiveness condition is to analyse how the effectiveness of mask (em) affects the difference in transmission rate between mask wearers and non-mask wearers (dmu). The effectiveness of masks is inversely proportional to the rate of infection.

Experiment 3: Improper Mask Handling Condition
------
The objective of the improper mask handling condition is to analyse how the severity of improper mask handling (su) affects the difference in transmission rate between mask wearers and non-mask wearers (dmu). An assumption made to reduce simulation complexity is that 50% of mask wearers handle masks improperly. As the study investigates relative values, the severity of improper mask handling sufficiently captures the consequence of the improper mask handling situation.

Experiment 4: Improved Hygiene Policy
------
The objective of the improved hygiene policy is to analyse how reduced transmission rate from improved hygiene (ri) affects the difference in transmission rate between mask wearers and non-mask wearers (dmu). The effectiveness of the hygiene policy is inversely proportional to the infection rate (ri). The rate of infection (ri) between non-mask wearers is thus modified to indicate the effectiveness of the hygiene policy.

Experiment 5: Lockdown Policy
------
The objective of the lockdown policy is to analyse how percentage of people abiding by a lockdown (ps) affects the difference in transmission rate between mask wearers and non-mask wearers (dmu). Agents that abide by the lockdown policy will remain stationary, while agents that do not will continue movement.

Experiment 6: Aggressive Testing Policy
------
The objective of the aggressive testing policy is to analyse how aggressiveness of testing (and quarantine) (pq) affects the difference in transmission rate between mask wearers and non-mask wearers (dmu). At every timestamp, every infected agent has a probability of being tested and quarantined at pq, and removed from the environment, preventing further spread from the diagnosed agent. An assumption made to enable inter-experimental comparison is that the probability pq is restricted to a low value to enable at least 50% of the population to be infected at the end of the simulation.

Data Analysis
======
Experiment 1: Experimental control
------
Objective 1: To compare the transmission rate between mask wearers and non-mask wearers.
<img src="https://miro.medium.com/max/374/1*_36QtN4M2w1yIdv6dwCM3A.png">
Figure 2: No. of transmissions over time. Left: Sample 1, Right: Averaged (n=30)

According to Figure 2, the rate of transmission for non-mask wearers (red) is higher than that of mask wearers (green). The no. of transmissions over time for mask wearers and non-mask wearers can be modelled by performing logistic regression. The sigmoid function is defined as:
<img src="https://miro.medium.com/max/495/1*cYqzmIz-QxzG1F6ZhBlASA.png">
Using least squares optimisation of residuals, the no. of transmissions of mask wearers can be defined as:
<img src="https://miro.medium.com/max/794/1*zWfQz0D3_iRFTuSdbl-E9A.png">
while the no. of transmissions of non-mask wearers can be defined as:
<img src="https://miro.medium.com/max/738/1*MHBKgVd9j7Ndc0KvbSM0Dg.png">
By visually observing the gradients of the graphs, one can observe that the transmission rate of non-mask wearers is greater than that of mask wearers. Therefore, mask wearers are much safer than non-mask wearers, ceteris paribus.

Objective 2: To analyse how the percentage of mask wearers, nM/nT affect transmission rate.
<img src="https://miro.medium.com/max/380/1*DwuDXk_gjStutVBEb4ri1g.png">
Figure 3: Time taken to infect T% of the population against nM)/nT, Left: Sample 1, Right: Averaged (n=30)

As inferred from Figure 3, the time taken to infect 25%, 50%, 75% and 100% of the population with respect to the percentage of mask wearers can be modelled by performing exponential regression. The exponential function is defined as:
Using least squares optimisation of residuals, the time taken to infected T% of the population as a function of the percentage of mask wearers can be defined as:
<img src="https://miro.medium.com/max/459/1*d8WxpL2_zWJ3BqJLuy2phw.png">
Figure 4: Exponential regression of time taken against nM)/nT

Therefore, according to Figure 4, the higher the percentage of mask wearers, the longer the time required to infect 25%, 50%, 75% and 100% of the population, and the lower the rate of transmission in the population.
<img src="https://miro.medium.com/max/471/1*4EmNi2gwO25JUmO10i3fbg.png">
Figure 5: Standard deviation of time taken to infect 50% of the population (t_50) against nM)/nT

According to Figure 5, the standard deviation of the time for 50% of the population to be infected is also positively correlated to the proportion of masked. This shows that the outcomes will also vary more widely when there is a larger proportion of masked and the mean of t50 is larger. This can be explained by the fact that the simulation would exhibit more variance between experimental runs the longer the time it is run for. Thus, this effect will apply to all of the following experiments and would also be exacerbated by the policies that increase the time taken for the disease to spread.

Experiment 2: Mask Effectiveness Condition
Objective: To analyse how effectiveness of mask (em) affects the difference in transmission rate between mask wearers and non-mask wearers (dmu).

TBC
