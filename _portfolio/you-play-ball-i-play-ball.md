---
title: "Reinforcement Learning: You Play Ball, I Play Ball"
excerpt: "Bayesian Multi-Agent Reinforcement Learning for Slime Volleyball, won 1st prize at 17th STePS 2020."
date: 2020-11-01
collection: portfolio
---

Multi-agent reinforcement learning has recently seen rising interest, popularised by projects like Multi-agent Hide-and-Seek and the AI Economist. Fascinating behaviour emerge from the interactions of multiple learning agents. Bayesian learning has also gotten popular from its ability to quantify and minimise uncertainty.

I led my project team to combine Bayesian methods with multi-agent reinforcement learning in the Slime Volleyball gym environment that was recently created by David Ha, a researcher whose work I look up to. This project is done as part of the CS3244 Machine Learning class, taught by Prof Min-Yen Kan at NUS. Our work achieved 1st place by popular vote at the biannual NUS School of Computing project showcase, 17th STePS 2020.

To go straight to the point, our result shows that applying Bayesian methods in multi-agent reinforcement learning improves 4 aspects of training: performance, training stability, uncertainty and generalisability.

We implemented a Bayesian version of Proximal Policy Optimization (PPO), a popular reinforcement learning baseline algorithm, using TensorFlow Probability and Stable Baselines.

We designed 3 types of experiments to evaluate the Bayesian version of PPO with its default counterpart: Single-agent training (training against an expert), Self-play (training against itself) and Multi-agent training (training against each other).

WIP

# Links
* [Project Report](https://docs.google.com/document/d/1HJ3IjbatOBlOJoJhyHPoM7hVIhLsD-Vcos-aLb9nVfY/edit?usp=sharing)
* [Project Website](https://slimerl.tech/)
* [Project Slides](https://docs.google.com/presentation/d/1lpYF99HBASFVS0ECSQm6nuvcRfUnbh9zDNzUaoKXSpo/edit?usp=sharing)
* [Project Poster](https://github.com/jetnew/SlimeRL/blob/master/Project%20Poster.pdf)
* [Project Video](https://www.youtube.com/watch?v=8qjV19gkZXc)
* [GitHub Repository](https://github.com/jetnew/SlimeRL)
* [STePS Website](https://isteps.comp.nus.edu.sg/event/17th-steps/module/CS3244/project/3)
* [1st Prize Award LinkedIn Post](https://www.linkedin.com/posts/jetnew_machinelearning-reinforcementlearning-datascience-activity-6732485574315401216-1W-t)
* [Project Proposal](https://drive.google.com/file/d/1xvRyS5ofoN8bw9RjzibqT9YAHRRE6CYF/view?usp=sharing)
* [Project Consultation 1](https://docs.google.com/presentation/d/1O6ExD_vdRdhKRxXiab7q1AjSzT7YCTlh76wlFFkfdyg/edit?usp=sharing)
* [Project Consultation 2](https://docs.google.com/presentation/d/114YImbSTDSkVc3V_F6YmyryZBxyx8mnf2T0Vw_w9glw/edit?usp=sharing)