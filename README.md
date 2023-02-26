# Transformers Can Do Bayesian Inference
Paper Presentation by Austin Coursey
## Overview
### Motivation
Let's consider a situation.
- A casino publishes data on winnings
- They send out a survey to the recent visitors who they have contact information for
- "8 out of the 10 people who responded actually made money!"
- Likelihood of making money given data follows (Binomial likelihood):

$$p(D | \theta)={10 \choose 8}\theta^8(1-\theta)^{2}$$

<p align="center">
  <img src="https://github.com/acoursey3/transformer-bayesian/blob/main/figures/likelihood.png?raw=true">
</p>

- Does this seem right? 
- What **prior** information might we want to consider when modeling casino win probability?

#### Being Bayesian

- Model **prior** knowledge as a Beta distribution

$$
Beta(2,18) = p(x|2,18)=\frac{x^{1}(1-x)^{17}}{B(2,18)}
$$

<p align="center">
  <img src="https://github.com/acoursey3/transformer-bayesian/blob/main/figures/prior.png?raw=true">
</p>

- We want to update the misleading likelihood given our prior information
- Bayes theorem:

$$
p(\theta | D) = \frac{p(D | \theta) p(\theta)}{p(D)}
$$

- Our beta-binomial posterior:

<p align="center">
  <img src="https://github.com/acoursey3/transformer-bayesian/blob/main/figures/form_posterior.png?raw=true">
</p>

<p align="center">
  <img src="https://github.com/acoursey3/transformer-bayesian/blob/main/figures/posterior.png?raw=true">
</p>

- Continually adjust beliefs under new data
- Incorporate knowledge
- Useful with little data
- Distribution over parameter

#### Issues with Bayesian Inference

- $p(D)$ is often intractable
- Turn towards methods like Markov Chain Monte Carlo
- Slow, need to sample
- Variational Inference
- Approximate, less accurate

### Transformers for Bayesian Inference
