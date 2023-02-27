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

- Often we want to Posterior Predictive Distribution rather than posterior (Gaussian Processes and Bayesian Neural Networks)
- To solve this, use Transformers!

#### Approach Overview

<p align="center">
  <img src="https://github.com/acoursey3/transformer-bayesian/blob/main/figures/pfn_overview.png?raw=true">
</p>

- Create a prior over a supervised learning task (functions of datasets)
- Sample task/function from prior
- Draw X/y
- Mask a $y_i$ and learn to make predictions (Transformer model)
- At test time, feed in your dataset and $X$
- Outputs PPD for test point

This will approximate the exact posterior predictive distribution at better speeds than MCMC and VI!

## Paper Details

### Modifications to Transformer Architecture

- Feed (x,y) pairs as sum of their encodings (linear projections)
- Encoder Transformer - head depends on task
  - Softmax/sigmoid for classification
  - Riemann distribution for regression (contribution)
- Attention:

<p align="center">
  <img src="https://github.com/acoursey3/transformer-bayesian/blob/main/figures/attention.png?raw=true">
</p>

- Why do they need a transformer?
  - Connected set neural network (graph nn)
  - Permutation equivariant (invariant after head)
  
### Other Important Details

- Custom loss function
  - Proved to approximate PPD
  
### Main Results

#### GP Approximation
- GP is a type of Bayesian ML model
- Distribution over functions
- For fixed hyperparameters, it is tractable
- [GP Demo](https://huggingface.co/spaces/samuelinferences/transformers-can-do-bayesian-inference)

- For hyperparameter priors
  - 200x faster than MLE-II and 1000-8000x faster than MCMC
  


## Code Demonstration
