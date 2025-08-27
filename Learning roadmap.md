# Variational Inference Basics (VI)

- what it is: (approximation for intractable posteriors)
- Key idea of computing instead of the intractable $p(z |x)$ $q(z | \theta)$
	- ELBO
	- the role of $q(\cdot)$ as variational distribution
	- why we want distributions in Bayesian deep learning
Not the whole theory, but the one required to understand $ q(z_j | \pi_j) $ and why optimizing over $\pi_j$.

# Reparametrization trick
Given that you cant backpropagate if sample from Bernoulli's distribution ($Bernoulli (\pi)$), use a trick:
- parameter-free noise (???) $\varepsilon$
- distribution's parameters $\theta$

$$ s = \sigma \left( \left( 
	log \alpha + log u - log (1 - u)\right) / \beta
	\right)$$
$$ u \backsim \mathrm{ Uniform(0, 1) } $$

# Continuous relaxations of discrete distributions
Bernoulli's not differentiable so we replace it with a smooth approximation (Concrete distribution).

- Hard-concrete = Concrete distribution stratched + clipped to hit exactly 0 and 1 with a non zero probability.

This becomes almost discrete and, moreover, fully differentiable.