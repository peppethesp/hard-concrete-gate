# %%
# This is an old script; first steps in the learning of mask learning for network parameters. 
# Script to generate, through numpy, the hard-CONCRETE sampling from a distribution
import numpy as np

def sigmoid(u):
    """
    Function that computes the elementwise sigmoid function for the input vector u
    """
    s = 1 / (1 + np.exp(-u))

    return s

def concrete_distr(alpha, beta, size):
    """
    Reparametrization:
    Special case for "binary" CONCRETE distribution

    alpha learnable parameter
    beta: temperature; as it approaches zero the CONCRETE resembles Bernoulli's distribution
    """
    u = np.random.uniform(size=(size,1))
    s = sigmoid((np.log(alpha) + np.log(u) - np.log(1 - u)) / beta)

    return s

def hard_concrete_distr(alpha, beta, gamma, zeta, size):
    """
    generator Hard-CONCRETE distribution from parameter-less noise
    with:
    - alpha-beta parameters of the CONCRETE
    - gamma-zeta typical of Hard-CONCRETE
    """
    size = int(size)
    s = concrete_distr(alpha, beta, size)

    s_bar = s * (zeta - gamma) + gamma
    z = np.minimum( 1, np.maximum( 0, s_bar ) )
    return z

# %%
import matplotlib.pyplot as plt

alpha = 1
gamma = -0.05
zeta = 1.05
beta = 1
size = 1e6
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
s = hard_concrete_distr(alpha, beta, gamma, zeta, size)
plt.hist(s, 100, density=True)
plt.title(f"Temperature: {beta:.2f}")

beta = 2/3
plt.subplot(1, 3, 2)
# plt.figure()
s = hard_concrete_distr(alpha, beta, gamma, zeta, size)
plt.hist(s, 100, density=True)
plt.title(f"Temperature: {beta:.2f}")

beta = 0.1
plt.subplot(1, 3, 3)
# plt.figure()
s = hard_concrete_distr(alpha, beta, gamma, zeta, size)
plt.hist(s, 100, density=True)
plt.title(f"Temperature: {beta:.2f}")
