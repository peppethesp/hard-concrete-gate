# %%
import tensorflow as tf

def concrete_distr_sample(alpha, beta, u):
    """
    Function that samples from concrete distribution with parameter:
    - alpha (propability-related)
    - beta (temperature)
    """
    s = tf.math.sigmoid((tf.math.log(alpha) + tf.math.log(u) - tf.math.log(1 - u)) / beta)

    return s

def hard_concrete_distr_sample(alpha, beta, gamma, zeta, u):
    """
    Function that samples, provided noise 'u', from the hard concrete distribution
    """
    s = concrete_distr_sample(alpha, beta, u)
    s_bar = s * (zeta - gamma) + gamma

    z = tf.math.minimum( 1, tf.math.maximum( 0, s_bar ) )
    return z


alpha = 1.0
gamma = -.05
zeta = 1.05

u = tf.random.uniform(shape=(100000, 1))

import matplotlib.pyplot as plt
plt.figure(figsize=(7, 4))

beta = 2/3
z = hard_concrete_distr_sample(alpha, beta, gamma, zeta, u)
plt.subplot(1, 2, 1)
plt.title(f"Temperature {beta:.2f}")
plt.hist(z, 50, density=True)

beta = 0.2
z = hard_concrete_distr_sample(alpha, beta, gamma, zeta, u)
plt.subplot(1, 2, 2)
plt.title(f"Temperature {beta:.2f}")
plt.hist(z, 50, density=True)
