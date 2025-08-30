# %% [markdown]
# Sample code to start with the Hard-CONCRETE distribution

# %%
import tensorflow as tf

def concrete_distr_sample(alpha, beta):
    """
    Function that samples from concrete distribution with parameter:
    - alpha (probability-related) for the logic gates.
    - beta (temperature)
    """

    # Create u for random gates to be sampled
    u = tf.random.uniform(shape=alpha.shape,
                          minval=1e-15,
                          maxval=1)

    s = tf.math.sigmoid((tf.math.log(alpha) + tf.math.log(u) - tf.math.log(1 - u)) / beta)

    return s

def hard_concrete_distr_sample(alpha, beta=1.0, gamma=0.0, zeta=1.0):
    """
    Function that samples, provided alpha parameter of the gates, from the hard concrete distribution
    """

    # Error handling
    if (beta <= 0 or beta > 1):
        raise("Value \"beta\" can only be in (0;1] ")

    if (gamma > 0):
        raise("Value \"gamma\" can only be less or equal zero ")
    
    if (zeta < 1):
        raise("Value \"zeta\" can only be greater or equal one ")
    # END Error handling

    s = concrete_distr_sample(alpha, beta)
    s_bar = s * (zeta - gamma) + gamma

    z = tf.math.minimum( 1, tf.math.maximum( 0, s_bar ) )
    return z

# LOSS computation L_0
def complexity_loss(alpha, beta, gamma, zeta):
    """
    Compute the complexity loss as in (Louizos et al., 2018).
        Sigmoid(log(alpha_j) - beta*log(-gamma/zeta))
    Is essentially the probability that each mask element is different than zero.
    This means, by summing for all j, an approximation quite close to the original L_0 regularization

    - alpha: tensor with the mask
    - beta: temperature of the CONCRETE distribution
    - gamma: Parameters related to the Hard-CONCRETE distribution (stretching)
    - zeta: Parameters related to the Hard-CONCRETE distribution (stretching)
    """
    L_c = tf.reduce_sum(
                tf.math.sigmoid( 
                    tf.math.log(alpha) - beta * tf.math.log(-gamma / zeta)
                )
            )

    return L_c

# %% - Plot results
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    alpha = tf.Variable(tf.maximum( tf.random.normal(shape=(10, 10),mean=3), 0 ))
    alpha = .50 * tf.ones(shape=(500,500))
    beta = 0.5
    gamma=-0.1
    zeta=1.1

    z = hard_concrete_distr_sample(alpha=alpha,
                                beta=beta,
                                gamma=gamma,
                                zeta=zeta)

    plt.hist(z.numpy().flatten(),
            bins=100,
            density=True)

    plt.figure()
    plt.imshow(z)
    plt.colorbar()

    print(complexity_loss(alpha, beta, gamma, zeta))