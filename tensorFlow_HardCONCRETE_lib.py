# %% ADD CODE TO test, then separate in different cells
import tensorflow as tf

def concrete_distr_sample(log_alpha, beta):
    """
    Function that samples from concrete distribution with parameter:
    - alpha (probability-related) for the logic gates.
    - beta (temperature)
    """

    # Create u matrix for random gates to be sampled
    u = tf.random.uniform(shape=log_alpha.shape,
                          minval=1e-15,
                          maxval=1)

    s = tf.math.sigmoid((log_alpha + tf.math.log(u) - tf.math.log(1 - u)) / beta)

    return s

def hard_concrete_distr_sample(log_alpha, beta=2/3, gamma=-0.1, zeta=1.1):
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

    s = concrete_distr_sample(log_alpha, beta)
    s_bar = s * (zeta - gamma) + gamma
    z_hard = tf.math.minimum( 1., tf.math.maximum( 0., s_bar ) )
    
    # Gradient flowing through s_bar to avoid it stopping learning because of zero gradient of the hard_sigmoid
    z = s_bar+tf.stop_gradient(z_hard - s_bar)
    return z

def test_time_z_estimator(log_alpha, gamma=-0.1, zeta=1.1):
    """
    For the Hard_CONCRETE distribution return the value of the estimated z_hat given:
    - "log_alpha"
    - "gamma"
    - "zeta"
    """
    z_hat = tf.math.minimum(1, tf.math.maximum(0,
                tf.math.sigmoid(log_alpha)*(zeta - gamma) + gamma))
    return z_hat

if __name__=="__main__":
    import matplotlib.pyplot as plt

    log_alpha=tf.ones(
        shape=(50, 50),
        dtype=tf.float32,
    )
    beta=0.5
    gamma=-0.1
    zeta=1.1

    z=hard_concrete_distr_sample(
        log_alpha,
        beta=beta,
        gamma=gamma,
        zeta=zeta)
    
    plt.hist(
        z.numpy().flatten(),
        bins=100,
        density=True,
        )
    
    plt.figure()
    plt.imshow(z)
    plt.colorbar()
    # # Show changes of beta ======
    log_alpha=tf.zeros(shape=(100000,1))
    plt.figure()
    beta=2/3
    z = hard_concrete_distr_sample(log_alpha=log_alpha, beta=beta, gamma=gamma, zeta=zeta)
    plt.subplot(1, 2, 1)
    plt.title(f"Temperature {beta:.2f}")
    plt.hist(z.numpy(), 50, density=True)

    beta=0.2
    z = hard_concrete_distr_sample(log_alpha=log_alpha, beta=beta, gamma=gamma, zeta=zeta)
    plt.subplot(1, 2, 2)
    plt.title(f"Temperature {beta:.2f}")
    plt.hist(z.numpy(), 50, density=True)