# %%
import numpy as np
import scipy as sp

def system(Y, t):
    mu = 2.5
    x = Y[0]
    y = Y[1]
    dydt = [
        (-x - y),
        -5* y,
    ]
    return dydt

t = np.linspace(0, 5, 201)
dT = t[1] - t[0]

from scipy.integrate import odeint
y_0 = [1., -1.]

sol = odeint(system, y_0, t)


# %%
import matplotlib.pyplot as plt
plt.plot(sol)

x = sol[:, 0]
y = sol[:, 1]

# Compute derivatives approximation
x_dot = np.gradient(x, dT)
y_dot = np.gradient(y, dT)

X_dot = np.stack((x_dot, y_dot), axis=1)
# Matrix theta
Theta = np.stack( (x, y), axis=1 )

# %% ADD CODE TO test, then separate in different cells
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

def hard_concrete_distr_sample(alpha, beta=2/3, gamma=-0.1, zeta=1.1):
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
def complexity_loss(alpha, beta=2/3, gamma=-0.1, zeta=1.1):
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

# %%
import tensorflow as tf
import keras as keras

X_dot_tf = tf.convert_to_tensor(X_dot, dtype=tf.float32)
Theta_tf = tf.convert_to_tensor(Theta, dtype=tf.float32)

class identification_class(keras.Model):
    def __init__(self, P, n):
        super().__init__()
        # Trainable coefficient matrix
        self.Xi = self.add_weight(
            shape=(P,n),
            name="Xi_params",
            dtype=tf.float32,
            initializer="RandomNormal",
            trainable=True,
        )
        self.alpha = self.add_weight(
            shape=(P,n),
            name="Alpha_params",
            dtype=tf.float32,
            initializer="ones",
            trainable=True,
        )

    def call(self, Theta):
        # Add constraint to alpha, which has to be strictly positive

        safe_alpha = tf.clip_by_value(self.alpha, 1e-6, 1e2)
        z = hard_concrete_distr_sample(alpha=safe_alpha)
        Xi_eff = z * self.Xi

        res = (tf.matmul(Theta, Xi_eff))
        self.add_loss(
            0.001 * complexity_loss(alpha=safe_alpha)
        )
        return res

model = identification_class(2, 2)

# %%
import keras as keras
opt = keras.optimizers.Adam(1e-2)
n_epoch = 500
for epoch in range(n_epoch):
    with tf.GradientTape() as tape:
        y_pred = model(Theta_tf)
        loss = tf.reduce_mean((y_pred - X_dot_tf)**2)
        loss += sum(model.losses)
    
    # Compute gradient
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

Xi = model.Xi
alpha = model.alpha
# %%
safe_alpha = tf.clip_by_value(alpha, 1e-6, 1e2)
z_hat = tf.math.minimum(1, tf.math.maximum(0,
            tf.math.sigmoid(tf.math.log(safe_alpha))*(1.1 + 0.1) -0.1))
z_hat_2 = hard_concrete_distr_sample(alpha=safe_alpha)

print(z_hat)
print(z_hat_2)
print((Xi*safe_alpha))

# %%
