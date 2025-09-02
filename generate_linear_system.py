# %%
import numpy as np
import scipy as sp

# Linear-Nonlinear System to simulate
def system(Y, t):
    mu = 2.5
    x = Y[0]
    y = Y[1]
    z = Y[2]
    dydt = [
        (- x - y),
        (- 5*y - z),
        (- z),
    ]
    return dydt
t = np.linspace(0, 2, 501)
dT = t[1] - t[0]

from scipy.integrate import odeint
y_0 = [1., -1., -3]

# Matrices where to save the data
# Estimate of differentiation has to be carried out BEFORE building matrices to reduce its error
n_IC=10         # Number of ICs
Theta=[]
X_dot=[]
for i in range(n_IC):
    y_0 = np.random.uniform(size=(3))*10 - 5
    sol=odeint(system, y_0, t)
    
    X_dot.append(np.gradient(sol, dT, axis=0))
    Theta.append(odeint(system, y_0, t))
# Matrices of the results
X_dot = np.concatenate(X_dot, axis=0)
Theta = np.concatenate(Theta, axis=0)

# Plot time evolution
import matplotlib.pyplot as plt
plt.plot(X_dot)

# %% ADD CODE TO test, then separate in different cells
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

# LOSS computation L_0
def complexity_loss(log_alpha, beta=2/3, gamma=-0.1, zeta=1.1):
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
                    log_alpha - beta * tf.math.log(-gamma / zeta)
                )
            )

    return L_c

def test_time_z_estimator(log_alpha, gamma=1.1, zeta=-0.1):
    """
    For the Hard_CONCRETE distribution return the value of the estimated z_hat given:
    - "log_alpha"
    - "gamma"
    - "zeta"
    """
    z_hat = tf.math.minimum(1, tf.math.maximum(0,
                tf.math.sigmoid(log_alpha)*(gamma - zeta) + zeta))
    return z_hat

# %%
# MODEL DEFINITION
import tensorflow as tf
import keras as keras

class log_alpha_limiter(keras.constraints.Constraint):
    # Class to limit log_alpha in a specified range
    def __init__(self, min_value, max_value):
        super().__init__()
        self.min_value=min_value
        self.max_value=max_value
    
    def __call__(self, w):
        return tf.clip_by_value(
            w,
            clip_value_min=self.min_value, clip_value_max=self.max_value
        )

class alpha_init(keras.Initializer):
    # Custom initializer class to choose the degree of opening of the logic gates
    # Uniform initializer
    def __init__(self, init_value):
        super().__init__()
        self.init_value = init_value

    def __call__(self, shape, dtype=tf.float32):
        return tf.ones(shape=shape, dtype=dtype)*self.init_value

class identification_class(keras.Model):
    def __init__(self, P, n, alpha_in, alpha_min, alpha_max, beta=2/3, gamma=1.1, zeta=0.1):
        super().__init__()
        # Set attributes
        self.beta=beta
        self.gamma=gamma
        self.zeta=zeta

        # Trainable coefficient matrices
        self.Xi = self.add_weight(
            shape=(P,n),
            name="Xi_params",
            dtype=tf.float32,
            initializer="RandomNormal",
            trainable=True,
        )

        initializer=alpha_init(alpha_in)
        log_limiter=log_alpha_limiter(min_value=alpha_min, max_value=alpha_max)
        self.log_alpha = self.add_weight(
            shape=(P,n),
            name="Alpha_params",
            dtype=tf.float32,
            initializer=initializer,
            trainable=True,
            constraint=log_limiter,
        )

    # Model call
    def call(self, Theta):
        # Add constraint to alpha, which has to be strictly positive
        z = hard_concrete_distr_sample(
            log_alpha=self.log_alpha,
            beta=self.beta,
            gamma=self.gamma,
            zeta=self.zeta,
            )
        Xi_eff = z * self.Xi

        res = tf.matmul(Theta, Xi_eff)
        return res

# %%
# Run the model
import keras as keras
opt = keras.optimizers.Adam(1e-2)
beta=2/3
zeta=1.1
gamma=-0.1
P=3
n=3
n_epoch = 5000
lam = 0.05
warm_up_epoch = 0       # lambda should start later in the model

model = identification_class(P=P, n=n, alpha_in=-2., alpha_min=-3, alpha_max=10, beta=beta, zeta=zeta, gamma=gamma)

X_dot_tf = tf.convert_to_tensor(X_dot, dtype=tf.float32)
Theta_tf = tf.convert_to_tensor(Theta, dtype=tf.float32)

# Monte-Carlo error for finding expected "error loss" L_c 
def monte_carlo_error(model, X, Y, L=2):
    """
    Compute the Monte-Carlo LOSS of the system according tha depends on random quantities (inside the "model").
    The expected value is computed as average among with "L"
    """
    Loss = 0
    for _ in range (L):
        Y_pred = model(X)
        mse = tf.reduce_mean((Y_pred - Y)**2)

        Loss+= mse
    return Loss/L

def print_net_params():
    """
    Print parameters of the network, especially "log_alpha", the value of the estimated mask "z_hat" and the estimated parameters "Xi"
    """
    Xi = model.Xi
    log_alpha = model.log_alpha

    z_hat = test_time_z_estimator(log_alpha=log_alpha)

    print(log_alpha.numpy())
    print(z_hat)
    print((Xi*z_hat))

# Effective Learning of the parameters
for epoch in range(n_epoch):
    with tf.GradientTape() as tape:
        # Computing Error-Losses L:c
        mc_loss=monte_carlo_error(model, Theta_tf, X_dot_tf, L=5)
        
        # Computing Complexity-Losses L_c (regularization losses)
        if(epoch<warm_up_epoch):
            eff_lam=0
        else:
            eff_lam=lam
        L_c=complexity_loss(model.log_alpha) * lam

        loss = mc_loss + L_c
    
    # Compute gradient and learn
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    if ((epoch + 1) % 50 == 0):
        print(f"Epoch: {epoch+1}: loss = {loss.numpy():.6f}", end="\r")
    
    if ((epoch + 1) % 200 == 0):
        print()
        print_net_params()
        print()


# %%
log_alpha = model.log_alpha
L=2
for L_i in range(1, L):
    with tf.GradientTape(persistent=True) as tape:
        mc_loss=monte_carlo_error(model, Theta_tf, X_dot_tf, L=L_i)
        L_c=complexity_loss(model.log_alpha) * lam

        loss=L_c + mc_loss
        # print(f"The loss is: {mc_loss:.4f}")
    grad=tape.gradient(mc_loss, model.log_alpha)
    grad2=tape.gradient(L_c, model.log_alpha)
    grad3=tape.gradient(loss, model.log_alpha)
    del tape
    print(f"L_e Loss_gradient for L={L_i}\n {grad}")
    print(f"L_c Loss_gradient for L={L_i}\n {grad2}")
    print(f"L Loss_gradient for L={L_i}\n {grad3}")