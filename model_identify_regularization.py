# %%
# MODEL DEFINITION
import tensorflow as tf
import keras as keras
from tensorFlow_HardCONCRETE_lib import hard_concrete_distr_sample, test_time_z_estimator

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

# Monte-Carlo error for finding expected "error loss" L_c 
def loss_monte_carlo_error(model, X, Y, L=2):
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
    
def print_net_params(model):
    """
    Print parameters of the network, especially "log_alpha", the value of the estimated mask "z_hat" and the estimated parameters "Xi"
    """
    Xi = model.Xi
    log_alpha = model.log_alpha

    z_hat = test_time_z_estimator(log_alpha=log_alpha)

    print(log_alpha.numpy())
    print(z_hat)
    print((Xi*z_hat))