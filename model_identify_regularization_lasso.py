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
    def __init__(self, P, n):
        super().__init__()

        # Trainable coefficient matrices
        self.Xi = self.add_weight(
            shape=(P,n),
            name="Xi_params",
            dtype=tf.float32,
            initializer="RandomNormal",
            trainable=True,
        )

    # Model call
    def call(self, Theta):
        # Add constraint to alpha, which has to be strictly positive

        res = tf.matmul(Theta, self.Xi)
        return res

# Monte-Carlo error for finding expected "error loss" L_c 
def loss_original(model, X, Y, L=2):
    """
    Compute the LOSS of the given "model" (MSE).
    """
    Y_pred = model(X)
    Loss_mse = tf.reduce_mean((Y_pred - Y)**2)
    return Loss_mse

def print_net_params(model):
    """
    Print parameters of the network, the estimated parameters "Xi"
    """
    Xi = model.Xi

    print(Xi.numpy())