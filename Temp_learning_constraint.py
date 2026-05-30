# %%
import numpy as np
import matplotlib.pyplot as plt
import keras as keras
import tensorflow as tf

# generate random vectors
np.random.seed(100)
u0 = np.random.normal(size=(100,1),
                      loc=5,
                      scale=1)
u1 = np.random.normal(size=(100,1),
                      loc=5,
                      scale=1)
u2 = np.random.normal(size=(100,1),
                      loc=5,
                      scale=1)
u3 = np.random.normal(size=(100,1),
                      loc=5,
                      scale=1)
u4 = np.random.normal(size=(100,1),
                      loc=5,
                      scale=1)
u5 = np.random.normal(size=(100,1),
                      loc=5,
                      scale=1)
u6 = np.random.normal(size=(100,1),
                      loc=5,
                      scale=1)
u7 = np.random.normal(size=(100,1),
                      loc=5,
                      scale=1)
u8 = 10* np.random.normal(size=(100,1),
                      loc=5,
                      scale=1)

Theta = np.concatenate( (u0, u1, u2, u3, u4, u5, u6, u7, u8), axis=1)
Xi = np.zeros(shape=(9,1))
Xi[0] = 1
Xi[1] = 2
Xi[2] = +3
Xi[3] = -1
Xi[4] = 5
Xi[5] = 1
Xi[6] = -3
Xi[7] = -2
Xi[8] = -1.1
X_dot = Theta @ Xi

plt.plot(X_dot)
Theta_tf = tf.convert_to_tensor(Theta,
                                dtype=tf.float32,
                                )
X_dot_tf = tf.convert_to_tensor(X_dot,
                                dtype=tf.float32,
                                )

# Simple keras model class
class identification_class(keras.Model):
    def __init__(self, P, n):
        super().__init__()
        # Set attributes
        # Trainable coefficient matrices
        self.magnitudes = self.add_weight(
            shape=(P-4,n),
            name="Xi_mags",
            dtype=tf.float32,
            initializer="RandomNormal",
            trainable=True,
        )
        self.signs = self.add_weight(
            shape=(P,n),
            name="Xi_mags",
            dtype=tf.float32,
            initializer="zeros",
            trainable=True,
        )

    # Model call
    def call(self, Theta):
        # Add constraint to alpha, which has to be strictly positive
        mags = tf.gather(self.magnitudes,
                            indices=[0,1,2,3,4,3,2,1,0])
        mags_pos = tf.nn.softplus(mags)
        signs = tf.tanh(self.signs)
        Xi_real = signs * mags_pos
        res = tf.matmul(Theta, Xi_real)
        return res

P=Theta_tf.shape[-1]
n=X_dot_tf.shape[-1]

opt = keras.optimizers.Adam(1e-2)
model=identification_class(P=P, n=n)

# %%
n_epoch = 2000

def mse_loss(model, X, Y):
    Y_pred = model(X)
    loss = tf.math.reduce_mean((Y_pred - Y)**2)
    return loss

def asymmetry_loss(X, index):
    """
    given a matrix X and some indices create a penalty loss for not enforcing equality
    """

    first_slice = tf.gather(X,
                               indices=[0,1,2,3])
    second_slice = tf.gather(X,
                               indices=[8,7,6,5])
    error = tf.reduce_sum(tf.abs(
        (tf.abs(first_slice) - tf.abs(second_slice))
        ))
    return error

for epoch in range(n_epoch):
    with tf.GradientTape() as tape:
        # Computing Error-Losses L:c
        loss=mse_loss(model, Theta_tf, X_dot_tf) + 100*asymmetry_loss(model.signs, 0)
    
    # Compute gradient and learn
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

mags = tf.gather(model.magnitudes,
                            indices=[0,1,2,3,4,3,2,1,0])
mags_pos = tf.nn.softplus(mags)
signs = tf.tanh(model.signs)
Xi_real = signs * mags_pos

print(Xi_real)
# %%
