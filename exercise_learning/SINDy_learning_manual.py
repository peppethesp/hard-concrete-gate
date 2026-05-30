# %%

import numpy as np
import tensorflow as tf

# Generate the synthetic data
T = 500
dT = 0.01
x = np.zeros( shape=(T,1), dtype = np.float32 )
x[0] = -1
# Numeric integration
for t in range(1, T):
    dx = -x[t-1] + 0.5 * (x[t-1]**3)
    # dx = -0.7226*x[t-1] - 0.2091 * (x[t-1]**3) 
    x[t] = x[t-1] + dT * dx

xdot = np.gradient(x, dT, axis = 0)
import matplotlib.pyplot as plt
plt.plot(x)
# plt.plot(xdot)

Theta = np.concatenate([np.ones_like(x), x, x**2, x**3], axis=1)

print("Theta shape: ", Theta.shape)
print("Xdot shape: ", xdot.shape)

# %% ============================================================

P = Theta.shape[1]
n = xdot.shape[1]

# Convert data to tensors
Theta_tf = tf.convert_to_tensor(Theta, dtype=tf.float32)
xdot_tf = tf.convert_to_tensor(xdot, dtype=tf.float32)

class SYNDy_learnable_mask(tf.keras.Model):
    def __init__(self, P, n):
        super().__init__()
        # Trainable coefficient matrix
        self.Xi = self.add_weight( shape=(P, n),
                                  initializer="random_normal",
                                  trainable=True )
        
        self.mask_logits = self.add_weight(shape=(P, n),
                                           initializer="random_normal",
                                           trainable=True,
        )

    def call(self, Theta):
        mask = tf.sigmoid(self.mask_logits)
        Xi_eff = self.Xi * mask
        return tf.matmul(Theta, Xi_eff)

model = SYNDy_learnable_mask(P, n)
# %%
# Optimizer
import keras as keras
opt = keras.optimizers.Adam(1e-2)
n_epoch = 500
for epoch in range(n_epoch):
    with tf.GradientTape() as tape:
        y_pred = model(Theta)
        loss = tf.reduce_mean( (y_pred - xdot_tf)**2 )

    # Compute gradient
    grads = tape.gradient(loss, model.trainable_variables)

    opt.apply_gradients(zip(grads, model.trainable_variables))

    if ((epoch + 1) % 100 == 0):
        print(f"Epoch: {epoch+1}: loss = {loss.numpy():.6f}")


mask_values = tf.sigmoid(model.mask_logits).numpy()
Xi_values = model.Xi.numpy()

Xi_eff = Xi_values * mask_values

print("Learned effective coefficients: ")
for term, coef in zip(["1", "x", "x^2", "x^3"], Xi_eff[:,0]):
    print(f"{term:>4s}: {coef:+.4f}")