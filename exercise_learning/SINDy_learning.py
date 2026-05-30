# %%

import numpy as np
import tensorflow as tf

# Numeric integration function
def nl_system(T, dT, x_0):
    """
    Numerically integrate singla state NonLinear system;

    Returns state evolution
    """
    x = np.zeros( shape=(T,1), dtype = np.float32 )
    x[0] = x_0
    for t in range(1, T):
        dx = -1*x[t-1] + 0.5 * (x[t-1]**3)
        # dx = 0.2797*x[t-1] +0 * (x[t-1]**2) +0.4862 * (x[t-1]**3)
        x[t] = x[t-1] + dT * dx
    
    return x
# Generate the synthetic data
T = 500
dT = 0.01

# Generate Dataset in tensorflow dataset friendly format for different ICs
# so that .from_tensor_slices() can be used
ICs = np.array([1, -1, 0.5, -0.4, 0.3, 0.8, -0.7], dtype=np.float32)

n_IC = ICs.shape[0]
x = np.empty(shape=(n_IC, T, 1), dtype=np.float32)
xdot = np.empty(shape=(n_IC, T, 1), dtype=np.float32)
Theta = np.empty(shape=(n_IC, T, 4), dtype=np.float32)
for i, X0 in enumerate(ICs):
    x[i,:,:] = nl_system(T, dT, x_0=X0)
    xdot[i,:,:] = np.gradient(x[i], dT, axis = 0)
    Theta[i,:,:] = np.concatenate((np.ones_like(x[i]),
                            x[i],
                            x[i]**2,
                            x[i]**3,
                            ), axis=1)
    

import matplotlib.pyplot as plt
plt.plot(Theta[0,:,1], label="x")
plt.plot(xdot[0], label="xdot")
plt.legend()

print("Theta shape: ", Theta.shape)
print("Xdot shape: ", xdot.shape)

# %% ============================================================

P = Theta.shape[-1]
n = xdot.shape[-1]

# Convert data to suitable dataset (data, label)
Theta_tf = tf.data.Dataset.from_tensor_slices((Theta[:,:,:], xdot[:,:,:]))

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

model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
              loss='mse')

model.fit(x = Theta_tf,
          epochs = 500)


mask_values = tf.sigmoid(model.mask_logits).numpy()
Xi_values = model.Xi.numpy()

Xi_eff = Xi_values * mask_values

print("Learned effective coefficients: ")
for term, coef in zip(["1", "x", "x^2", "x^3"], Xi_eff[:,0]):
    print(f"{term:>4s}: {coef:+.4f}")
