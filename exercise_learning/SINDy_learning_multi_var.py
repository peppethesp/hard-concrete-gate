# %%

import numpy as np
import tensorflow as tf

# Numeric integration function
def nl_system(T, dT, X_0):
    """
    Numerically integrate Lorenz nonlinear system;
    - T: number of steps
    - dT: integration time step
    - X_0: Initial conditions

    Returns numpy ndarray state vector evolution with shape (TIME, STATES)
    """
    X = np.zeros( shape=(T,3), dtype = np.float32 )
    sigma = 10
    rho = 28
    beta = 8/3

    X[0] = X_0
    for t in range(1, T):
        x = X[t-1, 0]
        y = X[t-1, 1]
        z = X[t-1, 2]
        dx = np.array(([sigma*(y - x),
                        x*(rho - z) - y,
                        x*y - beta*z]))
        X[t, :] = X[t-1, :] + dT * dx
    return X
# Generate the synthetic data
T = 2000
dT = 0.001
X = nl_system(T, dT, [0, 1, 2])
Xdot = np.gradient(X, dT, axis=0)

import matplotlib.pyplot as plt

plt.plot(X[:,0], label="x")
plt.plot(X[:,1], label="xdot")
plt.plot(X[:,2], label="xdot")
plt.grid()

# Building library of nonlinear functions
Theta = X
for i in range(3):
    for j in range(i, 3):
        nl_term = np.expand_dims(X[:,i] * X[:,j], 1)
        Theta = np.concatenate((Theta, nl_term), axis=1)
print(Theta.shape)


# %%
# Generate Dataset in tensorflow dataset friendly format for different ICs
# so that .from_tensor_slices() can be used
n_IC = 40
X = np.empty(shape=(n_IC, T, 3), dtype=np.float32)
Xdot = np.empty(shape=(n_IC, T, 3), dtype=np.float32)
Theta = np.empty(shape=(n_IC, T, 9), dtype=np.float32)
for i in range(n_IC):
    ICs = np.random.random_sample(size=(1,3))*20 - 10
    X[i,:,:] = nl_system(T, dT, X_0=ICs)

    # Add noise
    X[i,:,:] += np.random.random(size=X[i,:,:].shape)*0

    Xdot[i,:,:] = np.gradient(X[i], dT, axis = 0)

    Theta_tmp = X[i,:,:]
    for j in range(3):
        for k in range(j, 3):
            nl_term = np.expand_dims(X[i,:,j] * X[i,:,k], 1)
            Theta_tmp = np.concatenate((Theta_tmp, nl_term), axis=1)

    Theta[i,:,:] = Theta_tmp

# PLOTTING
import matplotlib.pyplot as plt
ind = 15
plt.plot(Theta[ind,:,1], label="x")
plt.plot(Xdot[ind], label="xdot")
plt.legend()

print("Theta shape: ", Theta.shape)
print("Xdot shape: ", Xdot.shape)


P = Theta.shape[-1]
n = Xdot.shape[-1]

# Convert data to suitable dataset (data, label)
Theta_tf = tf.data.Dataset.from_tensor_slices((Theta, Xdot))
# %% =========== DEFINE THE MODEL

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

# %% COMPILE AND FIT MODEL TO DATA ========================
model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
              loss='mse')
model.fit(x = Theta_tf,
          epochs = 500)

mask_values = tf.sigmoid(model.mask_logits).numpy()
Xi_values = model.Xi.numpy()

Xi_eff = Xi_values * mask_values

# print("Learned effective coefficients: ")
# for term, coef in zip(["1", "x", "x^2", "x^3"], Xi_eff[:,0]):
#     print(f"{term:>4s}: {coef:+.4f}")