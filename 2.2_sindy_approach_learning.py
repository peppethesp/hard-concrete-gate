# %%
# Extension of SINDy algorithm to work with RD_CNN FitzHugh-Nagumo CeNN to get to SINDy for fluido-dynamics
# %%
# Import data section
import numpy as np
from LoadDataCNN import read_data, sample_data

def add_noise(X : np.ndarray, std : float):
    """ Function that adds Noise to an array with zero mean and a specified standard deviation"""
    return (X + np.random.normal(0, std, size=X.shape ))



# time indices of the CeNN simulation to take; for now I chose these because they are empirically associated to the transient 
t_start = 0
t_end = 50

X_data, Y_data = read_data(t_start, t_end)

# Simulate noise on data
X_data = add_noise(X_data, std=0.00001)
Y_data = add_noise(Y_data, std=0.00001)

dT = 0.005
time = np.arange( start=0, stop=(t_end-t_start) ) * dT
Xdot, X, U = sample_data(X_data, Y_data, time)


# %% =================================================================
# Let us compute the value of the matrix \Theta with polynomial nonlinearities up to order 3

# BUILD LIBRARY 
Theta = X

num_state = X.shape[1]

# Create nonlinearities with external code function "Combination.py"
from Combination import CombinationRepetitionComplete

combination_total = CombinationRepetitionComplete(3, num_state)
for i in (range(1, len(combination_total))):
    # loop through the polynomial nonlinearity order; skip linear variables because already present
    for j in range(len(combination_total[i])):
        # loop through the various combinations
        print("term: ", end=" ")

        nl_term = np.ones_like(X[:,0])
        for k in range(len(combination_total[i][j])):
            # Loop through each term of the combination
            term_index = combination_total[i][j][k]
            print(term_index, end=" ")
            nl_term *= X[:,term_index]
        print()
        # Put the row as column then concatenate to Theta
        nl_term = np.expand_dims(nl_term, 1)
        Theta = np.concatenate([Theta, nl_term], axis = 1)


print("Theta with nonlinearities of second order: ", end="")
print(Theta.shape)
print("Theta with linear inputs of neighbors: ", end="")
Theta = np.concatenate([Theta, U], axis = 1)
print(Theta.shape)

# %% ==========================================================
import tensorflow as tf
P = Theta.shape[-1]
n = Xdot.shape[-1]

# Convert data to suitable dataset (data, label)
# Theta_tf = tf.data.Dataset.from_tensor_slices((Theta[:,:,:], Xdot[:,:,:]))
# Theta_tf = tf.data.Dataset.from_tensor_slices((Theta, Xdot))
# For future implementation: separate data in more (data, label) pairs
Theta_tf = tf.convert_to_tensor(Theta, dtype=tf.float32)
Xdot_tf = tf.convert_to_tensor(Xdot, dtype=tf.float32)

# %% ============================================================

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
          y = Xdot_tf,
          epochs = 100)

mask_values = tf.sigmoid(model.mask_logits).numpy()
Xi_values = model.Xi.numpy()

Xi_eff = Xi_values * mask_values

import matplotlib.pyplot as plt
plt.pcolormesh(Xi_eff, cmap="seismic", vmin=-6, vmax=6)
plt.colorbar()

# print("Learned effective coefficients: ")
# for term, coef in zip(["1", "x", "x^2", "x^3"], Xi_eff[:,0]):
#     print(f"{term:>4s}: {coef:+.4f}")