# %%
# Extension of SINDy algorithm for fluido-dynamics
# %%
# Import data section
import numpy as np
import matplotlib.pyplot as plt
T = 100
size=(3,3,T)

filename="DATA_CFD_CeNN/dataset_ni0.5_ro1.npz"
npzfile=np.load(filename)
U_final = npzfile['U_final']
V_final = npzfile['V_final']
P_final = npzfile['P_final']
dT = npzfile['dT']

def sample_CeNN(U_in, V_in, P_in):
    """
    Samples the layers of a Cellular Neural Network (CeNN) and its external inputs.

    This function extracts samples from the input tensors at different positions,
    stacking them along the third dimension. Derivatives or gradients should be
    computed at this stage if needed.

    Parameters:
        u_in (ndarray): Input tensor U with shape (W, H, T)
        v_in (ndarray): Input tensor V with shape (W, H, T)
        p_in (ndarray): Input tensor P with shape (W, H, T)

    Returns:
        tuple:
            u (ndarray): Sampled tensor U with shape (2*r, 2*r, n*T)
            v (ndarray): Sampled tensor V with shape (2*r, 2*r, n*T)
            p (ndarray): Sampled tensor P with shape (2*r, 2*r, n*T)

    Notes:
        - 'r' is the influence radius of the CeNN.
        - 'n' is the number of samples taken.
        - 'n' is the number of samples taken.
        - 'n' is the number of samples taken.
    """
    r = 1
    W, H, T = U_in.shape

    #
    # Choose sampling points
    L_x = 10
    L_y = 10
    N_x = W//(L_x+1)
    N_y = H//(L_y+1)

    U, V, P = [], [], []
    U_dot, V_dot, P_dot = [], [], []
    for n_x in range(1, N_x):
        for n_y in range(1, N_y):
            i = n_x*(L_x+1)
            j = n_y*(L_y+1)

            # For now we discard first samples (simulation artifacts)
            t_in_extract = 50
            U_extracted = (U_final[i-r:i+r+1,j-r:j+r+1,t_in_extract:])
            V_extracted = (V_final[i-r:i+r+1,j-r:j+r+1,t_in_extract:])
            P_extracted = (P_final[i-r:i+r+1,j-r:j+r+1,t_in_extract:])

            # Compute the derivative
            U_dot.append ( np.gradient(U_extracted[r,r,:, None], dT, axis=0) )
            V_dot.append ( np.gradient(V_extracted[r,r,:, None], dT, axis=0) )
            P_dot.append ( np.gradient(P_extracted[r,r,:, None], dT, axis=0) )
            
            U.append(U_extracted)
            V.append(V_extracted)
            P.append(P_extracted)

    U_dot = np.concatenate(U_dot, axis=0)
    V_dot = np.concatenate(V_dot, axis=0)
    P_dot = np.concatenate(P_dot, axis=0)
    
    U = np.concatenate(U, axis=2)
    V = np.concatenate(V, axis=2)
    P = np.concatenate(P, axis=2)
    return U_dot, V_dot, P_dot, U, V, P
    #


# reshape the matrix into Theta structure
def unwrap_matrix(m):
    """
    Function that unwraps a tensor of nonlinear terms in a Theta-like shape.

    Supposing the matrix has the shape (W, H, T), with each nonlinear term arranged as the slice along the last dimension, the final output will be Theta with shape (T, W*H)
    """
    W, H, T = m.shape
    Theta = np.empty(
        shape=(T, W*H),
        dtype=m.dtype,
    )

    for i in range(W):
        for j in range(H):
            Theta[:, i*H+j] = m[i,j,:]
    return Theta

# Building data
U_dot, V_dot, P_dot, U, V, P = sample_CeNN(U_final, V_final, P_final)

u = U[1,1]
v = V[1,1]

# Computing nonlinearities for Theta
t1 = U
t2 = u * U
t3 = v * U
t4 = P

Theta1 = unwrap_matrix(t1)
Theta2 = unwrap_matrix(t2)
Theta3 = unwrap_matrix(t3)
Theta4 = unwrap_matrix(t4)

# Building final Theta matrix
Theta = np.concatenate((Theta1, Theta2, Theta3, Theta4), axis=1)

print(f"The shape of Theta is: \n\t{Theta.shape}\n")
# %%
# Loop through the derivatives of state variables
import numpy as np
from numpy.linalg import lstsq

X_dot = U_dot

lib_size = Theta.shape[1]
num_state = X_dot.shape[-1]

# %% ==========================================================
import tensorflow as tf
P = Theta.shape[-1]
n = X_dot.shape[-1]

# Convert data to suitable dataset (data, label)
# Theta_tf = tf.data.Dataset.from_tensor_slices((Theta[:,:,:], Xdot[:,:,:]))
# Theta_tf = tf.data.Dataset.from_tensor_slices((Theta, Xdot))
# For future implementation: separate data in more (data, label) pairs
Theta_tf = tf.convert_to_tensor(Theta, dtype=tf.float32)
Xdot_tf = tf.convert_to_tensor(X_dot, dtype=tf.float32)

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
epochs = 10000
# model.compile(optimizer=tf.keras.optimizers.Adam(1e-2),
#               loss='mse')
# model.fit(x = Theta_tf,
#           y = Xdot_tf,
#           epochs = epochs)

import keras as keras
opt = keras.optimizers.Adam(1e-2)
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = model(Theta)
        loss = tf.reduce_mean((y_pred - X_dot)**2)
    # compute gradient
    grads = tape.gradient(loss, model.trainable_variables)

    opt.apply_gradients(zip(grads, model.trainable_variables))
    if ((epoch + 1) % 10 == 0):
        print(f"Epoch: {epoch+1}: loss = {loss.numpy():.6f}")

mask_values = tf.sigmoid(model.mask_logits).numpy()
Xi_values = model.Xi.numpy()

Xi_eff = Xi_values * mask_values

import matplotlib.pyplot as plt
bound = np.ceil(np.max(np.abs(Xi_eff)))
plt.pcolormesh(Xi_eff, cmap="seismic",
               vmin=-bound,
               vmax=bound)
plt.colorbar()

# print("Learned effective coefficients: ")
# for term, coef in zip(["1", "x", "x^2", "x^3"], Xi_eff[:,0]):
#     print(f"{term:>4s}: {coef:+.4f}")


# %%
plt.plot(Theta@Xi_eff,
         label="Predicted")
plt.plot(X_dot,
         label="Ground truth")

# NOTE: IT just rediscovered other differentiation SCHEMES! No sparcity is ensured even though 
# predicted value and ground truth are really close