# %%
import numpy as np
import matplotlib.pyplot as plt
T = 100
size=(3,3,T)

filename="DATA_CFD_CeNN/dataset_ni0.5_ro2small_eddys.npz"
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
    L_x = 15
    L_y = 18
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
            t_end_extract = 400
            U_extracted = (U_final[i-r:i+r+1,j-r:j+r+1,t_in_extract:t_end_extract])
            V_extracted = (V_final[i-r:i+r+1,j-r:j+r+1,t_in_extract:t_end_extract])
            P_extracted = (P_final[i-r:i+r+1,j-r:j+r+1,t_in_extract:t_end_extract])

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
    #k

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

X_dot = U_dot

print(Theta.shape)

from NormalizeMatrix import normalize_matrices

Theta, X_dot, Theta_norms, X_dot_std = normalize_matrices(Theta, X_dot)
# compute spectral norm (cheap for p=36)
U, S, Vt = np.linalg.svd(Theta, full_matrices=False)
spec = S[0]
L = (spec**2) / Theta.shape[0]
0.9/L

# %%
# Instantiate the model
import tensorflow as tf
import keras as keras
from model_identify_regularization_lasso import identification_class, loss_original, print_net_params

warm_up_epoch = 0       # lambda should start later in the model

X_dot_tf = tf.convert_to_tensor(X_dot, dtype=tf.float32)
Theta_tf = tf.convert_to_tensor(Theta, dtype=tf.float32)

P=Theta.shape[-1]
n=X_dot.shape[-1]
model = identification_class(P=P, n=n)

# %%
# Run the model
lr=965
n_epoch = 60000
lam = .000005*0
# Normalization of Lambda
lam_norm = Theta_norms.transpose() * lam / X_dot_std
lam_norm_tf = tf.convert_to_tensor(lam_norm, dtype=tf.float32)
# Effective Learning of the parameters
for epoch in range(n_epoch):
    with tf.GradientTape() as tape:
        if(epoch<warm_up_epoch):
            eff_lam=0
        else:
            eff_lam=lam_norm
        
        # Computing Error-Losses L_c
        loss=loss_original(model, Theta_tf, X_dot_tf)

    # Compute gradient and learn
    dL_dXi, = tape.gradient(loss, [model.Xi])
    model.Xi.assign_sub(lr*dL_dXi)

    # Apply shrinkage
    # Proximal Gradient Descent
    Xi=model.Xi
    Xi_new = tf.sign(Xi) * tf.maximum(tf.abs(Xi)-lam*lr, 0.0)
    model.Xi.assign(Xi_new)
    
    if ((epoch + 1) % 400 == 0):
        print()
        print_net_params(model)
        print()

    if ((epoch + 1) % 50 == 0):
        print(f"Epoch: {epoch+1}: loss = {loss.numpy():.6f}", end="\r")
# %%
# Compute Gradient
Xi = model.Xi
L=2
for L_i in range(1, L):
    with tf.GradientTape(persistent=True) as tape:
        loss=loss_original(model, Theta_tf, X_dot_tf)

        # print(f"The loss is: {mc_loss:.4f}")

    grad=tape.gradient(loss, Xi)
    del tape
    print(f"L_e Loss_gradient for L={L_i}\n {grad}")

# %%
Xi = model.Xi


plt.plot( X_dot)
plt.plot( Theta@(Xi))

