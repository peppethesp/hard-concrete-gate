# %%
# FINAL AIM: Having the U, V, P matrices as the catrices containing the neighboring cells respect to the central, build Theta as the nonlinear combination
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

ind_ind = 00
end_ind = -1
Theta_sub = Theta[ind_ind:end_ind,:]
U_dot_sub = U_dot[ind_ind:end_ind,:]
Xi, err, _, _ = np.linalg.lstsq(Theta_sub, U_dot_sub)

ind_ind = 1000
end_ind = 3000
error=(Theta[ind_ind:end_ind,:] @ Xi) - U_dot[ind_ind:end_ind,:]
plt.plot(error)
plt.figure()
plt.plot((Theta[ind_ind:end_ind,:] @ Xi), label="predicted")
plt.plot(U_dot[ind_ind:end_ind,:], label="ground truth")
plt.legend()

# %%

plt.figure()
plt.title("Derivative of state variable")
plt.plot(U_dot[0:100,:], '.')
plt.figure()
plt.title("State variable")
plt.plot(U[1,1, 0:100])

plt.figure()
plt.title("Thetas")
plt.plot(Theta[0:,:])