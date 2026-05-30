# %%
# Extension of SINDy algorithm to work with RD_CNN FitzHugh-Nagumo CeNN to get to SINDy for fluido-dynamics
# %%
# Import data section
import numpy as np
from LoadDataCNN import read_data, sample_data

# Function to add Noise to an array with zero mean and a specified standard deviation
def add_noise(X : np.ndarray, std : float):
    return (X + np.random.normal(0, std, size=X.shape ))

# time indices of the CeNN simulation to take; for now I chose these because they are empirically associated to the transient 
t_start = 0
t_end = 50

X_data, Y_data = read_data(t_start, t_end)

dT = 0.005
time = np.arange( start=0, stop=(t_end-t_start) ) * dT
X_dot, X, U = sample_data(X_data, Y_data, time)


# =================================================================
# %%
# INTERPOLATIONS
def interp(X_data):
    """
    Function that upscales then downscales a 3D tensor along the time direction and then returns the distorted tensor. It effectively adds noise
    """
    from scipy.interpolate import RegularGridInterpolator
    x_dim, y_dim, t_dim = X_data.shape
    x = np.arange(x_dim)
    y = np.arange(y_dim)
    
    SCALE_FACTOR = 1.2
    x_new = np.linspace(0, int(x_dim)-1, int(x_dim * SCALE_FACTOR))
    y_new = np.linspace(0, int(y_dim)-1, int(y_dim * SCALE_FACTOR))
    xg_new, yg_new = np.meshgrid(x_new, y_new,
                                indexing='ij')
    X_data_ds = np.zeros_like(X_data, dtype=np.float64)
    # Resample-downsample for every time step in the dataset; this will add noise
    for k in range(t_dim):    
        # Scale-up
        upscaler = RegularGridInterpolator((x, y), X_data[:,:,k],
                                        method="linear",
                                        )

        pts = upscaler((xg_new, yg_new))
        # Scale-down
        downscaler = RegularGridInterpolator( (x_new, y_new),
                                                pts,
                                                method='linear',
                                                    )
        xg, yg = np.meshgrid(x, y,
                            indexing='ij')
        X_data_ds[:,:,k] = downscaler((xg,yg))
    
    return X_data_ds


X_data_ds = interp(X_data)
Y_data_ds = interp(Y_data)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.imshow(X_data_ds[:,:,-1])
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(X_data[:,:,-1])
plt.colorbar()

plt.figure()
plt.hist((X_data_ds[:,:,-1]- X_data[:,:,-1]).flatten(), 30)

# Sample distorted data for identification
# X_dot, X, U = sample_data(X_data_ds, Y_data_ds, time)


# %% [markdown]
# # Reconstruct the matrices needed by SINDy algorithm
# These are:
# - $\dot{X}$: derivatives of the state vector
# - $\Theta$: libraries
# - $\Xi$: index of parameters

# =================================================================

# %%
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

# =================================================================

# %% [markdown]
# # Implementation of Sequential Thresholded Least Squares

# =================================================================

# %%
# Loop through the derivatives of state variables
import numpy as np
from numpy.linalg import lstsq
lib_size = Theta.shape[1]


lam = 0.1
Csi = np.zeros(shape=(lib_size, num_state), dtype=np.float64)
iterations = 10

S = [np.arange(lib_size) for i in range(num_state)]
for i in range(iterations):
    # S_j is the set of indices, for each state variable x_j, of the functions in the libraries that contribute to its dynamics
    for j in range(num_state):
        csi_j_tmp = lstsq(Theta[:,S[j]], X_dot[:, j])[0]
        
        # filter out csi smaller than lambda
        csi_j_tmp[np.absolute(csi_j_tmp) < lam] = 0
        Csi[S[j],j] = csi_j_tmp

        S[j] = np.nonzero(Csi[:, j])[0]
        

print(Csi)
# xdot
Csi_real = np.zeros_like(Csi)
Csi_real[0, 0] = - 0.2
Csi_real[1, 0] = -1
Csi_real[5, 0] = -0.33
Csi_real[11, 0] = 0.3
Csi_real[15, 0] = 0.3
Csi_real[17, 0] = 0.3
Csi_real[21, 0] = 0.3
# ydot
Csi_real[0, 1] = - 0.1
Csi_real[1, 1] = -5.75
Csi_real[5, 1] = 0
Csi_real[12, 1] = 1.4
Csi_real[16, 1] = 1.4
Csi_real[18, 1] = 1.4
Csi_real[22, 1] = 1.4
print(Csi_real)

# print(Csi)
import matplotlib.pyplot as plt
# Make the plot symmetric, choose the maximum value
bound = np.ceil(np.max(np.abs(Csi)))
plt.pcolormesh(Csi, cmap="seismic",
               vmin=-bound,
               vmax=bound,
               )
plt.colorbar()
plt.figure()
plt.pcolormesh(Csi_real, cmap="seismic",
               vmin=-bound,
               vmax=bound,
               )
plt.colorbar()