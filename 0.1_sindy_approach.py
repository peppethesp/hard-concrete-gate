# %%
# Import the data from a dataset folder
import numpy as np
import os

foldername = "DATA"

path = os.path.join(os.getcwd(), foldername)
files = os.listdir(path)

# Function to add Noise to an array with zero mean and a specified standard deviation
def add_noise(X : np.ndarray, std : float):
    mean = np.mean(X)

    return (X + np.random.normal(0, std, size=X.shape ))

# Variable to store the imported dataset
X = np.empty(shape=(0,3))
X_dot = np.empty(shape=(0,3))

# Loading of the time data
for i, filename in enumerate(files):
    file_path = os.path.join(path, filename)
    try:
        with np.load(file_path) as data:
            x = data['x']
            y = data['y']
            z = data['z']
            time = data['time']
            dT = time[1]-time[0]
    except Exception as e:
        print("FIle was not found")
    # Get state
    X_temp = np.stack([x,
                       y,
                       z]).transpose()
    
    X_temp = add_noise(X_temp, std=0.2)
    
    # Let us start to compute X_dot
    x_dot = np.gradient(X_temp[:,0], time)
    y_dot = np.gradient(X_temp[:,1], time)
    z_dot = np.gradient(X_temp[:,2], time)

    X_dot_temp = np.stack([x_dot,
                           y_dot,
                           z_dot]).transpose()
    
    X = np.append(X, X_temp, axis=0)
    X_dot = np.concatenate((X_dot, X_dot_temp), axis=0)

# %% [markdown]
# # Reconstruct the matrices needed by SINDy algorithm
# These are:
# - $\dot{X}$: derivatives of the state vector
# - $\Theta$: libraries
# - $\Xi$: index of parameters

# %%
# Let us compute the value of the matrix \Theta with polynomial nonlinearities up to order 2

Theta = X
for i in range(3):
    for j in range(i, 3):
        nl_term = np.expand_dims(X[:,i]*X[:,j], 1)
        print(i, j)
        Theta = np.concatenate([Theta, nl_term], axis = 1)
print(Theta.shape)

# %% [markdown]
# # Implementation of Sequential Thresholded Least Squares

# %%
# Loop through the derivatives of state variables
import numpy as np
from numpy.linalg import lstsq
lib_size = Theta.shape[1]
num_state = 3

lam = 0.05
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

Csi_true = np.array([[-10, 28, 0],
            [10, -1, 0],
            [0, 0, 2.66],
            [0, 0, 0],
            [0, 0, 1],
            [0, -1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            ])

# print(Csi)
import matplotlib.pyplot as plt
bound = np.ceil(np.max(np.abs(Csi)))
plt.pcolormesh(Csi, cmap="seismic",
               vmin=-bound,
               vmax=bound,
               )
plt.colorbar()

plt.figure()
plt.pcolormesh(Csi_true, cmap="seismic",
               vmin=-bound,
               vmax=bound,
               )
plt.colorbar()


