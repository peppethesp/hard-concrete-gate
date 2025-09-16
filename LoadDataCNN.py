# Function to import the data for CeNN for the REACTION-DIFFUSION CeNN
# %%
import os
import numpy as np

def read_data(t_start, t_end):
    """
    Function that reads the data saved as .npz file containing time evolution X and Y variables for a 2 dimensional Reaction Diffusion CeNN.

    - t_start: sample from which to read data
    - t_end: sample at which to stop reading data

    The data returned consists of a Tensor with dimensions (W, H, T) with 'W' and 'H' spatial dimension, while 'T' accounts for the time dimension
    """
    foldername = "DATA_2_RD_CeNN"

    path = os.path.join(os.getcwd(), foldername)
    files = os.listdir(path)

    # Variable to store the imported dataset
    # STATE VARIABLES

    # Loading of the time data

    for i, filename in enumerate(files):
        file_path = os.path.join(path, filename)
        try:
            with np.load(file_path) as data:
                X_data = data['X'][:,:,t_start:t_end]
                Y_data = data['Y'][:,:,t_start:t_end]
                
        except Exception as e:
            print("FIle was not found")
            
    return X_data, Y_data

def sample_data(X_data, Y_data, time):
    """
    Function to sample 'X_data' and 'Y_data' spatially in points

    Returns:
    - X_data: the state vector matrix sampled at certain grid points stacked vertically
    - X_dot: the corresponding time derivative matrix
    - U: the neighboring state now considered as external input for the SINDy algorithm
    """
    U=[]
    X=[]
    X_dot = []
    # Get state: sample data ad different spatial points with space of 'L' from each other
    N_x, N_y = X_data[:,:,0].shape
    L = 4
    Number_x_max = (N_x - 1)/L - 1
    Number_y_max = (N_y - 1)/L - 1
    i_x = np.arange(1, Number_x_max, dtype=np.int16) * L + 1
    i_y = np.arange(1, Number_y_max, dtype=np.int16) * L + 1

    # Loop through that space to get data
    for i in (i_x):
        for j in (i_y):
            # Add state variables
            time_size = X_data.shape[2]
            X_temp = np.stack([X_data[i,j,:],
                            Y_data[i,j,:]]).transpose()
            
            x_dot = np.gradient(X_temp[:,0], time)
            y_dot = np.gradient(X_temp[:,1], time)
            
            X_dot_temp = np.stack([x_dot,
                                y_dot]).transpose()

            # Add neighborhood which are considered as if they were external inputs
            U_temp = np.empty(shape=(time_size, 0))
            for k in range(3):
                for l in range(3):
                    # Do not consider the cell itself in the neighborhood
                    if (not(k == 1 and l == 1)):
                        U_temp = np.append(U_temp, X_data[i+k-1, j+l-1, :].reshape((time_size, 1)), axis=1)
                        U_temp = np.append(U_temp, Y_data[i+k-1, j+l-1, :].reshape((time_size, 1)), axis=1)

            if (len(U) == 0):
                U = U_temp
                X = X_temp
                X_dot = X_dot_temp
            else:
                X = np.append(X, X_temp, axis=0)
                X_dot = np.append(X_dot, X_dot_temp, axis=0)
                U = np.append(U, U_temp, axis=0)
    return X_dot, X, U