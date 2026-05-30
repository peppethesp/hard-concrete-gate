# %%
import numpy as np

def Boundary(X, Y, type):
    Xt = np.copy(X)
    Yt = np.copy(Y)
    if (type == 2):
        # Zero-flux boundary conditions
        Xt[:,0] = X[:,1]
        Xt[:,-1] = X[:,-2]
        Xt[0,:] = X[1,:]
        Xt[-1,:] = X[-2,:]
        Yt[:,0] = Y[:,1]
        Yt[:,-1] = Y[:,-2]
        Yt[0,:] = Y[1,:]
        Yt[-1,:] = Y[-2,:]
    else:
        # Periodic boundary conditions
        Xt[:,0] = X[:,-2]
        Xt[:,-1] = X[:,1]
        Xt[0,:] = X[-2,:]
        Xt[-1,:] = X[1,:]
        Yt[:,0] = Y[:,-2]
        Yt[:,-1] = Y[:,1]
        Yt[0,:] = Y[-2,:]
        Yt[-1,:] = Y[1,:]
    return(Xt, Yt)

# input has to be between [-1,1]
# Simulation for FitzHugh-Nagumo
def cnn2d_FN(X_0, Y_0, A11, A22, I1, I2, eps, b, dt, step):
    (N_1, M_1) = X_0.shape

    N = N_1 + 2 
    M = M_1 + 2 

    # Preallocating vectors
    X = np.zeros(shape=(N, M, step), dtype=np.float64)
    Y = np.zeros(shape=(N, M, step), dtype=np.float64)

    X[1:-1, 1:-1, 0] = X_0
    Y[1:-1, 1:-1, 0] = Y_0

    for t in range(1, step):
        for i in range(1, N-1):
            for j in range(1, M-1):
                coup1A = 0
                coup2A = 0

                for k in [-1, 0, 1]:
                    for l in [-1, 0, 1]:
                        coup1A += A11[k+1, l+1]*X[i+k, j+l, t-1]
                        coup2A += A22[k+1, l+1]*Y[i+k, j+l, t-1]
                X[i,j,t] = X[i,j,t-1] + dt * ( - (X[i,j,t-1]**3/3 - X[i,j,t-1]) - Y[i,j,t-1] + coup1A + I1)
                Y[i,j,t] = Y[i,j,t-1] + dt * ( - eps * (X[i,j,t-1] - b * Y[i,j,t-1]) + coup2A + I2)

        (X_t, Y_t) = Boundary(X[:,:,t], Y[:,:,t], 2)

        X[:,:,t] = X_t
        Y[:,:,t] = Y_t
        print(("Time: " + str(t)), end="\r")
    return (X,Y)

# %%
(N_i, M_i) = (50, 50)
X_i = (np.random.uniform(low=0, high=2, size=(N_i, M_i)).astype(np.float64) - 1) * 1.5 
Y_i = (np.random.uniform(low=0, high=2, size=(N_i, M_i)).astype(np.float64) - 1) * 0.4

NabSq = np.array([[0, 1, 0],
                 [1, -4, 1],
                 [0, 1, 0]] )
D1 = 0.3
D2 = 1.4

# Normalization of the inputs
# X_i = -2*(X_i - 0.5)

# Y_i = -2*(Y_i - 0.5)

(X,Y) = cnn2d_FN(X_0=X_i, Y_0=Y_i, A11=D1*NabSq, A22=D2*NabSq, I1=0, I2=0, eps=-0.1, b=1.5, dt=0.005, step=2500)

# %%
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(X_i, cmap='grey')
plt.colorbar()
plt.figure()
plt.imshow(Y_i, cmap='grey')
plt.colorbar()

# %%
plt.figure(figsize=(7, 5))
plt.subplot(1,2,1)
plt.imshow(X[:,:,-1], cmap="grey")
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(Y[:,:,-1], cmap="grey")
plt.colorbar()

plt.figure()
plt.plot(Y[10,25,:])
plt.plot(Y[10,24,:])
plt.plot(Y[10,26,:])
plt.figure()
plt.plot(X[10,25,:])
plt.plot(X[10,24,:])
plt.plot(X[10,26,:])

# %% 
# Plot animated
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from IPython.display import HTML
fig, ax = plt.subplots()
im = ax.imshow(X[:,:,0])

def update_frame(frame):
    new_data = X[:,:,frame]
    im.set_array(new_data)
    return im


# %%
# Save dataset to .pz file for later system identification
import os
folder= "DATA_2_RD_CeNN"
path = os.path.join(os.getcwd(), folder)
if (os.path.isdir(path) == False):
    try:
        os.mkdir(path)
    except:
        print("Error creating folder")

np.savez(os.path.join(path, "data"), X=X, Y=Y)