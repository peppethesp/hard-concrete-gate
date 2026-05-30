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

# input has to be between [0,1]
def cnn2d(X1_0, X2_0, A11, A22, I1, I2, mu, eps, s, dt, step):
    (N_1, M_1) = X1_0.shape

    N = N_1 + 2 
    M = M_1 + 2 

    # Preallocating vectors
    X1 = np.zeros(shape=(N, M, step), dtype=np.float64)
    X2 = np.zeros(shape=(N, M, step), dtype=np.float64)
    Y1 = np.zeros(shape=(N, M, step), dtype=np.float64)
    Y2 = np.zeros(shape=(N, M, step), dtype=np.float64)

    X1[1:-1, 1:-1, 0] = X1_0
    X2[1:-1, 1:-1, 0] = X2_0
    Y1[:,:,0] = np.clip(X1[:,:,0], a_min=-1, a_max=1)
    Y2[:,:,0] = np.clip(X2[:,:,0], a_min=-1, a_max=1)

    for t in range(1, step):
        for i in range(1, N-1):
            for j in range(1, M-1):
                coup1A = 0
                coup2A = 0

                for k in [-1, 0, 1]:
                    for l in [-1, 0, 1]:
                        coup1A += A11[k+1, l+1]*Y1[i+k, j+l, t-1]
                        coup2A += A22[k+1, l+1]*Y2[i+k, j+l, t-1]
                X1[i,j,t] = X1[i,j,t-1] + dt * (-X1[i,j,t-1] + (1+mu+eps)*Y1[i,j,t-1] - s*Y2[i,j,t-1] + coup1A + I1)
                X2[i,j,t] = X2[i,j,t-1] + dt * (-X2[i,j,t-1] + s * Y1[i,j,t-1] + (1+mu-eps)*Y2[i,j,t-1] + coup2A + I2)

        Y1[:,:,t] = np.clip(X1[:,:,t], a_min=-1, a_max=1)
        Y2[:,:,t] = np.clip(X2[:,:,t], a_min=-1, a_max=1)
        (X1_t, Y1_t) = Boundary(X1[:,:,t], Y1[:,:,t], 2)
        (X2_t, Y2_t) = Boundary(X2[:,:,t], Y2[:,:,t], 2)

        X1[:,:,t] = X1_t
        X2[:,:,t] = X2_t
        Y1[:,:,t] = Y1_t
        Y2[:,:,t] = Y2_t
        print(("Time: " + str(t)), end="\r")
    return (X1,X2,Y1,Y2)

# %%
(N_i, M_i) = (50, 50)
X1_i = np.ones(shape=(N_i, M_i), dtype=np.float64)
X2_i = np.zeros_like(X1_i)

X1_i[0:N_i//2, M_i//2-5] = 0
X1_i[0:N_i//2, M_i//2-5+1] = 1 * 0.25
X1_i[0:N_i//2, M_i//2-5+2] = 1 * 0.75

X2_i[0:N_i//2, M_i//2-5] = 0
X2_i[0:N_i//2, M_i//2-5+1] = 0.25
X2_i[0:N_i//2, M_i//2-5+2] = 0.75
X2_i[0:N_i//2, M_i//2-5+3] = 1
X2_i[0:N_i//2, M_i//2-5+4] = 1

A11 = np.array( [[0, 0.1, 0],
                 [0.1, -4*0.1, 0.1],
                 [0, 0.1, 0]] )
A22 = np.array( [[0, 0.1, 0],
                 [0.1, -4*0.1, 0.1],
                 [0, 0.1, 0]] )

X1_i = -2*(X1_i - 0.5)
X2_i = -2*(X2_i - 0.5)

(X1,X2,Y1,Y2) = cnn2d(X1_i, X2_i, A11=A11, A22=A22, I1=-0.3, I2=0.3, mu=0.7, eps=0., s=1, dt=0.1, step=2000)

# %%
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(X1_i, cmap='grey')
plt.figure()
plt.imshow(X2_i, cmap='grey')
# %%
plt.imshow(Y2[:,:,500], cmap="grey")
plt.colorbar()

plt.figure()
plt.plot(Y1[10,25,:])

# %%
# Save dataset to .pz file for later system identification
import os
folder= "2_RD_CeNN"
path = os.path.join(os.getcwd(), folder)
if (os.path.isdir(path) == False):
    try:
        os.mkdir(path)
    except:
        print("Error creating folder")

np.savez(os.path.join(path, "data"))