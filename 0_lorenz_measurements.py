# %% [markdown]
# # Save data

# %%
import os
import numpy as np
def save_data(x, y, z, t, folder_name, filename):
    """
    Function for saving the dataset inside the specified folder with a filename
    """
    path = os.path.join(os.getcwd(), folder_name)

    if(os.path.isdir(path) == False):
        try:
            os.mkdir(path)
        except Exception as e:
            print("Could not create folder.")
    print(path + folder_name)
    np.savez(os.path.join(path, filename),
            x = x,
            y = y,
            z = z,
            time = t)

# %% [markdown]
# # Let us begin by integrating A lorentz system

# %%
import numpy as np
import matplotlib.pyplot as plt
def lorenz(X, t, sigma, rho, beta):
    x, y, z = X

    dydt = 1 * np.array([sigma*(y - x),
            x*(rho - z) - y,
            x*y - beta*z])
    
    return dydt

sigma = 10
rho = 28
beta = 8/3

dT = 0.001
t = np.arange(0, 10, dT)

from scipy.integrate import odeint
in_cond = 15

for i in range(in_cond):
    y0 = np.random.uniform(-5, 5, (3))
    sol = odeint(lorenz, y0, t, args=(sigma, rho, beta))
    x = sol[:,0]
    y = sol[:,1]
    z = sol[:,2]
    filename = "dataset_"+str(i)
    save_data(x,y,z,t,"DATA",filename)
# Extract solutions

print(sol.shape)
x = sol[:,0]
y = sol[:,1]
z = sol[:,2]
t_new = np.arange(x.size)*dT 
plt.plot(t_new, x, 'b', label='x(t)')
plt.plot(t_new, y, 'g', label='y(t)')
plt.plot(t_new, z, 'r', label='z(t)')
plt.legend(loc='best')
plt.xlabel('time')
plt.grid()

plt.figure()
plt.plot(x,y)