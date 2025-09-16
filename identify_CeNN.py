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
import matplotlib.pyplot as plt
plt.plot(Xdot)

# %%
# Instantiate and Run the model
import tensorflow as tf
import keras as keras
from model_identify_regularization import identification_class, complexity_loss
from tensorFlow_HardCONCRETE_lib import test_time_z_estimator
opt = keras.optimizers.Adam(1e-2)
beta=2/3
zeta=1.1
gamma=-0.1
P = Theta.shape[-1]
n = Xdot.shape[-1]

model = identification_class(
        P=P, n=n,
        alpha_in=0., alpha_min=-3, alpha_max=10,
        beta=beta, zeta=zeta, gamma=gamma
    )

X_dot_tf = tf.convert_to_tensor(Xdot, dtype=tf.float32)
Theta_tf = tf.convert_to_tensor(Theta, dtype=tf.float32)

# Monte-Carlo error for finding expected "error loss" L_c 
def monte_carlo_error(model, X, Y, L=2):
    """
    Compute the Monte-Carlo LOSS of the system according tha depends on random quantities (inside the "model").
    The expected value is computed as average among with "L"
    """
    Loss = 0
    for _ in range (L):
        Y_pred = model(X)
        mse = tf.reduce_mean((Y_pred - Y)**2)

        Loss+= mse
    return Loss/L

def print_net_params():
    """
    Print parameters of the network, especially "log_alpha", the value of the estimated mask "z_hat" and the estimated parameters "Xi"
    """
    Xi = model.Xi
    log_alpha = model.log_alpha

    z_hat = test_time_z_estimator(log_alpha=log_alpha)

    print(log_alpha.numpy())
    print(z_hat)
    print((Xi*z_hat))
# %%
# Effective Learning of the parameters
n_epoch = 10000
lam = 0.0001
warm_up_epoch = 5000       # lambda should start later in the model

for epoch in range(n_epoch):
    with tf.GradientTape() as tape:
        # Computing Error-Losses L:c
        mc_loss=monte_carlo_error(model, Theta_tf, X_dot_tf, L=5)
        
        # Computing Complexity-Losses L_c (regularization losses)
        if(epoch<warm_up_epoch):
            eff_lam=0
        else:
            eff_lam=lam
        L_c=complexity_loss(model.log_alpha) * lam

        loss = mc_loss + L_c
    
    # Compute gradient and learn
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    if ((epoch + 1) % 50 == 0):
        print(f"Epoch: {epoch+1}: loss = {loss.numpy():.6f}", end="\r")
    
    if ((epoch + 1) % 200 == 0):
        print()
        print_net_params()
        print()


# %%
print()
print_net_params()
print()