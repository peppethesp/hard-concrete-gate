# %%
import numpy as np
import scipy as sp

# Linear-Nonlinear System to simulate
def system(Y, t):
    x = Y[0]
    y = Y[1]
    z = Y[2]
    dydt = [
        (- x - y),
        (- 5*y - z),
        (- z),
    ]
    return dydt
t = np.linspace(0, 2, 501)
dT = t[1] - t[0]

from scipy.integrate import odeint
y_0 = [1., -1., -3]

# Matrices where to save the data
# Estimate of differentiation has to be carried out BEFORE building matrices to reduce its error
n_IC=10         # Number of ICs
Theta=[]
X_dot=[]
for i in range(n_IC):
    y_0 = np.random.uniform(size=(3))*10 - 5
    sol=odeint(system, y_0, t)
    
    X_dot.append(np.gradient(sol, dT, axis=0))
    Theta.append(odeint(system, y_0, t))
# Matrices of the results
X_dot = np.concatenate(X_dot, axis=0)
Theta = np.concatenate(Theta, axis=0)

# Plot time evolution
import matplotlib.pyplot as plt
plt.plot(X_dot)

# %%
# Instantiatethe model
import tensorflow as tf
import keras as keras
from model_identify_regularization import identification_class, complexity_loss, print_net_params, loss_monte_carlo_error
from tensorFlow_HardCONCRETE_lib import test_time_z_estimator

opt = keras.optimizers.Adam(1e-2)
beta=2/3
zeta=1.1
gamma=-0.1
warm_up_epoch = 0       # lambda should start later in the model

X_dot_tf = tf.convert_to_tensor(X_dot, dtype=tf.float32)
Theta_tf = tf.convert_to_tensor(Theta, dtype=tf.float32)

P=Theta.shape[-1]
n=X_dot.shape[-1]
model = identification_class(P=P, n=n,
                             alpha_in=-2., alpha_min=-3, alpha_max=10,
                             beta=beta, zeta=zeta, gamma=gamma)
# %%
# Run the model
n_epoch = 5000
lam = 0.05
# Effective Learning of the parameters
for epoch in range(n_epoch):
    with tf.GradientTape() as tape:
        if(epoch<warm_up_epoch):
            eff_lam=0
        else:
            eff_lam=lam
        
        # Computing Error-Losses L:c
        mc_loss=loss_monte_carlo_error(model, Theta_tf, X_dot_tf, L=5)
        # Computing Complexity-Losses L_c (regularization losses)
        L_c=complexity_loss(model.log_alpha) * lam

        loss = mc_loss + L_c
    
    # Compute gradient and learn
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    if ((epoch + 1) % 50 == 0):
        print(f"Epoch: {epoch+1}: loss = {loss.numpy():.6f}", end="\r")
    
    if ((epoch + 1) % 200 == 0):
        print()
        print_net_params(model)
        print()


# %%
# Compute Gradient
log_alpha = model.log_alpha
L=2
for L_i in range(1, L):
    with tf.GradientTape(persistent=True) as tape:
        mc_loss=loss_monte_carlo_error(model, Theta_tf, X_dot_tf, L=L_i)
        L_c=complexity_loss(model.log_alpha) * lam

        loss=L_c + mc_loss
        # print(f"The loss is: {mc_loss:.4f}")
    grad=tape.gradient(mc_loss, model.log_alpha)
    grad2=tape.gradient(L_c, model.log_alpha)
    grad3=tape.gradient(loss, model.log_alpha)
    del tape
    print(f"L_e Loss_gradient for L={L_i}\n {grad}")
    print(f"L_c Loss_gradient for L={L_i}\n {grad2}")
    print(f"L Loss_gradient for L={L_i}\n {grad3}")