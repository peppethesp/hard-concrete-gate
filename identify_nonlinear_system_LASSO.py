# %%
import numpy as np

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
# Instantiate the model
import tensorflow as tf
import keras as keras
from model_identify_regularization_lasso import identification_class, loss_original, print_net_params

warm_up_epoch = 0       # lambda should start later in the model

X_dot_tf = tf.convert_to_tensor(X_dot, dtype=tf.float32)
Theta_tf = tf.convert_to_tensor(Theta, dtype=tf.float32)

lr=1e-2

P=Theta.shape[-1]
n=X_dot.shape[-1]
model = identification_class(P=P, n=n)
opt = keras.optimizers.Adam(lr)
# %%
# Run the model
n_epoch = 5000
lam = 0.1
# Effective Learning of the parameters
for epoch in range(n_epoch):
    with tf.GradientTape() as tape:
        if(epoch<warm_up_epoch):
            eff_lam=0
        else:
            eff_lam=lam
        
        # Computing Error-Losses L_c
        loss=loss_original(model, Theta_tf, X_dot_tf)

    
    # Compute gradient and learn
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))

    # Apply shrinkage
    # Proximal Gradient Descent
    Xi=model.Xi
    Xi_new = tf.sign(Xi) * tf.maximum(tf.abs(Xi)-lam*lr, 0.0)
    model.Xi.assign(Xi_new)

    if ((epoch + 1) % 50 == 0):
        print(f"Epoch: {epoch+1}: loss = {loss.numpy():.6f}", end="\r")
    
    if ((epoch + 1) % 200 == 0):
        print()
        print_net_params(model)
        print()

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