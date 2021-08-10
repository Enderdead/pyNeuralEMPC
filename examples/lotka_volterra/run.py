import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--force", help="Recompute even if a cache file exists !",
                    action="store_true")
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from dyna_toy_envs.virtuel.lotka_volterra import lotka_volterra_energy
from dyna_toy_envs.display import plot_experience
import numpy as np
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
import pyNeuralEMPC as nEMPC

#from utils import *
import progressbar
import pysindy
import numpy as np 
import jax.numpy as jnp

import tensorflow as tf
import pickle


### WARNING ####
# We need to normalize every thing...
# x = x/30. -1 
# y = y/30. -1 
# u = u/50.0 - 1



NB_STEPS = 700
Hb = 2
U_MAX = 60
U_MIN = 0
REFRESH_EVERY = 2
TRAINING_SIZE = 600
DT = 0.05
X_MAX = 60


X_MAX = X_MAX/30.0 - 1
U_MAX = U_MAX/50.0 -1 
U_MIN = U_MIN/50.0 -1 

H = Hb

system = lotka_volterra_energy(dt=DT, init_state=np.array([50.0, 5.0]), H=Hb)

keras_model = tf.keras.models.load_model("./nn_model.h5")


input_tensor = tf.constant(np.array([[50.0,5.0,0.0],[50.0,5.0,0.0]]))



model_nmpc = nEMPC.model.tensorflow.KerasTFModel(keras_model, x_dim=2, u_dim=1)

def forward_jnp(x, u, p=None, tvp=None):
    result = jnp.repeat( u[:,0].reshape(-1,1), x.shape[-1],  axis=1)*x/2.0 
    return result

model_nmpc = nEMPC.model.jax.DiffDiscretJaxModel(forward_jnp, x_dim=2, u_dim=1, vector_mode=True)


constraints_nmpc = [nEMPC.constraints.DomainConstraint(
    states_constraint=[[-np.inf, X_MAX], [-np.inf, np.inf]],
    control_constraint=[[U_MIN, U_MAX]]),]

integrator = nEMPC.integrator.discret.DiscretIntegrator(model_nmpc, H)

class LotkaCost:
    def __init__(self, cost_vec):
        self.cost_vec = cost_vec

    def __call__(self, x, u, p=None, tvp=None):
        return jnp.sum(u.reshape(-1)*self.cost_vec.reshape(-1))


cost_func = LotkaCost(jnp.array([1.1,]*2))

objective_func = nEMPC.objective.jax.JAXObjectifFunc(cost_func)


MPC = nEMPC.controller.NMPC(integrator, objective_func, constraints_nmpc, H, DT)

curr_x, curr_cost = system.get_init_state()

cost_func.cost_vec = jnp.array(curr_cost)
aa = MPC.next(curr_x)

1/0
u, pred = MPC.next(curr_x)
1/0
"""
controller = NNLokeMPC(keras_model, DT, Hb, max_state=[X_MAX+0.0001, 1e20], u_min=U_MIN, u_max=U_MAX, derivative_model=False)




curr_x, curr_cost = system.get_init_state()

def un_normalize_u(u):
    return (u+1)*50.0



u, pred = controller.next(normalize_state(curr_x),curr_cost, verbosity=1)

states = [curr_x, ]
u_list = [u[0]]
cost_list   = [curr_cost[0]]
last_refresh = 0 
decision_making = [(u, pred),]
#sresult_u = us["x"]
for i in progressbar.progressbar(range(NB_STEPS)):
    curr_x, cpst = system.next(un_normalize_u(u[i-last_refresh]))
    #if ((curr_x[0]/30.0)-1)>X_MAX:
    #    break
    states.append(curr_x.copy())
    cost_list.append(cpst[0])
    u_list.append(u[i-last_refresh])
    if i%REFRESH_EVERY == 0:
        #recompute x0 = [1.0,]*self.H + [init_state[0],]*self.H + [init_state[1],]*self.H
        try:
            u,pred = controller.next(normalize_state(curr_x), cpst, delta=None, verbosity=5)#REFRESH_EVERY
        except ZeroDivisionError:
            i = 0
            while True:
                i+=1
                print("retry with random try {}".format(i))
                try:
                    u,pred = controller.next(normalize_state(curr_x), cpst, delta=None, verbosity=5, x0=np.random.uniform(low=-1, high=1, size=Hb*3))
                    break
                except ZeroDivisionError:
                    pass
        decision_making.append((u,pred))
        last_refresh = i+1


#states  = unnormalize(np.stack(states))

t = np.arange(0, len(states), 1)

u_list = (np.array(u_list)+1)*50.0



x = np.stack(states)[:,0]
y = np.stack(states)[:,1]


plt.figure(figsize=(9.5,5))
a = plt.plot(t*DT, x, label="x")
b = plt.plot(t*DT, y, label="y")
c = plt.plot(t*DT, u_list, label="u")

for local_t, cost in zip(t*DT, cost_list):
    if cost<0.5:
        plt.axvspan(local_t, local_t+DT, facecolor='b', alpha=0.2)


plt.xlabel("Temps (seconds)", size=15)
plt.ylabel("Valeur", size=15)

red_patch = mpatches.Patch(color='b', alpha=0.2, label='Periode à faible cout')
plt.legend(handles=[red_patch, a[0], b[0], c[0]], framealpha=1.0, loc=0, fontsize=11)

plt.tight_layout()
plt.show()


# Compute total cost 

b = np.array(cost_list).reshape(1, -1)
a = np.array(u_list).reshape(1, -1)

total_cost = np.dot(a, b.T)
"""