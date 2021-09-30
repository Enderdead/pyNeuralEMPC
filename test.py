import numpy as np 
import tensorflow as tf 
import jax.numpy as jnp

from pyNeuralEMPC.model.tensorflow import KerasTFModel, KerasTFModelRollingInput
import pyNeuralEMPC as nEMPC





@tf.function
def model(x):        #  x - x*y ; -0.5 +   u  + x * y
    #result = tf.concat([x[:,0:1] - x[:,0:1]*x[:,1:2], -1*x[:,1:2]+x[:,2:3] +x[:,0:1]*x[:,1:2]     ], axis=1)
    result = tf.concat([x[:,2:3] , x[:,4:5]     ], axis=1)

    return result


class FakeModel:

    def __init__(self):
        self.input_shape = (-1, 3 )
        self.output_shape = (-1, 2)

    @tf.function
    def __call__(self, x):
        return tf.concat([x[:,2:3]*x[:,2:3] , x[:,5:6]*x[:,5:6]/2     ], axis=1)

    @tf.function
    def predict(self, x):
        return tf.concat([x[:,2:3]*x[:,2:3] , x[:,5:6]*x[:,5:6]/2     ], axis=1)

fake = FakeModel()




x_past = np.array([[0.2,0.1]])
u_past  =np.array([[0.0]])

x0 = np.array([[-0.2,-0.1]])

x = np.array([[0.2,0.1],
                 [0.10832,0.0908]],dtype=np.float32)

u = np.array([[0.01309788],[-0.09964662]], dtype=np.float32)


test =   KerasTFModelRollingInput(fake, 2, 1, forward_rolling=True)
test.set_prev_data(x_past, u_past)

H = 10

class LotkaCost:
    def __init__(self, cost_vec):
        self.cost_vec = cost_vec

    def __call__(self, x, u, p=None, tvp=None):
        return jnp.sum(jnp.square(u.reshape(-1)-2.0))


cost_func = LotkaCost(jnp.array([1.1,]*25))
DT  = 1
integrator = nEMPC.integrator.discret.DiscretIntegrator(test, H)





constraints_nmpc = [nEMPC.constraints.DomainConstraint(
    states_constraint=[[-np.inf, np.inf], [-np.inf, np.inf]],
    control_constraint=[[-np.inf, np.inf]]),]

objective_func = nEMPC.objective.jax.JAXObjectifFunc(cost_func)

MPC = nEMPC.controller.NMPC(integrator, objective_func, constraints_nmpc, H, DT)

pred, u  = MPC.next(x_past.reshape(-1))
