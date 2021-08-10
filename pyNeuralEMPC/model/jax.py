from .base import Model

import numpy as np
import jax.numpy as jnp
import jax

def _check_func(func, state_dim, control_dim, p_dim, tvp_dim):
    try: #TODO c'est le bordel
        _ = jax.jacrev(func)(jnp.zeros((2, state_dim)),\
             jnp.zeros((2, control_dim)),\
             p=None if p_dim is None else jnp.zeros(p_dim),\
             tvp=None if tvp_dim is None else jnp.zeros((2, tvp_dim)))
    except AssertionError:
        return False
    return True

class DiffDiscretJaxModel(Model):
    
    def __init__(self, forward_func, x_dim: int, u_dim: int, p_dim=0, tvp_dim=0, vector_mode=False):
        if not _check_func(forward_func, x_dim, u_dim, p_dim, tvp_dim):
            raise ValueError("Your function is not differentiable w.r.t the JAX library")

        self.forward_func = forward_func
        self.vector_mode = vector_mode

        super(DiffDiscretJaxModel, self).__init__(x_dim, u_dim, p_dim, tvp_dim)


    def _split(self, x):# forcement en vector mode 
        states = x[:,0:self.x_dim]
        u = x[:,self.x_dim:self.x_dim+self.u_dim]
        tvp = None if self.tvp_dim == 0 else x[:,self.x_dim+self.u_dim:self.x_dim+self.u_dim+self.tvp_dim]
        p   = None if self.p_dim   == 0 else x[:,self.x_dim+self.u_dim+self.tvp_dim:self.x_dim+self.u_dim+self.tvp_dim+self.p_dim]
        return states, u, tvp, p


    def forward(self, x):
        if self.vector_mode:
            states, u, tvp, p = self._split(x)
            return self.forward_func(states, u, p=p, tvp=tvp)
        else:
            raise NotImplementedError("")


    def jacobian(self, x):
        if self.vector_mode:
                
            states, u, tvp, p = self._split(x)

            argnums_list = [0, 1]
            if self.tvp_dim>0:
                argnums_list.append(3)
            if self.p_dim>0:
                argnums_list.append(2)
            jacobians = list(jax.jacobian(self.forward_func, argnums=argnums_list)(states, u, p, tvp))
            #jacobians = [ np.stack([jacobian[i,:,i,:].to_py()  for i in range(x.shape[0])]) for jacobian in jacobians]


            jacobians[0] = jacobians[0].reshape(self.x_dim*x.shape[0], self.x_dim*x.shape[0] )
            jacobians[1] = jacobians[1].reshape(self.x_dim*x.shape[0], self.u_dim*x.shape[0] )

            jaco =  jnp.concatenate(jacobians, axis=1).to_py()
            return jaco
        else:
            raise NotImplementedError("")

    def hessian(self, x):
        raise NotImplementedError("")

