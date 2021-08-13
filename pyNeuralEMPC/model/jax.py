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


    def forward(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        if self.vector_mode:
            return self.forward_func(x, u, p=p, tvp=tvp)
        else:
            raise NotImplementedError("")


    def jacobian(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        if self.vector_mode:
            argnums_list = [0, 1]

            jacobians = list(jax.jacobian(self.forward_func, argnums=argnums_list)(x, u, p, tvp))

            jacobians[0] = jacobians[0].reshape(self.x_dim*x.shape[0], self.x_dim*x.shape[0] )
            jacobians[1] = jacobians[1].reshape(self.x_dim*x.shape[0], self.u_dim*x.shape[0] )

            jaco =  jnp.concatenate(jacobians, axis=1).to_py()
            return jaco
        else:
            raise NotImplementedError("")

    def hessian(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        if self.vector_mode:
            argnums_list = [0, 1]

            H = x.shape[0]
            X_dim = x.shape[1]
            U_dim = u.shape[1]

            hessians = list(jax.hessian(self.forward_func, argnums=argnums_list)(x, u, p, tvp))

            a = hessians[0][0].reshape(H  , X_dim,  H*X_dim, H*X_dim)
            b = hessians[0][1].reshape(H  , X_dim,  H*X_dim, H*U_dim)
            ab  = jnp.concatenate([a, b], axis=3)

            c = hessians[1][0].reshape(H  , X_dim,  H*U_dim, H*X_dim)
            d = hessians[1][1].reshape(H  , X_dim,  H*U_dim, H*U_dim)
            cd  = jnp.concatenate([c, d], axis=3)

            final_hessian = jnp.concatenate([ab, cd], axis=2)

            return final_hessian
        else:
            raise NotImplementedError("")

