import jax
import jax.numpy as jnp
import numpy as np

def _check_func(func, input_dim):
    try: #TODO c'est le bordel
        _ = jax.jacrev(func)(jnp.zeros((input_dim)))
    except AssertionError:
        return False
    return True


class ControlParam():
    def __init__(self, u_param_dim, u_dim):
        self.u_param_dim = u_param_dim
        self.u_dim       = u_dim

    def get_nb_var(self, H):
        raise NotImplementedError("")

    def forward(self, u_param):
        raise NotImplementedError("")

    def jacobian(self, u_param):
        raise NotImplementedError("")
    
    def hessian(self, u_param):
        raise NotImplementedError("")



class JaxControlParam(ControlParam):
    def __init__(self, func, u_param_dim, u_dim):
        if not _check_func(func, u_param_dim):
            raise ValueError("Given func isn't jax auto grad compatible !")
        self.func = func
        super().__init__(u_param_dim, u_dim)

    def forward(self, u_param):
        return self.func(u_param)

    def jacobian(self, u_param):
        jacobian_res = jax.jacobian(self.func, argnums=0)(u_param)
        return jacobian_res

    def hessian(self, u_param):
        hessian_res = jax.hessian(self.func, argnums=0)(u_param)
        return hessian_res

    