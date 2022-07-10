import numpy as np
import jax.numpy as jnp
import jax

from .base import ObjectiveFunc


def _check_func(func, state_dim, control_dim, constant_dim):
    try:
        _ = jax.jacrev(func)(jnp.zeros(state_dim), jnp.zeros(control_dim), jnp.zeros(constant_dim))
    except AssertionError:
        return False
    return True


class JAXObjectifFunc(ObjectiveFunc):
    def __init__(self, func):
        super(JAXObjectifFunc).__init__()

        # TODO find a way to get checking input dim
        """
        if not _check_func(func):
            raise ValueError("Your function is not differentiable w.r.t the JAX library")
        """
        self.func = func
        self.cached_hessian_structure = dict()

    def forward(self, states, u, p=None, tvp=None):

        return self.func(states, u, p, tvp)

    def gradient(self, states, u, p=None, tvp=None):
        
        #TODO maybe problem
        grad_states = jax.grad(self.func, argnums=0)(states, u, p, tvp).reshape(-1)
        grad_u = jax.grad(self.func, argnums=1)(states, u, p, tvp).reshape(-1)

        result_list = [grad_states, grad_u]
        final_res = jnp.concatenate(result_list, axis=0).to_py()
        final_res = jnp.nan_to_num(final_res, nan=0.0)
        return final_res

    def hessian(self, states, u, p=None, tvp=None):

        hessians = jax.hessian(self.func, argnums=[0,1])(states, u, p, tvp)
        a = hessians[0][0].reshape(states.shape[0]*states.shape[1], states.shape[0]*states.shape[1])
        b = hessians[0][1].reshape(states.shape[0]*states.shape[1], u.shape[0]*u.shape[1])

        c = hessians[1][0].reshape(u.shape[0]*u.shape[1], states.shape[0]*states.shape[1])
        d = hessians[1][1].reshape(u.shape[0]*u.shape[1], u.shape[0]*u.shape[1])

        ab = np.concatenate([a, b], axis=1)
        cd = np.concatenate([c, d], axis=1)

        final_hessian = np.concatenate([ab, cd], axis=0)

        return final_hessian

    def hessianstructure(self, H, model):
        if (H, model) in self.cached_hessian_structure.keys():
            return self.cached_hessian_structure[(H, model)]
        else:
            result = self._compute_hessianstructure(H, model)
            self.cached_hessian_structure[(H, model)] = result
            return result

    def _compute_hessianstructure(self, H, model, nb_sample=3):
        hessian_map = None

        for _ in range(nb_sample):

            x_random = np.random.uniform(size=(H, model.x_dim))
            u_random = np.random.uniform(size=(H, model.u_dim))
            p_random = None
            tvp_random = None

            if model.p_dim > 0:
                p_random = np.random.uniform(size=model.p_dim)
            
            if model.tvp_dim > 0:
                tvp_random = np.random.uniform(size=(H, model.tvp_dim))
            
            hessian  = self.hessian(x_random, u_random, p=p_random, tvp=tvp_random)

            if hessian_map is None:
                hessian_map = (hessian!= 0.0).astype(np.float64)
            else:
                hessian_map += (hessian!= 0.0).astype(np.float64)
                hessian_map = hessian_map.astype(np.bool).astype(np.float64)
        return hessian_map
