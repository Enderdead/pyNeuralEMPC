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

    def forward(self, states, u, p=None, tvp=None):

        return self.func(states, u, p, tvp)

    def gradient(self, states, u, p=None, tvp=None):
        
        grad_states = jax.grad(self.func, argnums=0)(states, u, p, tvp)
        grad_u = jax.grad(self.func, argnums=1)(states, u, p, tvp)

        result_list = [grad_states, grad_u]

        if not tvp is None:
            result_list.append(jax.grad(self.func, argnums=3)(states, u, p, tvp))

        if not p is None:
            result_list.append(jax.grad(self.func, argnums=2)(states, u, p, tvp))

        return jnp.concatenate(result_list, axis=0).to_py()

    def hessian(self, states, u, p=None, tvp=None):
        # TODO revoir
        hessian_states = jax.hessian(self.func, argnums=0)(states, u, p, tvp)
        hessian_u = jax.hessian(self.func, argnums=1)(states, u, p, tvp)

        return hessian_states, hessian_u