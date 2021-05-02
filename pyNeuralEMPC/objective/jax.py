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
    def __init__(self, func, state_dim, control_dim, constant_dim):
        super(JAXObjectifFunc).__init__(self, func, state_dim, control_dim, constant_dim)

        if not _check_func(func):
            raise ValueError("Your function is not differentiable w.r.t the JAX library")

    def forward(self, states, u, p=None, tvp=None):

        return self.func(states, u, p, tvp)

    def gradient(self, states, u, p=None, tvp=None):
        
        grad_states = jax.grad(func, argnums=0)(states, u, p, tvp)
        grad_u = jax.grad(func, argnums=1)(states, u, p, tvp)

        return grad_states, grad_u

    def hessian(self, states, u, tvp=None):

        hessian_states = jax.hessian(func, argnums=0)(states, u, p, tvp)
        hessian_u = jax.hessian(func, argnums=1)(states, u, p, tvp)

        return hessian_states, hessian_u