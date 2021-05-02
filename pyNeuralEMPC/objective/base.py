import numpy as np 
import inspect

class ObjectiveFunc:
    def __init__(self, func, state_dim: int, control_dim: int, constant_dim: int):
        
        if len(inspect.signature(func).parameters) != 4:
            raise ValueError("Your objectif function must take 3 arguments (states, u, p(=None), and  tvp(=None) )")

        self.func = func
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.constant_dim = constant_dim

    def forward(self, x, u, x0, p=None, tvp=None):
        raise NotImplementedError("")

    def gradient(self, x, u, x0, p=None, tvp=None):
        raise NotImplementedError("")

    def hessian(self, x, u, x0, p=None, tvp=None):
        raise NotImplementedError("")


class ManualObjectifFunc(ObjectiveFunc):
    def __init__(self, func, grad_func, hessian_func, state_dim: int, control_dim: int, constant_dim: int):
        super(ManualObjectifFunc).__init__(self, func, state_dim, control_dim, constant_dim)
        
        self.grad_func    = grad_func
        self.hessian_func = hessian_func

    def forward(self, x, u, x0, p=None, tvp=None):
        return self.func(x, u, x0, p, tvp)

    def gradient(self, x, u, x0, p=None, tvp=None):
        return self.grad_func(x, u, x0, p, tvp)

    def hessian(self, x, u, x0, p=None, tvp=None):
        return self.hessian_func(x, u, x0, p, tvp)
    
