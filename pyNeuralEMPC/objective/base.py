import numpy as np 
import inspect

class ObjectiveFunc:
    def __init__(self):
        pass

    def forward(self, x, u, x0, p=None, tvp=None):
        raise NotImplementedError("")

    def gradient(self, x, u, x0, p=None, tvp=None):
        raise NotImplementedError("")

    def hessian(self, x, u, x0, p=None, tvp=None):
        raise NotImplementedError("")


class ManualObjectifFunc(ObjectiveFunc):
    def __init__(self, func, grad_func, hessian_func):
        super(ManualObjectifFunc).__init__(self, func)
        
        self.grad_func    = grad_func
        self.hessian_func = hessian_func

    def forward(self, x, u, x0, p=None, tvp=None):
        return self.func(x, u, x0, p, tvp)

    def gradient(self, x, u, x0, p=None, tvp=None):
        return self.grad_func(x, u, x0, p, tvp)

    def hessian(self, x, u, x0, p=None, tvp=None):
        return self.hessian_func(x, u, x0, p, tvp)
    
