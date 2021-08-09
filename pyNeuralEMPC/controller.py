import numpy as np 
from .optimizer.base import Ipopt




class NMPC():
    def __init__(self, model, objective_func, constraint_list, H, DT, optimizer=Ipopt(), use_hessian=True):

        self.model = model
        self.objective_func = objective_func
        self.constraint_list = constraint_list
        self.H = H
        self.DT = DT 
        self.optimizer = optimizer
        self.use_hessian = use_hessian


        # TODO VERIF EVERY THINGS !

    def next(self, x0: np.array, p=None, tvp=None):
        
        # first check x0, p and tvp dim
        assert len(x0.shape) == 1, "x0 must be a vector"
        assert x0.shape[0] == self.model.x_dim, "x0 dim must set according to your model !"
        
        if not p is None:
            assert len(x0.shape) == 1, "p must be a vector"
            assert p.shape[0] == self.model.p_dim, "p dim must set according to your model !"

        if not tvp is None:
            assert len(tvp.shape) == 1, "tvp must be a vector"
            assert tvp.shape[0] == self.model.p_dim, "tvp dim must set according to your model !"


    