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

    def next(self, x0, p=None, tvp=None):
        raise NotImplementedError("")