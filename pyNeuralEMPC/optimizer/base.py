import numpy as np
from ..constraints import *
from ..model.base import Model
from ..objective.base import ObjectiveFunc

class ProblemInterface():
    def objective(self, x):
        raise NotImplementedError("")

    def gradient(self, x):
        raise NotImplementedError("")

    def constraints(self, x):
        raise NotImplementedError("")
    
    def hessianstructure(self):
        raise NotImplementedError("")
    
    def hessian(self, x):
        raise NotImplementedError("")

    def jacobian(self, x):
        raise NotImplementedError("")

    def get_constraint_lower_bounds(self):
        raise NotImplementedError("")

    def get_constraint_upper_bounds(self):
        raise NotImplementedError("")

    def get_init_value(self):
        raise NotImplementedError("")


class ProblemFactory():
    def __init__(self):

        self.x0 = None
        self.p = None
        self.tvp = None
        self.objective = None 
        self.constraints = None
        self.integrator = None
        self.u_param = None

    def getProblemInterface(self) -> ProblemInterface:
        if (not self.x0 is None) and \
           (not self.objective is None) and \
           (not self.constraints is None) and \
           (not self.integrator is None):

            return self._process()
        
        if self.x0 is None:
            raise RuntimeError("Not ready yet ! x0 is missing")
        if self.objective is None:
            raise RuntimeError("Not ready yet ! objective is missing")
        if self.constraints is None:
            raise RuntimeError("Not ready yet ! constraints is missing")
        if self.integrator is None:
            raise RuntimeError("Not ready yet ! integrator is missing")

    def set_integrator(self, integrator):
        self.integrator = integrator

    def set_x0(self, x0: np.array):
        self.x0 = x0

    def set_p(self, p: np.array):
        self.p = p

    def set_tvp(self, tvp: np.ndarray):
        self.tvp = tvp

    def set_objective(self, obj: ObjectiveFunc):
        self.objective = obj

    def set_constraints(self, ctrs: list):
        self.constraints = ctrs

    def set_u_param(self, u_param: list):
        self.u_param = u_param


    def _process(self):
        raise NotImplementedError("")

class Optimizer:

    FAIL = 1
    SUCCESS = 0

    def __init__(self):
        pass

    def get_factory(self) -> ProblemFactory:
        """
        Return the solver associeted factory.
        """

    def solve(self, problem: ProblemInterface, domain_constraint: DomainConstraint, x_init=None) -> np.ndarray:
        """Solve the problem


        Returns:
            np.ndarray: The optimal control vector

        """
        raise NotImplementedError("")

