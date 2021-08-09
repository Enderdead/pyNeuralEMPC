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
        self.objective = None 
        self.constraints = None
        self.model = None

    def getProblemInterface(self) -> ProblemInterface:
        if (not self.x0 is None) and \
           (not self.objective is None) and \
           (not self.constraints is None) and \
           (not self.model is None):

            return self._process()

    def set_x0(self, x0: np.array):
        self.x0 = x0

    def set_objective(self, obj: ObjectiveFunc):
        self.objective

    def set_constraints(self, ctrs: list):
        self.constraints = ctrs

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

    def solve(self, problem: ProblemInterface, domain_constraint: DomainConstraint) -> np.ndarray:
        """Solve the problem


        Returns:
            np.ndarray: The optimal control vector

        """
        raise NotImplementedError("")

