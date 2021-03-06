import numpy as np 

class DomainConstraint:
    
    def __init__(self, states_constraint: list, control_constraint: list):

        if len(states_constraint) == 0:
            raise ValueError("States constraint empty !")

        if len(control_constraint) == 0:
            raise ValueError("Control constraint empty !")

        if (len(set([len(element) for element in states_constraint])) != 1) or len(states_constraint[0])!=2:
            raise ValueError("Your states constraint must be a list of bound couple  ! [(lower_bound, upper_bound), ...]")

        if (len(set([len(element) for element in states_constraint])) != 1) or len(states_constraint[0])!=2:
            raise ValueError("Your control constraint must be a list of bound couple ! [(lower_bound, upper_bound), ...]")

        
        self.states_constraint = states_constraint
        self.control_constraint = control_constraint

    def get_dim(self, H):
        return len(self.states_constraint), len(self.control_constraint)
        
    def get_lower_bounds(self, H):
        return [ element[0] for element in self.states_constraint ]*H + [ element[0] for element in self.control_constraint]*H 

    def get_upper_bounds(self, H):
        return [ element[1] for element in self.states_constraint ]*H + [ element[1] for element in self.control_constraint]*H 
    
    def get_type(self):
        return Constraint.EQ_TYPE

        
class Constraint:
    EQ_TYPE = 0
    INEQ_TYPE = 1
    INTER_TYPE = 2
    """
    This class will implement a constraint under a formula declaration for a given state. 

    TODO add a mecanism to filter if it's an eq or ineq
    """
    def forward(self, x, u, p=None, tvp=None):
        pass

    def jacobian(self, x, u, p=None, tvp=None):
        pass

    def get_lower_bounds(self, H):
        raise NotImplementedError()

    def get_upper_bounds(self, H):
        raise NotImplementedError()

    def get_type(self, H=None):
        if (self.get_upper_bounds(H) == self.get_lower_bounds(H)).all() and (self.get_lower_bounds(H) == 0).all() :
            return Constraint.EQ_TYPE
        if (self.get_upper_bounds(H) == np.inf).all() and (self.get_lower_bounds(H) == 0).all():
            return Constraint.INEQ_TYPE
        else:
            return Constraint.INTER_TYPE


class EqualityConstraint(Constraint):
    def forward(self, x, u, p=None, tvp=None):
        raise NotImplementedError()

    def jacobian(self, x, u, p=None, tvp=None):
        raise NotImplementedError()

    def get_dim(self):
        raise NotImplementedError()
    
    def get_lower_bounds(self, H):
        return np.zeros(int(self.get_dim(H)))

    def get_upper_bounds(self, H):
        return np.zeros(int(self.get_dim(H)))

class InequalityConstraint(Constraint):
    def forward(self, x, u, p=None, tvp=None):
        raise NotImplementedError()

    def jacobian(self, x, u, p=None, tvp=None):
        raise NotImplementedError()

    def get_dim(self):
        raise NotImplementedError()
    
    def get_lower_bounds(self, H):
        return np.zeros(int(self.get_dim(H)))

    def get_upper_bounds(self, H):
        return np.ones(int(self.get_dim(H)))*np.inf

