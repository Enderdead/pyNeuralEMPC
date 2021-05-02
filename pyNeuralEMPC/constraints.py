import numpy as np 

class DomainConstraint:
    # Implement NONE , 

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

    def get_dim(self):
        return len(self.states_constraint), len(self.control_constraint)
        
    def get_lower_bounds(self):
        raise NotImplementedError("")

    def get_upper_bounds(self):
        raise NotImplementedError("")
        
class RelationConstraint:
    """
    This class will implement a constraint under a formula declaration for a given state. 
    """
    pass