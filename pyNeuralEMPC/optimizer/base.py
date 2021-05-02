import numpy as np
import cyipopt



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


class Optimizer:

    FAIL = 1
    SUCCESS = 0

    def __init__(self):
        pass

    def solve(self, problem, domain_constraint) -> np.ndarray:
        """Solve the problem


        Returns:
            np.ndarray: The optimal control vector

        """
        raise NotImplementedError("")

class Ipopt(Optimizer):
    def __init__(self, max_iteration=500, init_with_last_result=False, mu_strategy="monotone", mu_target=0, mu_linear_decrease_factor=0.2,\
         alpha_for_y="primal", obj_scaling_factor=1, nlp_scaling_max_gradient=100.0):

        super(Optimizer).__init__()

        self.max_iteration=max_iteration
        self.mu_strategy = mu_strategy
        self.mu_target = mu_target
        self.mu_linear_decrease_factor = mu_linear_decrease_factor
        self.alpha_for_y = alpha_for_y
        self.obj_scaling_factor = obj_scaling_factor
        self.nlp_scaling_max_gradient = nlp_scaling_max_gradient

        self.init_with_last_result = init_with_last_result
        self.prev_result = None


    def solve(self, problem, domain_constraint):
        
        if self.init_with_last_result:
            #TODO implement this
            raise NotImplementedError("")

        x0 = problem.get_init_value()

        lb = domain_constraint.get_lower_bounds()
        ub = domain_constraint.get_upper_bounds()

        cl = problem.get_constraint_lower_bounds()
        cu = problem.get_constraint_upper_bounds()

        nlp = cyipopt.Problem(
            n=len(x0),
            m=len(cl),
            problem_obj=problem,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
            )
        
        nlp.addOption('max_iteration',             self.max_iteration)
        nlp.addOption('mu_strategy',               self.mu_strategy) 
        nlp.addOption('mu_target',                 self.mu_target)
        nlp.addOption('mu_linear_decrease_factor', self.mu_linear_decrease_factor)
        nlp.addOption('alpha_for_y',               self.alpha_for_y)
        nlp.addOption('obj_scaling_factor',        self.obj_scaling_factor)
        nlp.addOption('nlp_scaling_max_gradient',  self.nlp_scaling_max_gradient)
            

        x, info = nlp.solve(x0)

        if info["status"] == 0 or info["status"] == 1 :
            self.prev_result = x
            return Optimizer.SUCCESS

        return Optimizer.FAIL