from .base import Optimizer, ProblemFactory, ProblemInterface

import numpy as np 
import cyipopt
import time

class IpoptProblem(ProblemInterface):
    def __init__(self, x0, objective_func, constraints, integrator, p=None, tvp=None):
        self.x0 = x0
        self.objective_func = objective_func 
        self.constraints_list = constraints
        self.integrator = integrator
        self.x_dim, self.u_dim, self.p_dim, self.tvp_dim = self.integrator.model.x_dim, self.integrator.model.u_dim, self.integrator.model.p_dim, self.integrator.model.tvp_dim  
        self.H = self.integrator.H
        self.p = p
        self.tvp = tvp

    def _split(self, x):
        prev_idx = 0
        states = x[prev_idx:prev_idx+self.x_dim*self.H]
        prev_idx += self.x_dim*self.H

        u = x[prev_idx:prev_idx+self.u_dim*self.H]
        prev_idx += self.u_dim*self.H

        return states.reshape(self.H, self.x_dim), u.reshape(self.H, self.u_dim), self.tvp, self.p

    def objective(self, x):
        # TODO split les dt !!!!
        states, u, tvp, p = self._split(x)
        res =  self.objective_func.forward(states, u, p=p, tvp=tvp)

        return res

    def gradient(self, x):
        states, u, tvp, p = self._split(x)

        res =  self.objective_func.gradient(states, u, p=p, tvp=tvp)

        return res

    def constraints(self, x):
        states, u, tvp, p = self._split(x)

        contraints_forward_list = [self.integrator.forward(states, u, self.x0, p=p, tvp=tvp),]

        for ctr in self.constraints_list:
            contraints_forward_list.append(ctr.forward(states, u, p=p, tvp=tvp))

        return np.concatenate(contraints_forward_list)

    
    def hessianstructure(self):
        hessian_map_objective = self.objective_func.hessianstructure(  self.integrator.H, self.integrator.model)

        hessian_map_integrator = self.integrator.hessianstructure()

        final_hessian_map  = (hessian_map_objective + hessian_map_integrator).astype(np.bool).astype(np.float32)

        return np.nonzero(np.tril(final_hessian_map))#np.nonzero(np.ones_like(final_hessian_map))
    

    
    def hessian(self, x, lagrange, obj_factor):
        states, u, tvp, p = self._split(x)
        
        hessian_matrice = np.zeros((x.shape[0], x.shape[0]))

        hessian_matrice += obj_factor*self.objective_func.hessian(states, u, p=p, tvp=tvp)

        integrator_hessian_matrice = self.integrator.hessian(states, u, self.x0, p=p, tvp=tvp)
        
        # TODO add constraints 
        for idx, lagrange_coef in enumerate(lagrange): # TODO remove this loop (vec comp)
            hessian_matrice+=lagrange_coef*integrator_hessian_matrice[idx]


        row, col = self.hessianstructure()

        return hessian_matrice[row, col]

    def jacobian(self, x):
        states, u, tvp, p = self._split(x)

        contraints_jacobian_list = [self.integrator.jacobian(states, u, self.x0, p=p, tvp=tvp),]

        for ctr in self.constraints_list:
            contraints_jacobian_list.append(ctr.jacobian(states, u, p=p, tvp=tvp))

        return np.concatenate(contraints_jacobian_list, axis=0)

    def get_init_value(self):
        return self.x0

    def get_constraint_lower_bounds(self):
        #TODO refaire 
        #return self.integrator.get_lower_bounds()
        #ajouter integrator
        return sum([ [  ctr.get_lower_bounds(self.H) for ctr in [self.integrator,]+self.constraints_list  ]], list())

    def get_constraint_upper_bounds(self):
        #TODO refaire 
        #return self.integrator.get_upper_bounds()

        return sum([ [  ctr.get_upper_bounds(self.H) for ctr in [self.integrator,]+self.constraints_list  ]], list())


class IpoptProblemFactory(ProblemFactory):
    def _process(self):
        return IpoptProblem(self.x0, self.objective, self.constraints, self.integrator, p=self.p, tvp=self.tvp)


class Ipopt(Optimizer):
    def __init__(self, max_iteration=500, init_with_last_result=False, mu_strategy="monotone", mu_target=0, mu_linear_decrease_factor=0.2,\
         alpha_for_y="primal", obj_scaling_factor=1, nlp_scaling_max_gradient=100.0):

        super(Ipopt, self).__init__()

        self.max_iteration=max_iteration
        self.mu_strategy = mu_strategy
        self.mu_target = mu_target
        self.mu_linear_decrease_factor = mu_linear_decrease_factor
        self.alpha_for_y = alpha_for_y
        self.obj_scaling_factor = obj_scaling_factor
        self.nlp_scaling_max_gradient = nlp_scaling_max_gradient

        self.init_with_last_result = init_with_last_result
        self.prev_result = None


    def get_factory(self):
        return IpoptProblemFactory()


    def solve(self, problem, domain_constraint):
        x0 = problem.get_init_value()

        if self.init_with_last_result and not (self.prev_result is None):
            x_dim = problem.integrator.model.x_dim
            u_dim = problem.integrator.model.u_dim
            x_init = np.concatenate([self.prev_result[x_dim:x_dim*problem.integrator.H], # x[1]-x[H]
              self.prev_result[x_dim*(problem.integrator.H-1):x_dim*problem.integrator.H], # x[H]
              self.prev_result[x_dim*problem.integrator.H+u_dim:(x_dim+u_dim)*problem.integrator.H],  # u[1] - u[H]
              self.prev_result[x_dim*problem.integrator.H+u_dim*(problem.integrator.H-1):x_dim*problem.integrator.H+u_dim*problem.integrator.H] ], axis=0) # u[H]
        else:
            x_init = np.concatenate( [np.concatenate( [x0,]*problem.integrator.H ), np.repeat(np.array([0.0,]*problem.integrator.model.u_dim),problem.integrator.H)])


        # TODO find a better way to get the horizon variable
        lb = domain_constraint.get_lower_bounds(problem.integrator.H)
        ub = domain_constraint.get_upper_bounds(problem.integrator.H)

        cl = problem.get_constraint_lower_bounds()
        cu = problem.get_constraint_upper_bounds()

        nlp = cyipopt.Problem(
            n=len(x_init),
            m=len(cl),
            problem_obj=problem,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu
            )

        nlp.addOption('max_iter',          self.max_iteration)#  self.max_iteration)# 
        #nlp.addOption('derivative_test', 'first-order')
        #nlp.addOption('derivative_test_print_all', 'yes')
        #nlp.addOption('point_perturbation_radius',1e-1)
        #nlp.addOption('derivative_test_perturbation',1e-1)

        #nlp.addOption('mu_strategy',               self.mu_strategy) 
        #nlp.addOption('mu_target',                 self.mu_target)
        #nlp.addOption('mu_linear_decrease_factor', self.mu_linear_decrease_factor)
        #nlp.addOption('alpha_for_y',               self.alpha_for_y)
        #nlp.addOption('obj_scaling_factor',        self.obj_scaling_factor)
        #nlp.addOption('nlp_scaling_max_gradient',  self.nlp_scaling_max_gradient)
        nlp.addOption("tol", 1e-1)
        nlp.addOption("acceptable_tol",1e-4)
        nlp.addOption("print_level", 0)


        x, info = nlp.solve(x_init)
        self.prev_result = x
        if info["status"] == 0 or info["status"] == 1 :
            self.prev_result = x
            return Optimizer.SUCCESS

        return Optimizer.FAIL