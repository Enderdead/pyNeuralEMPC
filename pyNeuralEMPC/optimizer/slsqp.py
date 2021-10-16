import warnings
from .base import Optimizer, ProblemFactory, ProblemInterface

import numpy as np 
from scipy.optimize import minimize, Bounds
from functools import lru_cache
import time

class SlsqpProblem(ProblemInterface):
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
        states, u, tvp, p = self._split(x)
        res =  self.objective_func.forward(states, u, p=p, tvp=tvp)

        return res

    def gradient(self, x):
        states, u, tvp, p = self._split(x)

        res =  self.objective_func.gradient(states, u, p=p, tvp=tvp)

        return res

    def constraints(self, x, eq=True):
        # TODO not taking into account other constraint than integrator ones      
        warnings.warn("Not taking into account other constraint than integrator ones  !")  
        states, u, tvp, p = self._split(x)

        contraints_forward_list = [self.integrator.forward(states, u, self.x0, p=p, tvp=tvp),]

        #for ctr in self.constraints_list:
        #    contraints_forward_list.append(ctr.forward(states, u, p=p, tvp=tvp))

        return np.concatenate(contraints_forward_list, axis=0)

    
    def hessianstructure(self):
        raise NotImplementedError("Not needed")

    
    def hessian(self, x, lagrange, obj_factor):
        raise NotImplementedError("Not needed")

    def jacobian(self, x, eq=True):
        # TODO not taking into account other constraint than integrator ones      
        warnings.warn("Not taking into account other constraint than integrator ones  !")
        states, u, tvp, p = self._split(x)

        contraints_jacobian_list = [self.integrator.jacobian(states, u, self.x0, p=p, tvp=tvp),]

        for ctr in self.constraints_list:
            contraints_jacobian_list.append(ctr.jacobian(states, u, p=p, tvp=tvp))

        return np.concatenate(contraints_jacobian_list, axis=0)

    def get_constraints_dict(self):
        # TODO not taking into account other constraint than integrator ones        
        return [{'type': 'eq',
           'fun' : lambda x: self.constraints(x),
           'jac' : lambda x: self.jacobian(x)}]


    def get_init_value(self):
        return self.x0




class SlsqpProblemFactory(ProblemFactory):
    def _process(self):
        return SlsqpProblem(self.x0, self.objective, self.constraints, self.integrator, p=self.p, tvp=self.tvp)


class Slsqp(Optimizer):
    def __init__(self, max_iteration=200, tolerance=0.5e-6, verbose=1, init_with_last_result=False, nb_max_try=15):

        super(Slsqp, self).__init__()

        self.max_iteration=max_iteration
        self.verbose = verbose
        self.tolerance=tolerance
        self.init_with_last_result = init_with_last_result

        self.prev_result = None
        self.nb_max_try = nb_max_try
        self.retry_max_iter = 15

    def get_factory(self):
        return SlsqpProblemFactory()


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
        bounds = Bounds(lb, ub)

        
        res = minimize(problem.objective, x_init, method="SLSQP", jac=problem.gradient, \
            constraints=problem.get_constraints_dict(), options={'maxiter': self.max_iteration, 'ftol': self.tolerance, 'disp': True, 'iprint':self.verbose}, bounds=bounds)

        if not res.success :
            warnings.warn("Process do not converge ! ")

            if np.max(problem.constraints(res.x))>1e-5:
                for i in range(self.nb_max_try):
                    print("RETRY SQP optimization")
                    x_init = np.concatenate( [np.concatenate( [x0,]*problem.integrator.H ), np.repeat(np.array([0.0,]*problem.integrator.model.u_dim),problem.integrator.H)])
                    res = minimize(problem.objective, x_init, method="SLSQP", jac=problem.gradient, \
                        constraints=problem.get_constraints_dict(), options={'maxiter': self.max_iteration, 'ftol': self.tolerance*(2.0**i), 'disp': True, 'iprint':self.verbose}, bounds=bounds)
                    if np.max(problem.constraints(res.x))<1e-5:
                        break

            assert np.max(problem.constraints(res.x))<1e-5, "Result not compliant ! "

        self.prev_result = res.x

        return Optimizer.SUCCESS