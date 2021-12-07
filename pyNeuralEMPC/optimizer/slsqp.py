import warnings
from .base import Optimizer, ProblemFactory, ProblemInterface, Constraint

import numpy as np 
from scipy.optimize import minimize, Bounds
from functools import lru_cache
import time


class SlsqpProblem(ProblemInterface):
    def __init__(self, x0, objective_func, constraints, integrator, p=None, tvp=None, u_param=None):
        self.x0 = x0
        self.objective_func = objective_func 
        self.constraints_list = constraints
        self.integrator = integrator
        self.x_dim, self.u_dim, self.p_dim, self.tvp_dim = self.integrator.model.x_dim, self.integrator.model.u_dim, self.integrator.model.p_dim, self.integrator.model.tvp_dim  
        self.H = self.integrator.H
        self.p = p
        self.tvp = tvp
        self.debug_mode = False
        self.debug_x, self.debug_u = list(), list()
        self.u_param = u_param
        self._custom_u_size = self.u_param.get_nb_var(self.H)

    def _split(self, x):
        prev_idx = 0
        states = x[prev_idx:prev_idx+self.x_dim*self.H]
        prev_idx += self.x_dim*self.H

        if self.u_param is None:
            u = x[prev_idx:prev_idx+self.u_dim*self.H]
            prev_idx += self.u_dim*self.H
            u.reshape(self.H, self.u_dim)
        else:
            u = x[prev_idx:prev_idx+self._custom_u_size]
            prev_idx += self._custom_u_size
            u = u.reshape(-1)

        return states.reshape(self.H, self.x_dim), u, self.tvp, self.p

    def objective(self, x):
        states, u, tvp, p = self._split(x)
        if self.debug_mode:
            self.debug_x.append(states.copy())
            self.debug_u.append(u.copy())

        if not self.u_param is None:
            u_param = u
            u = self.u_param.forward(u)    
        res =  self.objective_func.forward(states, u, p=p, tvp=tvp)

        return res

    def gradient(self, x):
        states, u, tvp, p = self._split(x)

        if not self.u_param is None:
            u_param = u
            u = self.u_param.forward(u_param)  

        res =  self.objective_func.gradient(states, u, p=p, tvp=tvp)

        if not self.u_param is None:
            jaco_mat = self.u_param.jacobian(u_param)
            projeted_u = res[self.x_dim*self.H:]

            grad_u_param = np.dot(projeted_u.reshape(1,-1),jaco_mat).reshape(-1)
            res = np.concatenate([res[:self.x_dim*self.H], grad_u_param])

        return res

    def set_debug(self, debug_mode):
        self.debug_mode = debug_mode

    def constraints(self, x, eq=True):
        # TODO not taking into account other constraint than integrator ones      
        warnings.warn("Not taking into account other constraint than integrator ones  !")  
        states, u, tvp, p = self._split(x)

        if not self.u_param is None:
            u_param = u
            u = self.u_param.forward(u)  

        contraints_forward_list = [self.integrator.forward(states, u, self.x0, p=p, tvp=tvp),] if eq else []

        for ctr in self.constraints_list:
            forward_res = []
            if eq and (ctr.get_type() == Constraint.EQ_TYPE):
                forward_res.append(ctr.forward(states, u if (not ctr.use_u_param()) else u_param , p=p, tvp=tvp))
            elif not eq and (ctr.get_type() == Constraint.INEQ_TYPE):
                forward_res.append(ctr.forward(states, u if (not ctr.use_u_param()) else u_param , p=p, tvp=tvp))
            elif not eq and (ctr.get_type() == Constraint.INTER_TYPE):
                forward_res.append(ctr.forward(states, u if (not ctr.use_u_param()) else u_param, p=p, tvp=tvp)-ctr.get_lower_bound())
                forward_res.append(-ctr.forward(states, u if (not ctr.use_u_param()) else u_param, p=p, tvp=tvp)+ctr.get_upper_bound())
            else:
                pass

            contraints_forward_list.extend(forward_res)


        return np.concatenate(contraints_forward_list, axis=0)

    
    def hessianstructure(self):
        raise NotImplementedError("Not needed")

    
    def hessian(self, x, lagrange, obj_factor):
        raise NotImplementedError("Not needed")

    def jacobian(self, x, eq=True):
        # TODO not taking into account other constraint than integrator ones      
        warnings.warn("Not taking into account other constraint than integrator ones  !")
        states, u, tvp, p = self._split(x)

        if not self.u_param is None:
            u_param = u
            u_jaco_mat = self.u_param.jacobian(u_param)
            u = self.u_param.forward(u_param)  

        contraints_jacobian_list = []

        if eq:
            inte_jaco = self.integrator.jacobian(states, u, self.x0, p=p, tvp=tvp)
            if not self.u_param is None:
                inte_jaco_x, inte_jaco_u = inte_jaco[:,:self.x_dim*self.H], inte_jaco[:,self.x_dim*self.H:]
                inte_jaco_u = np.dot(inte_jaco_u, u_jaco_mat)
                inte_jaco = np.concatenate([inte_jaco_x, inte_jaco_u],axis=1)

            contraints_jacobian_list.append(inte_jaco)

        for ctr in self.constraints_list:
            if (eq and (ctr.get_type() == Constraint.EQ_TYPE)) or (not eq and (ctr.get_type() == Constraint.INEQ_TYPE)):

                ctr_jaco = ctr.jacobian(states, u if (not ctr.use_u_param()) else u_param, p=p, tvp=tvp) 

                
                if not ctr.use_u_param() and (not self.u_param is None):
                    ctr_jaco_x, ctr_jaco_u = ctr_jaco[:,:self.x_dim*self.H], ctr_jaco[:,self.x_dim*self.H:]
                    ctr_jaco_u = np.dot(ctr_jaco_u, u_jaco_mat)

                    ctr_jaco = np.concatenate([ctr_jaco_x, ctr_jaco_u],axis=1)
                
                contraints_jacobian_list.append(ctr_jaco)

            elif not eq and (ctr.get_type() == Constraint.INTER_TYPE):

                ctr_jaco = ctr.jacobian(states, u if (not ctr.use_u_param()) else u_param, p=p, tvp=tvp) 

                if not ctr.use_u_param() and (not self.u_param is None):
                    ctr_jaco_x, ctr_jaco_u = ctr_jaco[:self.x_dim*self.H], ctr_jaco[self.x_dim*self.H:]
                    ctr_jaco_u = np.dot(ctr_jaco_u, u_jaco_mat)

                    ctr_jaco = np.concatenate([ctr_jaco_x, ctr_jaco_u],axis=1)
                contraints_jacobian_list.append(ctr_jaco)
                contraints_jacobian_list.append(-ctr_jaco)

            else:
                pass

        full_jacobian = np.concatenate(contraints_jacobian_list, axis=0)
        return full_jacobian

    def get_constraints_dict(self):
        # TODO not taking into account other constraint than integrator ones        
        return [{'type': 'eq',
           'fun' : lambda x: self.constraints(x, eq=True),
           'jac' : lambda x: self.jacobian(x, eq=True)},
           {'type': 'ineq',
           'fun' : lambda x: self.constraints(x, eq=False),
           'jac' : lambda x: self.jacobian(x, eq=False)}]



    def get_init_value(self):
        return self.x0




class SlsqpProblemFactory(ProblemFactory):
    def _process(self):
        return SlsqpProblem(self.x0, self.objective, self.constraints, self.integrator, p=self.p, tvp=self.tvp, u_param=self.u_param)


class Slsqp(Optimizer):
    def __init__(self, max_iteration=200, tolerance=0.5e-6, verbose=1, init_with_last_result=False, nb_max_try=15, debug=False):

        super(Slsqp, self).__init__()

        self.max_iteration=max_iteration
        self.verbose = verbose
        self.tolerance=tolerance
        self.init_with_last_result = init_with_last_result

        self.prev_result = None
        self.nb_max_try = nb_max_try
        self.debug = debug

    def get_factory(self):
        return SlsqpProblemFactory()


    def solve(self, problem, domain_constraint, x_init=None):
        x0 = problem.get_init_value()
        problem.set_debug(self.debug)
        if self.init_with_last_result and not (self.prev_result is None) and (problem.u_param is None):
            x_dim = problem.integrator.model.x_dim
            u_dim = problem.integrator.model.u_dim
            x_init = np.concatenate([self.prev_result[x_dim:x_dim*problem.integrator.H], # x[1]-x[H]
              self.prev_result[x_dim*(problem.integrator.H-1):x_dim*problem.integrator.H], # x[H]
              self.prev_result[x_dim*problem.integrator.H+u_dim:(x_dim+u_dim)*problem.integrator.H],  # u[1] - u[H]
              self.prev_result[x_dim*problem.integrator.H+u_dim*(problem.integrator.H-1):x_dim*problem.integrator.H+u_dim*problem.integrator.H] ], axis=0) # u[H]
        elif not (x_init is None):
            x_init = x_init
        else:
            if (problem.u_param is None):
                x_init = np.concatenate( [np.concatenate( [x0,]*problem.integrator.H ), np.repeat(np.array([0.0,]*problem.integrator.model.u_dim),problem.integrator.H)])
            else:
                u_total_dim = problem.u_param.get_nb_var(problem.integrator.H) 
                x_init = np.concatenate( [np.concatenate( [x0,]*problem.integrator.H ), np.array([0.0,]*u_total_dim)])

        # TODO find a better way to get the horizon variable
        lb = domain_constraint.get_lower_bounds(problem.integrator.H)
        ub = domain_constraint.get_upper_bounds(problem.integrator.H)
        bounds = Bounds(lb, ub)

        
        res = minimize(problem.objective, x_init, method="SLSQP", jac=problem.gradient, \
            constraints=problem.get_constraints_dict(), options={'maxiter': self.max_iteration, 'ftol': self.tolerance, 'disp': True, 'iprint':self.verbose}, bounds=bounds)
        if self.debug:
            self.constraints_val = problem.constraints(res.x)
            self.debug_x = problem.debug_x
            self.debug_u = problem.debug_u
        res.success = True
        if not res.success :
            warnings.warn("Process do not converge ! ")

            if self.debug:
                return Optimizer.FAIL

            if np.max(problem.constraints(res.x))>1e-5:
                for i in range(self.nb_max_try):
                    print("RETRY SQP optimization")
                    x_init = np.concatenate( [np.concatenate( [x0,]*problem.integrator.H ), np.repeat(np.array([0.0,]*problem.integrator.model.u_dim),problem.integrator.H)])
                    res = minimize(problem.objective, x_init, method="SLSQP", jac=problem.gradient, \
                        constraints=problem.get_constraints_dict(), options={'maxiter': self.max_iteration, 'ftol': self.tolerance*(2.0**i), 'disp': True, 'iprint':self.verbose}, bounds=bounds)
                    if np.max(problem.constraints(res.x))<1e-5  or res.success:
                        break
                    
            if not res.success and (np.max(problem.constraints(res.x))>1e-5):
                return Optimizer.FAIL

        self.prev_result = res.x

        return Optimizer.SUCCESS