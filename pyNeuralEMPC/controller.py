import numpy as np 
from .optimizer import Ipopt, Optimizer
from .constraints import DomainConstraint



class NMPC():
    def __init__(self, integrator, objective_func, constraint_list, H, DT, optimizer=Ipopt(), use_hessian=True):

        self.integrator = integrator
        self.objective_func = objective_func
        self.constraint_list = constraint_list
        self.domain_constraint = list(filter(lambda x: isinstance(x, DomainConstraint), constraint_list))[0]
        self.constraint_list.remove(self.domain_constraint)
        self.H = H
        self.DT = DT 
        self.optimizer = optimizer
        self.use_hessian = use_hessian


        # TODO VERIF EVERY THINGS !

    def get_pb(self, x0: np.array, p=None, tvp=None, init_x=None, init_u=None):
        
        # first check x0, p and tvp dim and look if the model need it !
        assert len(x0.shape) == 1, "x0 must be a vector"
        assert x0.shape[0] == self.integrator.model.x_dim, "x0 dim must set according to your model !"
        
        if not p is None:
            assert len(x0.shape) == 1, "p must be a vector"
            assert p.shape[0] == self.integrator.model.p_dim, "p dim must set according to your model !"

        if not tvp is None:
            assert len(tvp.shape) == 2, "tvp must be a vector"
            assert tvp.shape[1] == self.integrator.model.tvp_dim, "tvp dim must set according to your model !"
            assert tvp.shape[0] == self.H, "tvp first dim must set according to the horizon size !"

        assert init_x.shape[1] == self.integrator.model.x_dim, f"init_x dim must have the good feature size (expected {self.integrator.model.x_dim})"
        assert init_u.shape[1] == self.integrator.model.u_dim, f"init_u dim mist have the good feature size (expected {self.integrator.model.u_dim})"
        assert (init_x is None) == (init_u is None), f"you must give both init values"

        pb_facto = self.optimizer.get_factory()

        pb_facto.set_x0(x0)

        pb_facto.set_objective(self.objective_func)

        pb_facto.set_integrator(self.integrator)

        pb_facto.set_constraints(self.constraint_list)
        
        if not(init_x is None):
            pb_facto.set_init_values(init_x, init_u)

        if not tvp is None:
            pb_facto.set_tvp(tvp)

        if not p is None:
            pb_facto.set_p(p)   

        pb_obj = pb_facto.getProblemInterface()

        return pb_obj

    def next(self, x0: np.array, p=None, tvp=None, init_x=None, init_u=None):
        
        # first check x0, p and tvp dim and look if the model need it !
        assert len(x0.shape) == 1, "x0 must be a vector"
        assert x0.shape[0] == self.integrator.model.x_dim, "x0 dim must set according to your model !"
        
        if not p is None:
            assert len(x0.shape) == 1, "p must be a vector"
            assert p.shape[0] == self.integrator.model.p_dim, "p dim must set according to your model !"

        if not tvp is None:
            assert len(tvp.shape) == 2, "tvp must be a vector"
            assert tvp.shape[1] == self.integrator.model.tvp_dim, "tvp dim must set according to your model !"
            assert tvp.shape[0] == self.H, "tvp first dim must set according to the horizon size !"
    
        assert (init_x is None) == (init_u is None), f"you must give both init values"

        if not init_x is None:
            assert init_x.shape[1] == self.integrator.model.x_dim, f"init_x dim must have the good feature size (expected {self.integrator.model.x_dim})"
            assert init_u.shape[1] == self.integrator.model.u_dim, f"init_u dim mist have the good feature size (expected {self.integrator.model.u_dim})"

        pb_facto = self.optimizer.get_factory()

        pb_facto.set_x0(x0)

        pb_facto.set_objective(self.objective_func)

        pb_facto.set_integrator(self.integrator)

        pb_facto.set_constraints(self.constraint_list)

        if not(init_x is None):
            pb_facto.set_init_values(init_x, init_u)

        if not tvp is None:
            pb_facto.set_tvp(tvp)

        if not p is None:
            pb_facto.set_p(p)  
            
        pb_obj = pb_facto.getProblemInterface()

        res =  self.optimizer.solve(pb_obj, self.domain_constraint)

        if res == Optimizer.SUCCESS:
            return self.optimizer.prev_result[0: self.integrator.model.x_dim*self.integrator.H].reshape(self.integrator.H, -1), self.optimizer.prev_result[self.integrator.model.x_dim*self.integrator.H: ].reshape(self.integrator.H, -1)

        else:
            return None, None
