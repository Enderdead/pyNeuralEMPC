import numpy as np
from ..model.base import Model


class Integrator():
    def __init__(self, model, H: int, nb_contraints: int):
        """Base integrator constructor.

        Args:
            model : The system model
            H (int) : The horizon size.
            x0 (np.ndarray):  The initial state vector.
            cp (np.ndarray): The system parameters.
            tvp (np.ndarray): Time varying parameters stored in a matrix.
        """
        if not isinstance(model, (Model,)):
            raise ValueError("The model provided isn't a Model object !")

        self.H = H
        self.model = model
        self.nb_contraints = nb_contraints

        self.hessian_structure_cache = self._compute_hessianstructure()

    def get_dim(self) -> float:
        """Return the number of constraint generated.

        This method will return the number of constraint generated with this integration method.

        Returns:
            int: The number of constraint.

        """
        raise NotImplementedError("")


    def get_bound(self) -> list:
        """Return the lower and upper bound for the auto generated constraint.

        Returns:
            int: The number of constraint.

        """
        raise NotImplementedError("")

    def forward(self, u, x0, p=None, tvp=None)-> np.ndarray:
        """Generate the constraint error.

        Args:
            u (np.ndarray): The current control matrix.

        Returns:
            np.ndarray: The constraint vector.

        """
        raise NotImplementedError("")
    
    def jacobian(self, u, x0, p=None, tvp=None) -> np.ndarray:
        """Generate the jacobian matrix.

        Args:
            u (np.ndarray): The current control matrix.

        Returns:
            np.ndarray: The jacobian matrix.

        """
        raise NotImplementedError("")

    def hessian(self, u, x0, p=None, tvp=None) -> np.ndarray:
        """Generate the hessian matrix.

        Args:
            u (np.ndarray): The current control matrix.

        Returns:
            np.ndarray: The hessian matrix.

        """
        raise NotImplementedError("")
    

    def hessianstructure(self):
        return self.hessian_structure_cache


    def _compute_hessianstructure(self):
        # Try with brut force to identify non nul hessian coord
        # TODO add p and tvp 
        
        hessian_map = None

        for _ in range(3): # TODO maybe use an arg
            x_random = np.random.uniform(size=(self.H, self.model.x_dim))
            u_random = np.random.uniform(size=(self.H, self.model.u_dim))
            p_random = None
            tvp_random = None

            if self.model.p_dim > 0:
                p_random = np.random.uniform(size=self.model.p_dim)
            
            if self.model.tvp_dim > 0:
                tvp_random = np.random.uniform(size=(self.H, self.model.tvp_dim))
            
            final_hessian  = self.hessian(x_random, u_random, x_random[0:1], p=p_random, tvp=tvp_random)

            if hessian_map is None:
                hessian_map = (final_hessian!= 0.0).astype(np.float64)
            else:
                hessian_map += (final_hessian!= 0.0).astype(np.float64)
                hessian_map = hessian_map.astype(np.bool).astype(np.float64)
        hessian_map  = np.sum(hessian_map, axis=0).astype(np.bool).astype(np.float64)
        return hessian_map



    def get_lower_bounds(self):
        return [0.0,]*self.nb_contraints

    def get_upper_bounds(self):
        return [0.0,]*self.nb_contraints
        

class NoIntegrator(Integrator):
    # TODO implement a No integrator when user do not way to use conventional way
    pass

