from .base import Integrator
import numpy as np 



def make_diag_from_2D(A: np.ndarray):
    result = np.zeros((A.shape[0], A.shape[0]*A.shape[1]))
    for i, sub_element in enumerate(A):
        result[i:i+1, i*A.shape[1] : (i+1)*A.shape[1]] = sub_element
    return result


class DiscretIntegrator(Integrator):
    def __init__(self, model, H):
        super(DiscretIntegrator, self).__init__(model, H)

    def forward(self, x: np.ndarray, u: np.ndarray, x0: np.ndarray, p=None, tvp=None)-> np.ndarray:

        assert len(x.shape) == 2 and len(u.shape) == 2, "x and u tensor must have dim 2"

        assert len(x0.shape) == 1, "x0 shape must have dim 1"

        # TODO maybe look at the shape values

        # Create a view for t-1 and t in order to perform the substraction
        states_t_1 = np.concatenate([x0.reshape(1,-1),x])[:-1]
        states_t     = x.copy()

        states_t_1_extended = self._format_input(x, u, x0, p=p, tvp=tvp)

        # Get discret differential prediction from the model
        estim_state_t = states_t_1 + self.model.forward(states_t_1_extended)

        # return flatten constraint error
        return (estim_state_t - states_t).reshape(-1)

    def _format_input(self, x, u, x0, p=None, tvp=None):
        # Create the input tensorflow for model forcasting

        list_input_values_t_1 = [np.concatenate([x0.reshape(1,-1),x])[:-1], u] 

        # Include tvp if present
        if not tvp is None:

            assert len(tvp.shape) == 2, "tvp must have dim 2"
            assert tvp.shape[1] == x.shape[1], "tvp must get the 2nd dim as x"

            list_input_values_t_1.append(tvp)

        # Include p if present
        if not p is None:

            assert len(p.shape) == 1, "p must be a vector"

            list_input_values_t_1.append(np.ones((x.shape[0], len(p)))*p)

        states_t_1_extended = np.concatenate(list_input_values_t_1, axis=1)

        return states_t_1_extended

    def jacobian(self, x, u, x0, p=None, tvp=None) -> np.ndarray:
        state_dim   = self.model.x_dim
        control_dim = self.model.u_dim
        tvp_dim     = self.model.tvp_dim
        p_dim       = self.model.p_dim

        J_extended = np.zeros((state_dim*self.H, (state_dim+control_dim)*self.H)) 

        # Push -1 gradient estimate of identity
        J_extended[0:self.H*state_dim, 0:self.H*state_dim] = -np.eye(state_dim*self.H,state_dim*self.H)

        # Now we need to estimate jacobian of the model forecasting

        # generate model input and evaluate the jacobian matrix
        model_input = self._format_input(x, u, x0, p=p, tvp=tvp)
        model_jac = self.model.jacobian(model_input)

        # REFAIRE 
        #assert len(model_jac.shape) == 2, "Jacobien matrix need to be a matrix (flatten 3D matrix if needed)"
        #assert model_jac.shape[0] == self.H*state_dim # TODO wron if RNN
        #assert model_jac.shape[1] == (state_dim + control_dim + tvp_dim + p_dim)

        print(model_jac)
        # state t - 1 
        J_extended[state_dim:,0:state_dim*self.H] += model_jac[:-state_dim,0:state_dim*self.H]#make_diag_from_2D(model_jac[state_dim:, 0:state_dim])

        # U
        J_extended[:,state_dim:(state_dim+control_dim)*self.H] =  model_jac[:,state_dim:(state_dim+control_dim)*self.H]#make_diag_from_2D(model_jac[:, state_dim:state_dim+control_dim])

        """
        # tvp
        if not tvp is None:
            J_extended[:,state_dim:state_dim+control_dim] = make_diag_from_2D(model_jac[:, state_dim:state_dim+control_dim])
 
        # p
        if not p is None:
            J_extended[:,-p_dim:] = model_jac[:,-p_dim:]
        """
        return J_extended


    def hessian(self, u, x0, lagrange, p=None, tvp=None) -> np.ndarray:
        raise NotImplementedError("Hessian not supported yet !")

