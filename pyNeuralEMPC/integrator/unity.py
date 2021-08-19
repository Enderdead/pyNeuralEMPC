from .base import Integrator
import numpy as np 






class UnityIntegrator(Integrator):

    def __init__(self, model, H):
        nb_contraints = model.x_dim*H
        super(UnityIntegrator, self).__init__(model, H, nb_contraints)

    def forward(self, x: np.ndarray, u: np.ndarray, x0: np.ndarray, p=None, tvp=None)-> np.ndarray:

        assert len(x.shape) == 2 and len(u.shape) == 2, "x and u tensor must have dim 2"

        assert len(x0.shape) == 1, "x0 shape must have dim 1"

        # TODO maybe look at the shape values

        # Create a view for t-1 and t in order to perform the substraction
        x_t_1 = np.concatenate([x0.reshape(1,-1),x],axis=0)[:-1]
        x_t     = x.copy()


        # Get discret differential prediction from the model
        estim_x_t = self.model.forward(x_t_1, u, tvp=tvp, p=p)

        # return flatten constraint error
        return (estim_x_t - x_t).reshape(-1)

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

        x_t_1 =  np.concatenate([x0.reshape(1,-1),x],axis=0)[:-1]
        model_jac = self.model.jacobian(x_t_1, u, p=p, tvp=tvp)  

        print(model_jac.shape)
        10/0
        # state t - 1 
        J_extended[state_dim:,0:state_dim*(self.H-1)] += model_jac[state_dim:,state_dim:state_dim*self.H]#make_diag_from_2D(model_jac[state_dim:, 0:state_dim])

        # U
        J_extended[:,state_dim*self.H:(state_dim+control_dim)*self.H] +=  model_jac[:,state_dim*self.H:(state_dim+control_dim)*self.H]#make_diag_from_2D(model_jac[:, state_dim:state_dim+control_dim])

        return J_extended


    def hessian(self, x, u, x0, p=None, tvp=None) -> np.ndarray:
        x_t_1 =  np.concatenate([x0.reshape(1,-1),x],axis=0)[:-1]

        model_H = self.model.hessian(x_t_1, u, p=p, tvp=tvp)


        model_H = model_H.reshape(-1,*model_H.shape[2:])
        final_H = np.zeros_like(model_H)
        # x_t x x_t 
        final_H[:,:x.shape[1]*(x.shape[0]-1), 0:x.shape[1]*(x.shape[0]-1)] += model_H[:,x.shape[1]:x.shape[1]*x.shape[0], x.shape[1]:x.shape[1]*x.shape[0]]
        # u_t x u_t
        final_H[:,x.shape[1]*x.shape[0]:, x.shape[1]*x.shape[0]:] += model_H[:,x.shape[1]*x.shape[0]:, x.shape[1]*x.shape[0]:]

        # x_t x u_t
        final_H[:, :x.shape[1]*(x.shape[0]-1), x.shape[1]*x.shape[0]:] += model_H[:,x.shape[1]:x.shape[1]*x.shape[0], x.shape[1]*x.shape[0]:]

        # u_t x x_t
        final_H[:,x.shape[1]*x.shape[0]:,  :x.shape[1]*(x.shape[0]-1) ] += model_H[:,  x.shape[1]*x.shape[0]:, x.shape[1]:x.shape[1]*x.shape[0] ]

        
        return final_H

