from .base import Integrator
import numpy as np 


def extend_dim(array, size, axis=0, value=0.0):
    target_shape = list(array.shape)
    target_shape[axis] = size

    extented_array = np.ones(target_shape, dtype=array.dtype)*value
    return np.concatenate([array, extented_array], axis=axis)



class RK4Integrator(Integrator):

    def __init__(self, model, H, DT):
        self.DT = DT
        nb_contraints = model.x_dim*H
        super(RK4Integrator, self).__init__(model, H, nb_contraints)

    def forward(self, x: np.ndarray, u: np.ndarray, x0: np.ndarray, p=None, tvp=None)-> np.ndarray:

        assert len(x.shape) == 2 and len(u.shape) == 2, "x and u tensor must have dim 2"

        assert len(x0.shape) == 1, "x0 shape must have dim 1"

        # TODO maybe look at the shape values

        # Create a view for t-1 and t in order to perform the substraction
        x_t_1 = np.concatenate([x0.reshape(1,-1),x],axis=0)[:-1]
        x_t     = x.copy()

        k_1 = self.model.forward( x_t_1, u, p=p, tvp=tvp)
        k_2 = self.model.forward( x_t_1 + k_1*self.DT/2.0, u, p=p, tvp=tvp) 
        k_3 = self.model.forward( x_t_1 + k_2*self.DT/2.0, u, p=p, tvp=tvp) 
        k_4 = self.model.forward( x_t_1 + k_3*self.DT, u, p=p, tvp=tvp)

        d_x_t_1  = (k_1 + 2*k_2+ 2*k_3 + k_4)*self.DT/6.0
        # Get discret differential prediction from the model
        estim_x_t = x_t_1 + d_x_t_1

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

        k_1 = self.model.forward( x_t_1, u, p=p, tvp=tvp)
        dk1 = self.model.jacobian(x_t_1, u, p=p, tvp=tvp)
        dk1 = dk1.reshape(x_t_1.shape[0], x_t_1.shape[1], x_t_1.shape[1]+ u.shape[1])
        #TODO here we can't support RNN network....
        # Add a security 
        partial_dk2 = self.model.jacobian(x_t_1 + k_1*self.DT/2.0, u, p=p, tvp=tvp).reshape(x_t_1.shape[0], x_t_1.shape[1], x_t_1.shape[1]+ u.shape[1])
        dk1_extended = extend_dim(dk1,u.shape[1], axis=1)
        dk2 = np.einsum('ijk,ikl->ijl',partial_dk2,  np.eye(x_t_1.shape[1]+u.shape[1] ,x_t_1.shape[1]+u.shape[1] )+dk1_extended*self.DT/2.0)
        k_2 = self.model.forward( x_t_1 + k_1*self.DT/2.0, u, p=p, tvp=tvp) 


        partial_dk3 = self.model.jacobian(x_t_1 + k_2*self.DT/2.0, u, p=p, tvp=tvp).reshape(x_t_1.shape[0], x_t_1.shape[1], x_t_1.shape[1]+ u.shape[1])
        dk2_extended = extend_dim(dk2,u.shape[1], axis=1) 
        dk3 = np.einsum('ijk,ikl->ijl',partial_dk3,  np.eye(x_t_1.shape[1]+u.shape[1] ,x_t_1.shape[1]+u.shape[1] )+dk2_extended*self.DT/2.0)
        k_3 = self.model.forward( x_t_1 + k_2*self.DT/2.0, u, p=p, tvp=tvp) 


        partial_dk4 = self.model.jacobian(x_t_1 + k_3*self.DT, u, p=p, tvp=tvp).reshape(x_t_1.shape[0], x_t_1.shape[1], x_t_1.shape[1]+ u.shape[1])
        dk3_extended = extend_dim(dk3,u.shape[1], axis=1) 
        dk4 = np.einsum('ijk,ikl->ijl',partial_dk4,  np.eye(x_t_1.shape[1]+u.shape[1] ,x_t_1.shape[1]+u.shape[1] )+dk3_extended*self.DT)
        
        model_jac =  (self.DT/6.0) * (  dk1 +2.0 *dk2+ 2*dk3+ dk4)
        model_jac = model_jac.reshape(x_t_1.shape[0]*x_t_1.shape[1], x_t_1.shape[1]+u.shape[1])
        # state t - 1 
        J_extended[state_dim:,0:state_dim*(self.H-1)] += np.eye(state_dim*(self.H-1),state_dim*(self.H-1) )
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

    def hessianstructure(self, nb_sample=3):
        # Try with brut force to identify non nul hessian coord
        # TODO add p and tvp 

        hessian_map = None

        for _ in range(nb_sample):
            x_random = np.random.uniform(size=(self.H, self.model.x_dim))
            u_random = np.random.uniform(size=(self.H, self.model.u_dim))
            p_random = None
            tvp_random = None

            if self.model.p_dim > 0:
                p_random = np.random.uniform(size=self.model.p_dim)
            
            if self.model.tvp_dim > 0:
                tvp_random = np.random.uniform(size=(self.H, self.model.tvp_dim))
            
            hessian  = self.model.hessian(x_random, u_random, p=p_random, tvp=tvp_random)


            hessian = hessian.reshape(-1,*hessian.shape[2:])
            final_hessian = np.zeros_like(hessian)
            # x_t x x_t 
            final_hessian[:,:x_random.shape[1]*(x_random.shape[0]-1), 0:x_random.shape[1]*(x_random.shape[0]-1)] += hessian[:,x_random.shape[1]:x_random.shape[1]*x_random.shape[0], x_random.shape[1]:x_random.shape[1]*x_random.shape[0]]
            # u_t x u_t
            final_hessian[:,x_random.shape[1]*x_random.shape[0]:, x_random.shape[1]*x_random.shape[0]:] += hessian[:,x_random.shape[1]*x_random.shape[0]:, x_random.shape[1]*x_random.shape[0]:]

            # x_t x u_t
            final_hessian[:, :x_random.shape[1]*(x_random.shape[0]-1), x_random.shape[1]*x_random.shape[0]:] += hessian[:,x_random.shape[1]:x_random.shape[1]*x_random.shape[0], x_random.shape[1]*x_random.shape[0]:]

            # u_t x x_t
            final_hessian[:,x_random.shape[1]*x_random.shape[0]:,  :x_random.shape[1]*(x_random.shape[0]-1) ] += hessian[:,  x_random.shape[1]*x_random.shape[0]:, x_random.shape[1]:x_random.shape[1]*x_random.shape[0] ]

            if hessian_map is None:
                hessian_map = (final_hessian!= 0.0).astype(np.float64)
            else:
                hessian_map += (final_hessian!= 0.0).astype(np.float64)
                hessian_map = hessian_map.astype(np.bool).astype(np.float64)
        hessian_map  = np.sum(hessian_map, axis=0).astype(np.bool).astype(np.float64)
        return hessian_map
