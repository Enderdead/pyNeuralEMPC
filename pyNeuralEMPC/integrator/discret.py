from .base import Integrator
import numpy as np 



class DiscretIntegrator(Integrator):
    def __init__(self, model, H):
        super(DiscretIntegrator).__init__(model, H)

    def forward(self, x, u, x0, p=None, tvp=None)-> np.ndarray:
        # TODO Faire la vérif de la dim de X0 quelques part et la shape aussi
        # TODO Faire la vérif de la shape de u

        states_t_1 = np.concatenate([x0.reshape(1,-1),x])[:-1]
        states_t     = x.copy()

        gathering_states_t_1 = [states_t_1, u]
        if not p is None:
            # TODO check required
            gathering_states_t_1.append(np.ones((u.shape[0], len(p)))*p)

        if not tvp is None:
            # TODO check required
            gathering_states_t_1.append(tvp)

        states_t_1_extended = np.concatenate(gathering_states_t_1, axis=1)

        estim_state_t = states_t_1 + self.model.forward(states_t_1_extended)

        return (estim_state_t - states_t).reshape(-1)
        
    def jacobian(self, x, u, x0, p=None, tvp=None) -> np.ndarray:
        state_dim = x.shape[1]
        control_dim = u.shape[1]
        J_extended = np.zeros((state_dim*self.H, (state_dim+control_dim)*(self.H + 1)))

        # STATE t + 1
        for i in range(self.H):
            J_extended[i*state_dim:(i+1)*state_dim,(i+1)*(state_dim+control_dim):(i+1)*(state_dim+control_dim)+state_dim] = -np.eye(state_dim,state_dim)

        # STATE T & control
        res_derivative = self.model.jacobian(np.concatenate( [state_dim,control_dim], axis=1))
        for i in range(self.H):
            J_extended[i*state_dim:(i+1)*state_dim,i*(state_dim+control_dim):(i+1)*(state_dim+control_dim)] = res_derivative[i]

        return J_extended


    def hessian(self, u, x0, lagrange, p=None, tvp=None) -> np.ndarray:
        raise NotImplementedError("Hessian not supported yet !")

