from .base import Model

import numpy as np
import jax.numpy as jnp
import jax


def gen_jac_proj_mat(T, X_DIM, WIN_SIZE):
    X_SLIDED = X_DIM * WIN_SIZE
    proj_mat = np.zeros((T * X_DIM, T, X_SLIDED))

    for i in range(T):
        for k in range(X_DIM):
            for o in range(T):
                pos = (WIN_SIZE-1)*X_DIM + k + i*X_DIM - o*X_DIM
                if o>=i and pos>=0:
                    proj_mat[i*X_DIM + k, o, pos] = 1.0
        
    proj_mat = proj_mat.reshape(T, X_DIM, T, X_SLIDED)
    return proj_mat

def _check_func(func, state_dim, control_dim, p_dim, tvp_dim, rolling_window=1):
    try: #TODO c'est le bordel
        _ = jax.jacrev(func)(jnp.zeros((2, state_dim*rolling_window)),\
             jnp.zeros((2, control_dim*rolling_window)),\
             p=None if p_dim is None else jnp.zeros(p_dim),\
             tvp=None if tvp_dim is None else jnp.zeros((2, tvp_dim*rolling_window)))
    except AssertionError:
        return False
    return True

class DiffDiscretJaxModel(Model):
    
    def __init__(self, forward_func, x_dim: int, u_dim: int, p_dim=0, tvp_dim=0, vector_mode=False, safe_mode=True):
        if safe_mode:
            if not _check_func(forward_func, x_dim, u_dim, p_dim, tvp_dim):
                raise ValueError("Your function is not differentiable w.r.t the JAX library")

        self.forward_func = forward_func
        self.vector_mode = vector_mode

        super(DiffDiscretJaxModel, self).__init__(x_dim, u_dim, p_dim, tvp_dim)


    def forward(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        if self.vector_mode:
            return self.forward_func(x, u, p=p, tvp=tvp)
        else:
            raise NotImplementedError("")


    def jacobian(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        if self.vector_mode:
            argnums_list = [0, 1]

            jacobians = list(jax.jacobian(self.forward_func, argnums=argnums_list)(x, u, p, tvp))

            jacobians[0] = jacobians[0].reshape(self.x_dim*x.shape[0], self.x_dim*x.shape[0] )
            jacobians[1] = jacobians[1].reshape(self.x_dim*x.shape[0], self.u_dim*x.shape[0] )

            jaco =  jnp.concatenate(jacobians, axis=1).to_py()
            return jaco
        else:
            raise NotImplementedError("")

    def hessian(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        if self.vector_mode:
            argnums_list = [0, 1]

            H = x.shape[0]
            X_dim = x.shape[1]
            U_dim = u.shape[1]

            hessians = list(jax.hessian(self.forward_func, argnums=argnums_list)(x, u, p, tvp))

            a = hessians[0][0].reshape(H  , X_dim,  H*X_dim, H*X_dim)
            b = hessians[0][1].reshape(H  , X_dim,  H*X_dim, H*U_dim)
            ab  = jnp.concatenate([a, b], axis=3)

            c = hessians[1][0].reshape(H  , X_dim,  H*U_dim, H*X_dim)
            d = hessians[1][1].reshape(H  , X_dim,  H*U_dim, H*U_dim)
            cd  = jnp.concatenate([c, d], axis=3)

            final_hessian = jnp.concatenate([ab, cd], axis=2)

            return final_hessian.to_py()
        else:
            raise NotImplementedError("")




class DiffDiscretJaxModelRollingWindow(Model):
    
    def __init__(self, forward_func, x_dim: int, u_dim: int, p_dim=0, tvp_dim=0, rolling_window=1, forward_rolling=True, vector_mode=True, safe_mode=True):
        
        if safe_mode:
            if not _check_func(forward_func, x_dim, u_dim, p_dim, tvp_dim, rolling_window=rolling_window):
                raise ValueError("Your function is not differentiable w.r.t the JAX library")

        self.forward_func = forward_func
        self.vector_mode = vector_mode

        self.prev_x = None
        self.prev_u = None

        self.forward_rolling = forward_rolling

        if not forward_rolling:
            raise NotImplementedError("Sorry ='(")

        self.rolling_window = rolling_window

        self.jacobian_proj = None
        self.hessian_proj  = None

        super(DiffDiscretJaxModelRollingWindow, self).__init__(x_dim, u_dim, p_dim, tvp_dim)

    def set_prev_data(self, x_prev: np.ndarray, u_prev: np.ndarray, tvp_prev=None):

        assert x_prev.shape == (self.rolling_window-1, self.x_dim), f"Your x prev tensor must have the following shape {(self.rolling_window-1, self.x_dim)} (received : {x_prev.shape})" 
        assert u_prev.shape == (self.rolling_window-1, self.u_dim), f"Your u prev tensor must have the following shape {(self.rolling_window-1, self.u_dim)} (received : {u_prev.shape})"

        self.prev_x = x_prev
        self.prev_u = u_prev

        if not tvp_prev is None:
            assert tvp_prev.shape == (self.rolling_window-1, self.tvp_dim), f"Your tvp prev tensor must have the following shape {(self.rolling_window-1, self.tvp_dim)} (received : {tvp_prev.shape})"
            self.prev_tvp  = tvp_prev  


    def _gather_input(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        
        assert (not self.prev_x is None) and (not self.prev_u is None), "You must give history window with set_prev_data before calling any inferance function."

        x_extended = np.concatenate([self.prev_x, x], axis=0)
        u_extended = np.concatenate([self.prev_u, u], axis=0)


        if not tvp is None:
            tvp_extended = np.concatenate([self.prev_tvp, tvp], axis=0)
        else:
            tvp_extended = None


        return x_extended, u_extended, p, tvp_extended 


    def _slide_input(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        x_slided = np.stack([x[i:i+self.rolling_window].reshape(-1) for i in range(x.shape[0]-self.rolling_window+1)])

        u_slided = np.stack([u[i:i+self.rolling_window].reshape(-1) for i in range(x.shape[0]-self.rolling_window+1)])


        if not tvp is None:
            tvp_slided = np.stack([tvp[i:i+self.rolling_window].reshape(-1) for i in range(x.shape[0]-self.rolling_window+1)])
        else:
            tvp_slided = None

        return x_slided, u_slided, p, tvp_slided


    def forward(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        x_extended, u_extended, p, tvp_extended = self._gather_input(x, u, p, tvp)
        x_slided, u_slided, p, tvp_slided = self._slide_input(x_extended, u_extended, p, tvp_extended)

        if self.vector_mode:
            return self.forward_func(x_slided, u_slided, p=p, tvp=tvp_slided)
        else:
            raise NotImplementedError("")


    def jacobian(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        x_extended, u_extended, p_extended, tvp_extended = self._gather_input(x, u, p, tvp)
        x_slided, u_slided, p_slided, tvp_slided = self._slide_input(x_extended, u_extended, p_extended, tvp_extended)

        if self.vector_mode:
            argnums_list = [0, 1]

            jacobians = list(jax.jacobian(self.forward_func, argnums=argnums_list)(x_slided, u_slided, p_slided, tvp_slided))

            jacobians[0] = jacobians[0].reshape(x.shape[0], self.x_dim,  x.shape[0]*x_slided.shape[1])
            jacobians[1] = jacobians[1].reshape(x.shape[0], self.x_dim, u_slided.shape[1]*x.shape[0] )

            if (self.jacobian_proj is None) or self.jacobian_proj[0].shape[0] != x.shape[0]:
                # TODO handle reverse order
                self.jacobian_proj = [ gen_jac_proj_mat(x.shape[0], self.x_dim, self.rolling_window),
                       gen_jac_proj_mat(x.shape[0], self.u_dim, self.rolling_window) ]

            jacobians_x = np.einsum("abc,cd->abd",np.array(jacobians[0]), self.jacobian_proj[0].reshape(x.shape[0]*self.x_dim, x.shape[0]*x_slided.shape[1]).T )\
                .reshape(self.x_dim*x.shape[0], self.x_dim*x.shape[0])

            jacobians_u = np.einsum("abc,cd->abd", np.array(jacobians[1]), self.jacobian_proj[1].reshape(x.shape[0]* self.u_dim, u_slided.shape[1]*x.shape[0]).T )\
                .reshape(self.x_dim*x.shape[0], self.u_dim*x.shape[0])

            jaco =  np.concatenate([jacobians_x, jacobians_u], axis=1)
            return jaco
        else:
            raise NotImplementedError("")

    def hessian(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        x_extended, u_extended, p_extended, tvp_extended = self._gather_input(x, u, p, tvp)
        x_slided, u_slided, p_slided, tvp_slided = self._slide_input(x_extended, u_extended, p_extended, tvp_extended)

        if self.vector_mode:
            argnums_list = [0, 1]

            H = x.shape[0]
            X_dim = x.shape[1]
            U_dim = u.shape[1]

            hessians = list(jax.hessian(self.forward_func, argnums=argnums_list)(x_slided, u_slided, p_slided, tvp_slided))

            if (self.hessian_proj is None) or int(self.hessian_proj[0].shape[0]/(X_dim*self.rolling_window)) != H:
                # TODO handle reverse order

                proj_mat_X = np.zeros((H*X_dim*self.rolling_window, H*X_dim+X_dim*(self.rolling_window-1)))
                proj_mat_U = np.zeros((H*U_dim*self.rolling_window, H*U_dim+U_dim*(self.rolling_window-1)))

                for t in range(H):
                    proj_mat_X[ t*self.rolling_window*X_dim:(t+1)*self.rolling_window*X_dim, t*X_dim:t*X_dim+self.rolling_window*X_dim] = np.eye(X_dim*self.rolling_window, X_dim*self.rolling_window)
                    proj_mat_U[ t*self.rolling_window*U_dim:(t+1)*self.rolling_window*U_dim, t*U_dim:t*U_dim+self.rolling_window*U_dim] = np.eye(U_dim*self.rolling_window, U_dim*self.rolling_window)

                self.hessian_proj = [proj_mat_X, proj_mat_U]


            a = hessians[0][0].reshape(H  , X_dim,  H*X_dim*self.rolling_window, H*X_dim*self.rolling_window)
            b = hessians[0][1].reshape(H  , X_dim,  H*X_dim*self.rolling_window,H*U_dim*self.rolling_window)

            a = np.einsum("abcd,de->abce", np.array(a), self.hessian_proj[0])
            a = np.einsum("ec,abcd->abed", self.hessian_proj[0].T, a)
            a = a[:,:,X_dim*(self.rolling_window-1):, X_dim*(self.rolling_window-1):]


            b = np.einsum("abcd,de->abce", np.array(b), self.hessian_proj[1])
            b = np.einsum("ec,abcd->abed", self.hessian_proj[0].T, b)
            b = b[:,:,X_dim*(self.rolling_window-1):, U_dim*(self.rolling_window-1):]

            ab  = np.concatenate([a, b], axis=3)

            c = hessians[1][0].reshape(H  , X_dim,  H*U_dim*self.rolling_window, H*X_dim*self.rolling_window)
            d = hessians[1][1].reshape(H  , X_dim,  H*U_dim*self.rolling_window, H*U_dim*self.rolling_window)

            c = np.einsum("abcd,de->abce", np.array(c), self.hessian_proj[0])
            c = np.einsum("ec,abcd->abed", self.hessian_proj[1].T, c)
            c = c[:,:,U_dim*(self.rolling_window-1):, X_dim*(self.rolling_window-1):]


            d = np.einsum("abcd,de->abce", np.array(d), self.hessian_proj[1])
            d = np.einsum("ec,abcd->abed", self.hessian_proj[1].T, d)
            d = d[:,:,U_dim*(self.rolling_window-1):, U_dim*(self.rolling_window-1):]

            cd  = np.concatenate([c, d], axis=3)

            final_hessian = np.concatenate([ab, cd], axis=2)

            return final_hessian
        else:
            raise NotImplementedError("")

