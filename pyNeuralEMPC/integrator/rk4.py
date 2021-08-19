from .base import Integrator
import numpy as np 
import time
import hashlib
def extend_dim(array, size, axis=0, value=0.0):
    target_shape = list(array.shape)
    target_shape[axis] = size

    extented_array = np.ones(target_shape, dtype=array.dtype)*value
    return np.concatenate([array, extented_array], axis=axis)


def make_diag_from_2D(A: np.ndarray):
    result = np.zeros((A.shape[0], A.shape[0]*A.shape[1]))
    for i, sub_element in enumerate(A):
        result[i:i+1, i*A.shape[1] : (i+1)*A.shape[1]] = sub_element
    return result


class TensorCache:
    def __init__(self, max_size=5):
        self.keys = list()
        self.datas = list()
        self.max_size = max_size
    
    def pull(self, *args):
        hash_key = hash("".join(map(lambda x: str(x), args)))
        if hash_key in self.keys:
            return self.datas[self.keys.index(hash_key)]
        else:
            return None

    def push(self, key, value):
        hash_key = hash("".join(map(lambda x: str(x), key)))
        if hash_key in self.keys:
            self.datas[self.keys.index(hash_key)] = value
        else:
            if len(self.keys)>=self.max_size:
                del self.keys[0]
                del self.datas[0]
            
            self.keys.append(hash_key)
            self.datas.append(value)
        

class RK4Integrator(Integrator):

    def __init__(self, model, H, DT, cache_mode=False, cache_size=2):
        self.DT = DT
        nb_contraints = model.x_dim*H
        self.cache_mode = cache_mode
        self.forward_cache = TensorCache(max_size=cache_size) if self.cache_mode else None
        self.jacobian_cache = TensorCache(max_size=cache_size) if self.cache_mode else None
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

        # Push into cache if needed
        if self.cache_mode:
            self.forward_cache.push( (x, u, x0, p, tvp),  (k_1, k_2, k_3, k_4) )

        d_x_t_1  = (k_1 + 2*k_2+ 2*k_3 + k_4)*self.DT/6.0
        # Get discret differential prediction from the model
        estim_x_t = x_t_1 + d_x_t_1

        # return flatten constraint error
        return (estim_x_t - x_t).reshape(-1)

    def _get_model_jacobian(self, x_t_1, u, p=None, tvp=None):
        jaco = self.model.jacobian(x_t_1, u, p=p, tvp=tvp)
        reshape_indexer = sum([ list(np.arange(x_t_1.shape[1])+i*x_t_1.shape[1]) + \
         list( x_t_1.shape[1]*x_t_1.shape[0]+np.arange(u.shape[1])+ i*u.shape[1])  for i in range(x_t_1.shape[0])  ], list())
        jaco = np.take(jaco, reshape_indexer, axis=1)
        jaco = jaco.reshape(x_t_1.shape[0], x_t_1.shape[1], x_t_1.shape[0], x_t_1.shape[1]+ u.shape[1])
        jaco = np.array([jaco[i, :, i, :] for i in range(x_t_1.shape[0])]) #TODO ca marhce ?
        return jaco

    def _get_model_hessian(self, x_t_1, u, p=None, tvp=None):
        state_dim  = x_t_1.shape[1]
        u_state = u.shape[1]
        hessian = self.model.hessian(x_t_1, u, p=p, tvp=tvp)

        hessian = hessian.reshape(x_t_1.shape[0], x_t_1.shape[1], x_t_1.shape[0]*(x_t_1.shape[1]+u.shape[1]), x_t_1.shape[0]*(x_t_1.shape[1]+u.shape[1]))

        reshape_indexer = sum([ list(np.arange(x_t_1.shape[1])+i*x_t_1.shape[1]) + \
         list( x_t_1.shape[1]*x_t_1.shape[0]+np.arange(u.shape[1])+ i*u.shape[1])  for i in range(x_t_1.shape[0])  ], list())

        

        hessian = np.take(hessian, reshape_indexer, axis=-1)
        hessian = np.take(hessian, reshape_indexer, axis=-2)
        hessian = np.array([hessian[i, :, (state_dim+u_state)*i:(state_dim+u_state)*(i+1), (state_dim+u_state)*i:(state_dim+u_state)*(i+1)] for i in range(x_t_1.shape[0])])

        return hessian


    def jacobian(self, x, u, x0, p=None, tvp=None) -> np.ndarray:
        start = time.time()
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

        if self.cache_mode:
            cache_data = self.forward_cache.pull(x, u, x0, p, tvp)
        else:
            cache_data = None

        if cache_data is None:
            k_1 = self.model.forward( x_t_1, u, p=p, tvp=tvp)
            k_2 = self.model.forward( x_t_1 + k_1*self.DT/2.0, u, p=p, tvp=tvp) 
            k_3 = self.model.forward( x_t_1 + k_2*self.DT/2.0, u, p=p, tvp=tvp) 
        else:
            k_1, k_2, k_3, _ = cache_data

        dk1 = self._get_model_jacobian(x_t_1, u, p=p, tvp=tvp)

        #TODO here we can't support RNN network....
        # Add a security 
        partial_dk2 = self._get_model_jacobian(x_t_1 + k_1*self.DT/2.0, u, p=p, tvp=tvp).reshape(x_t_1.shape[0], x_t_1.shape[1], x_t_1.shape[1]+ u.shape[1])
        dk1_extended = extend_dim(dk1,u.shape[1], axis=1)
        dk2 = np.einsum('ijk,ikl->ijl',partial_dk2,  np.eye(x_t_1.shape[1]+u.shape[1] ,x_t_1.shape[1]+u.shape[1] )+dk1_extended*self.DT/2.0)

        partial_dk3 = self._get_model_jacobian(x_t_1 + k_2*self.DT/2.0, u, p=p, tvp=tvp).reshape(x_t_1.shape[0], x_t_1.shape[1], x_t_1.shape[1]+ u.shape[1])
        dk2_extended = extend_dim(dk2,u.shape[1], axis=1) 
        dk3 = np.einsum('ijk,ikl->ijl',partial_dk3,  np.eye(x_t_1.shape[1]+u.shape[1] ,x_t_1.shape[1]+u.shape[1] )+dk2_extended*self.DT/2.0)

        partial_dk4 = self._get_model_jacobian(x_t_1 + k_3*self.DT, u, p=p, tvp=tvp).reshape(x_t_1.shape[0], x_t_1.shape[1], x_t_1.shape[1]+ u.shape[1])
        dk3_extended = extend_dim(dk3,u.shape[1], axis=1) 
        dk4 = np.einsum('ijk,ikl->ijl',partial_dk4,  np.eye(x_t_1.shape[1]+u.shape[1] ,x_t_1.shape[1]+u.shape[1] )+dk3_extended*self.DT)
        
        model_jac =  (self.DT/6.0) * (  dk1 + 2*dk2+ 2*dk3+ dk4)

        if self.cache_mode:
            self.jacobian_cache.push((x, u, x0, p, tvp), (dk1, partial_dk2, partial_dk3, partial_dk4))

        model_jac = model_jac.reshape(x_t_1.shape[0],x_t_1.shape[1], x_t_1.shape[1]+u.shape[1])

        #model_jac_extended = np.array([ np.concatenate([np.zeros((x_t_1.shape[1]*i, x_t_1.shape[1]+u.shape[1])) ,model_jac[i,:,:],np.zeros(((x_t_1.shape[0]-i)* x_t_1.shape[1], x_t_1.shape[1]+u.shape[1]))  ], axis=1)   for i in range(x_t_1.shape[0])], )
        # state t - 1 
        J_extended[state_dim:,0:state_dim*(self.H-1)] += np.eye(state_dim*(self.H-1),state_dim*(self.H-1) )

        for i in range(self.H):
            # state t - 1
            if i>0:
                J_extended[state_dim*i:state_dim*(i+1),state_dim*(i-1):state_dim*(i)] += model_jac[i, :, :state_dim]

            # U
            J_extended[state_dim*i:state_dim*(i+1),state_dim*self.H+control_dim*i:state_dim*self.H+control_dim*(i+1)] +=  model_jac[i, :,state_dim:]

        return J_extended


    def hessian(self, x, u, x0, p=None, tvp=None) -> np.ndarray:
        def T(array):
            return np.transpose(array, (0, 2, 1))
        
        def dot_left(a, b):
            return np.einsum('ijk,ipkl->ipjl', a, b)

        def dot_right(a, b):
            return np.einsum('ipkl,ilj->ipkj', a, b)

        def cross_sum(j, h):
            result = np.zeros_like(h)
            for i in range(j.shape[1]):
                for k in range(j.shape[1]):
                    result[:,i,:,:] += j[:,i,k].reshape(-1,1,1)*h[:,k,:,:]
            
            return result

        x_t_1 =  np.concatenate([x0.reshape(1,-1),x],axis=0)[:-1]
        # TODO add security about rnn


        if self.cache_mode:
            forward_cache_data = self.forward_cache.pull(x, u, x0, p, tvp)
        else:
            forward_cache_data = None

        if forward_cache_data is None:
            k_1 = self.model.forward( x_t_1, u, p=p, tvp=tvp)# (2,2)
            k_2 = self.model.forward( x_t_1 + k_1*self.DT/2.0, u, p=p, tvp=tvp) # (2,2)
            k_3 = self.model.forward( x_t_1 + k_2*self.DT/2.0, u, p=p, tvp=tvp) # (2,2)
        else:
            k_1, k_2, k_3, _ = forward_cache_data


        if self.cache_mode:
            jacobian_cache_data = self.jacobian_cache.pull(x, u, x0, p, tvp)
        else:
            jacobian_cache_data = None

        if jacobian_cache_data is None:
            dk1 = self._get_model_jacobian(x_t_1, u, p=p, tvp=tvp)# (2,3,3)
            partial_dk2 = self._get_model_jacobian(x_t_1 + k_1*self.DT/2.0, u, p=p, tvp=tvp)# (2,3,3)
            partial_dk3 = self._get_model_jacobian(x_t_1 + k_2*self.DT/2.0, u, p=p, tvp=tvp) # (2,3,3)
            partial_dk4 = self._get_model_jacobian(x_t_1 + k_3*self.DT, u, p=p, tvp=tvp) # (2,3,3)
        else:
            dk1, partial_dk2, partial_dk3, partial_dk4 = jacobian_cache_data


 




        #dk1 = self._get_model_jacobian(x_t_1, u, p=p, tvp=tvp) # (2,2,3)

        dk1_extended = extend_dim(dk1,u.shape[1], axis=1) # (2,3,3)
        h_k1 = self._get_model_hessian(x_t_1, u, p=p, tvp=tvp)# (2, 2, 3, 3)
        

        # K2
        #partial_dk2 = self._get_model_jacobian(x_t_1 + k_1*self.DT/2.0, u, p=p, tvp=tvp).reshape(x_t_1.shape[0], x_t_1.shape[1], x_t_1.shape[1]+ u.shape[1])
        dk2 = np.einsum('ijk,ikl->ijl',partial_dk2,  np.eye(x_t_1.shape[1]+u.shape[1] ,x_t_1.shape[1]+u.shape[1] )+dk1_extended*self.DT/2.0)
        dk2_extended = extend_dim(dk2,u.shape[1], axis=1) # (2,3,3)
        #J_local_k2 = self._get_model_jacobian(x_t_1 + k_1*self.DT/2.0, u, p=p, tvp=tvp)# (2,2,3)
        relative_j_k1 = np.eye(3,3) + ((self.DT/2.0) * dk1_extended)# (2,3,3)
        relative_h_k2 = self._get_model_hessian(x_t_1 + k_1*self.DT/2.0, u, p=p, tvp=tvp)# (2, 2, 3, 3)
        h_k2 = dot_right( dot_left(T(relative_j_k1), relative_h_k2),  relative_j_k1) + (self.DT/2.0)*cross_sum(partial_dk2, h_k1)# (2, 2, 3, 3)

        # K3
        #partial_dk3 = self._get_model_jacobian(x_t_1 + k_2*self.DT/2.0, u, p=p, tvp=tvp).reshape(x_t_1.shape[0], x_t_1.shape[1], x_t_1.shape[1]+ u.shape[1])
        dk3 = np.einsum('ijk,ikl->ijl',partial_dk3,  np.eye(x_t_1.shape[1]+u.shape[1] ,x_t_1.shape[1]+u.shape[1] )+dk2_extended*self.DT/2.0)
        dk3_extended = extend_dim(dk3,u.shape[1], axis=1) # (2,3,3)
        #J_local_k3 = self._get_model_jacobian(x_t_1 + k_2*self.DT/2.0, u, p=p, tvp=tvp)# (2,2,3)
        relative_j_k2 = np.eye(3,3) + ((self.DT/2.0) * dk2_extended)# (2,3,3)
        relative_h_k3 = self._get_model_hessian(x_t_1+ k_2*self.DT/2.0, u, p=p, tvp=tvp)# (2, 2, 3, 3)
        h_k3 = dot_right( dot_left(T(relative_j_k2), relative_h_k3),  relative_j_k2) + (self.DT/2.0)*cross_sum(partial_dk3, h_k2)# (2, 2, 3, 3)
        
        # K4
        #partial_dk4 = self._get_model_jacobian(x_t_1 + k_3*self.DT, u, p=p, tvp=tvp)# (2,2,3)
        relative_j_k3 = np.eye(3,3) + ((self.DT/1.0) * dk3_extended)# (2,3,3)
        relative_h_k4 = self._get_model_hessian(x_t_1 + k_3*self.DT, u, p=p, tvp=tvp)# (2, 2, 3, 3)
        h_k4 = dot_right( dot_left(T(relative_j_k3), relative_h_k4),  relative_j_k3) + (self.DT/1.0)*cross_sum(partial_dk4, h_k3)# (2, 2, 3, 3)
        

        model_H =(h_k1 + 2*h_k2 + 2*h_k3 + h_k4)*(self.DT/6.0) #(2,2,3,3)
        final_H = np.zeros((self.H, x_t_1.shape[1], (x_t_1.shape[1]+u.shape[1])*self.H, (x_t_1.shape[1]+u.shape[1])*self.H))
        offset = x.shape[1]*x.shape[0]

        # x_t x x_t 
        for i in range(1, self.H):
            final_H[i,:,x.shape[1]*(i-1):x.shape[1]*i, x.shape[1]*(i-1):x.shape[1]*i] += model_H[i, :, :x.shape[1], :x.shape[1]]

        # u_t x u_t
        for i in range( self.H):
            final_H[i, :, offset+i*u.shape[1] :  offset+(i+1)*u.shape[1], offset+i*u.shape[1] :  offset+(i+1)*u.shape[1]] += model_H[i, :, x.shape[1]:, x.shape[1]:]
        # x_t x u_t       
        for i in range(1, self.H):
            final_H[i, :, x.shape[1]*(i-1):x.shape[1]*i, offset+i*u.shape[1] :  offset+(i+1)*u.shape[1]] += model_H[i, :, :x.shape[1], x.shape[1]:]
        
        # u_t x x_t
        for i in range(1, self.H):
            final_H[i, :, offset+i*u.shape[1] :  offset+(i+1)*u.shape[1], x.shape[1]*(i-1):x.shape[1]*i] += model_H[i, :, x.shape[1]:, :x.shape[1]]

        return final_H.reshape(-1,*final_H.shape[2:])


