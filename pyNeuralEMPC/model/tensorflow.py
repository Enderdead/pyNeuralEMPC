import tensorflow as tf 
import numpy as np  
from copy import copy
from .base import Model
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class KerasTFModel(Model):
    def __init__(self, model, x_dim: int, u_dim: int, p_dim=0, tvp_dim=0, standardScaler=None):

        if not standardScaler is None:
            raise NotImplementedError("This feature isn't supported yet !")

        #if not isinstance(model, (tf.keras.Model)):
        #    raise ValueError("The provided model isn't a Keras Model object !")

        if len(model.input_shape) != 2:
            raise NotImplementedError("Recurrent neural network are not supported atm.")

        if model.output_shape[-1] != x_dim:
            raise ValueError("Your Keras model do not provide a suitable output dim ! \n It must get the same dim as the state dim.")

        if model.input_shape[-1] != sum((x_dim, u_dim, p_dim, tvp_dim)):
            raise ValueError("Your Keras model do not provide a suitable input dim ! \n It must get the same dim as the sum of all input vars (x, u, p, tvp).")
        
        super(KerasTFModel, self).__init__(x_dim, u_dim, p_dim, tvp_dim)

        self.model = model
        self.test = None

    def __getstate__(self):
        result_dict = copy(self.__dict__)
        result_dict["model"] = None
        return result_dict

    def __setstate__(self, d):
        self.__dict__ = d

    def _gather_input(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        output_np = np.concatenate([x, u], axis=1)
        if not tvp is None:
            output_np = np.concatenate([output_np, tvp], axis=1)
        if not p is None:
            #TODO check this part
            output_np = np.concatenate([output_np, np.array([[p,]*x.shape[0]])], axis=1)

        return output_np

    def forward(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        input_net = self._gather_input(x, u, p=p, tvp=tvp)
        return self.model.predict(input_net)

    def jacobian(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        input_net = self._gather_input(x, u, p=p, tvp=tvp)

        input_tf = tf.constant(input_net)

        with tf.GradientTape(persistent=False) as tx:
            tx.watch(input_tf)
            output_tf = self.model(input_tf)
        
        jacobian_tf = tx.jacobian(output_tf, input_tf)
        jacobian_np  = jacobian_tf.numpy()[:,:,:,:-self.p_dim- self.tvp_dim]

        jacobian_np = jacobian_np.reshape(x.shape[0]*self.x_dim, (self.x_dim+self.u_dim)*x.shape[0])
        
        reshape_indexer = sum([ list(np.arange(x.shape[1])+i*(x.shape[1]+u.shape[1])) for i in range(x.shape[0])  ], list()) + \
             sum([ list( x.shape[1]+np.arange(u.shape[1])+i*(x.shape[1]+u.shape[1])) for i in range(x.shape[0])  ], list())

        jacobian_np = np.take(jacobian_np, reshape_indexer, axis=1)

        return jacobian_np
    
    @tf.function
    def _hessian_compute(self, input_tf):

        hessian_mask = tf.reshape(tf.eye(tf.shape(input_tf)[0]*self.model.output_shape[-1],tf.shape(input_tf)[0]*self.model.output_shape[-1]), (tf.shape(input_tf)[0]*self.model.output_shape[-1],tf.shape(input_tf)[0],self.model.output_shape[-1]))
        hessian_mask = tf.cast(hessian_mask, tf.float64)

        output_tf =  self.model(input_tf)
        output_tf = tf.cast(output_tf, tf.float64)
        result = tf.map_fn(lambda mask : tf.hessians(output_tf*mask, input_tf)[0] , hessian_mask, dtype=tf.float32)

        return result

    def hessian(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        input_np = self._gather_input(x, u, p=p, tvp=tvp)
        input_tf = tf.constant(input_np, dtype=tf.float32)
        #if self.test is None:
        #    self.test  = self._hessian_compute.get_concrete_function(input_tf=tf.TensorSpec([input_tf.shape[0], input_tf.shape[1]], tf.float64), output_shape=int(self.model.output_shape[-1]))
        #hessian_np = self.test(input_tf, int(self.model.output_shape[-1])).numpy()
        
        hessian_np = self._hessian_compute(input_tf).numpy()
        hessian_np = hessian_np[:,:,:-self.p_dim - self.tvp_dim, :, :-self.p_dim - self.tvp_dim]

        # TODO a better implem could be by spliting input BEFORE performing the hessian computation !
        hessian_np = hessian_np.reshape(x.shape[0], x.shape[1], (input_np.shape[1]-self.p_dim - self.tvp_dim)*input_np.shape[0],( input_np.shape[1]-self.p_dim - self.tvp_dim)*input_np.shape[0])

        reshape_indexer = sum([ list(np.arange(x.shape[1])+i*(x.shape[1]+u.shape[1])) for i in range(x.shape[0])  ], list()) + \
             sum([ list( x.shape[1]+np.arange(u.shape[1])+i*(x.shape[1]+u.shape[1])) for i in range(x.shape[0])  ], list())

        hessian_np = np.take(hessian_np, reshape_indexer, axis=2)
        hessian_np = np.take(hessian_np, reshape_indexer, axis=3)

        return hessian_np


@tf.function
def rolling_input(input_tf, x_dim, u_dim, rolling_window=2, H=2, forward=True):
    # TODO do not take into account p and tvp
    x = input_tf[:,0:x_dim]
    u = input_tf[:,x_dim:x_dim+u_dim]
    if forward:
        x_rolling = tf.stack([ tf.reshape(x[i:i+rolling_window, :],(-1,)) for i in range(H)], axis=0)
        u_rolling = tf.stack([ tf.reshape(u[i:i+rolling_window, :],(-1,))  for i in range(H)], axis=0)
    else:
        x_rolling = tf.stack([ tf.reshape( tf.reverse( x[i:i+rolling_window, :] , [0]),(-1,)) for i in range(H)], axis=0)
        u_rolling = tf.stack([ tf.reshape( tf.reverse( u[i:i+rolling_window, :] , [0]),(-1,))  for i in range(H)], axis=0)

    return tf.concat([x_rolling,u_rolling],axis=1)

class KerasTFModelRollingInput(Model):
    def __init__(self, model, x_dim: int, u_dim: int, p_dim=0, tvp_dim=0, rolling_window=2, forward_rolling=True,  standardScaler=None):

        if not standardScaler is None:
            raise NotImplementedError("This feature isn't supported yet !")

        # TODO make checking according to rolling_window
        
        #if not isinstance(model, (tf.keras.Model)):
        #    raise ValueError("The provided model isn't a Keras Model object !")

        #if len(model.input_shape) != 2:
        #    raise NotImplementedError("Recurrent neural network are not supported atm.")

        if model.output_shape[-1] != x_dim:
            raise ValueError("Your Keras model do not provide a suitable output dim ! \n It must get the same dim as the state dim.")

        #if model.input_shape[-1] != sum((x_dim, u_dim, p_dim, tvp_dim)):
        #    raise ValueError("Your Keras model do not provide a suitable input dim ! \n It must get the same dim as the sum of all input vars (x, u, p, tvp).")
        
        if not isinstance(rolling_window, int) or rolling_window<1:
            raise ValueError("Your rolling windows need to be an integer gretter than 1.")

        super(KerasTFModelRollingInput, self).__init__(x_dim, u_dim, p_dim, tvp_dim)

        self.model = model
        self.rolling_window = rolling_window
        self.forward_rolling = forward_rolling
        self.prev_x, self.prev_u, self.prev_tvp = None, None, None


        self.jacobian_proj = None

    def __getstate__(self):
        result_dict = copy(self.__dict__)
        result_dict["prev_x"] = None
        result_dict["prev_u"] = None
        result_dict["model"] = None
        result_dict["prev_tvp"] = None      

        return result_dict

    def __setstate__(self, d):
        self.__dict__ = d

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

        output_np = np.concatenate([x_extended, u_extended], axis=1)

        if not tvp is None:
            tvp_extended = np.concatenate([self.prev_tvp, tvp], axis=0)
            output_np = np.concatenate([output_np, tvp_extended], axis=1)

        if not p is None:
            output_np = np.concatenate([output_np, np.array([[p,]*x.shape[0]])], axis=1)

        return output_np


    def _gather_input_V2(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):

        assert (not self.prev_x is None) and (not self.prev_u is None), "You must give history window with set_prev_data before calling any inferance function."

        x_extended = np.concatenate([self.prev_x, x], axis=0)
        u_extended = np.concatenate([self.prev_u, u], axis=0)
        

        if self.forward_rolling:
            x_rolling = np.stack([ x_extended[i:i+self.rolling_window, :].reshape(-1) for i in range(x.shape[0])], axis=0)
            u_rolling = np.stack([ u_extended[i:i+self.rolling_window, :].reshape(-1) for i in range(x.shape[0])], axis=0)
            if not tvp is None:
                assert (not self.prev_tvp is None), "You must give history window with set_prev_data before calling any inferance function."
                tvp_extended = np.concatenate([self.prev_tvp, tvp], axis=0)
                tvp_rolling = np.stack([ tvp_extended[i:i+self.rolling_window, :].reshape(-1) for i in range(x.shape[0])], axis=0)
        else:
            x_rolling = np.stack([ (x_extended[i:i+self.rolling_window, :])[::-1,:].reshape(-1) for i in range(x.shape[0])], axis=0)
            u_rolling = np.stack([ (u_extended[i:i+self.rolling_window, :])[::-1,:].reshape(-1) for i in range(x.shape[0])], axis=0)
            if not tvp is None:
                assert (not self.prev_tvp is None), "You must give history window with set_prev_data before calling any inferance function."
                tvp_extended = np.concatenate([self.prev_tvp, tvp], axis=0)
                tvp_rolling = np.stack([ (tvp_extended[i:i+self.rolling_window, :])[::-1,:].reshape(-1) for i in range(x.shape[0])], axis=0)

        output_np = np.concatenate([x_rolling, u_rolling], axis=1)

        if not tvp is None:
            output_np = np.concatenate([output_np, tvp_rolling], axis=1)
        if not p is None:
            output_np = np.concatenate([output_np, np.array([[p,]*x.shape[0]])], axis=1)
        return output_np

    def forward(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        input_net = self._gather_input(x, u, p=p, tvp=tvp)
        input_net_rolled =  rolling_input(input_net, self.x_dim, self.u_dim, rolling_window=self.rolling_window, H=x.shape[0], forward=self.forward_rolling)
        res =  self.model.predict(input_net_rolled)
        if not isinstance(res, np.ndarray):
            return res.numpy()
        return res


    def jacobian(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        input_net = self._gather_input(x, u, p=p, tvp=tvp)

        input_tf = tf.constant(input_net)

        if self.jacobian_proj is None:
            with tf.GradientTape(persistent=False) as tx:
                tx.watch(input_tf)
                input_tf_rolled =  rolling_input(input_tf, self.x_dim, self.u_dim, rolling_window=self.rolling_window, H=x.shape[0], forward=self.forward_rolling)
            self.jacobian_proj = tx.jacobian(input_tf_rolled, input_tf)
        else:
            input_tf_rolled =  rolling_input(input_tf, self.x_dim, self.u_dim, rolling_window=self.rolling_window, H=x.shape[0], forward=self.forward_rolling)  

        with tf.GradientTape(persistent=False) as tx:
            tx.watch(input_tf_rolled)
            output_tf = self.model(input_tf_rolled)
        pre_jac_tf = tx.jacobian(output_tf, input_tf_rolled)
        
        jacobian_tf = tf.einsum("abcd,cdef->abef", pre_jac_tf, self.jacobian_proj)

        jacobian_np = jacobian_tf.numpy().reshape(x.shape[0]*self.x_dim, (self.x_dim+self.u_dim)*(x.shape[0]+self.rolling_window-1))
        reshape_indexer = sum([ list(np.arange(x.shape[1])+i*(x.shape[1]+u.shape[1])) for i in range(self.rolling_window-1 ,x.shape[0]+self.rolling_window-1)  ], list()) + \
             sum([ list( x.shape[1]+np.arange(u.shape[1])+i*(x.shape[1]+u.shape[1])) for i in range(self.rolling_window-1  ,x.shape[0]+self.rolling_window-1)  ], list())

        jacobian_np = np.take(jacobian_np, reshape_indexer, axis=1)

        return jacobian_np
    
    def jacobian_old(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        input_net = self._gather_input(x, u, p=p, tvp=tvp)

        input_tf = tf.constant(input_net)

        with tf.GradientTape(persistent=False) as tx:
            tx.watch(input_tf)
            input_tf_rolled =  rolling_input(input_tf, self.x_dim, self.u_dim, rolling_window=self.rolling_window, H=x.shape[0], forward=self.forward_rolling)
            output_tf = self.model(input_tf_rolled)
        
        jacobian_tf = tx.jacobian(output_tf, input_tf)

        jacobian_np = jacobian_tf.numpy().reshape(x.shape[0]*self.x_dim, (self.x_dim+self.u_dim)*(x.shape[0]+self.rolling_window-1))
        reshape_indexer = sum([ list(np.arange(x.shape[1])+i*(x.shape[1]+u.shape[1])) for i in range(self.rolling_window-1 ,x.shape[0]+self.rolling_window-1)  ], list()) + \
             sum([ list( x.shape[1]+np.arange(u.shape[1])+i*(x.shape[1]+u.shape[1])) for i in range(self.rolling_window-1  ,x.shape[0]+self.rolling_window-1)  ], list())

        jacobian_np = np.take(jacobian_np, reshape_indexer, axis=1)

        return jacobian_np

    @tf.function
    def _hessian_compute(self, input_tf):
        H = tf.shape(input_tf)[0]

        hessian_mask = tf.reshape(tf.eye(H*self.model.output_shape[-1],H*self.model.output_shape[-1]), (H*self.model.output_shape[-1],H,self.model.output_shape[-1]))
        hessian_mask = tf.cast(hessian_mask, tf.float32)

        output_tf =  self.model(input_tf)
        result = tf.map_fn(lambda mask : tf.hessians(output_tf*mask, input_tf)[0] , hessian_mask, dtype=tf.float32)

        return result

    def hessian(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        input_np = self._gather_input_V2(x, u, p=p, tvp=tvp)

        input_tf = tf.constant(input_np, dtype=tf.float32)
        #if self.test is None:
        #    self.test  = self._hessian_compute.get_concrete_function(input_tf=tf.TensorSpec([input_tf.shape[0], input_tf.shape[1]], tf.float64), output_shape=int(self.model.output_shape[-1]))
        #hessian_np = self.test(input_tf, int(self.model.output_shape[-1])).numpy()
        
        hessian_np = self._hessian_compute(input_tf).numpy()
        # TODO a better implem could be by spliting input BEFORE performing the hessian computation !
        hessian_np = hessian_np.reshape(x.shape[0], x.shape[1], input_np.shape[1]*input_np.shape[0], input_np.shape[1]*input_np.shape[0])



        project_mat = np.zeros(shape=(input_np.shape[1]*x.shape[0], (self.x_dim+self.u_dim)*(x.shape[0]+self.rolling_window-1)))

        for dt in range(x.shape[0]):
            project_mat[dt*self.rolling_window*(self.x_dim+self.u_dim):dt*self.rolling_window*(self.x_dim+self.u_dim)+self.x_dim*self.rolling_window, dt*self.x_dim : dt*self.x_dim + self.x_dim*self.rolling_window] += np.eye(self.x_dim*self.rolling_window, self.x_dim*self.rolling_window)
            project_mat[self.x_dim*self.rolling_window +   dt*self.rolling_window*(self.x_dim+self.u_dim):self.x_dim*self.rolling_window + dt*self.rolling_window*(self.x_dim+self.u_dim)+self.u_dim*self.rolling_window,self.x_dim*(x.shape[0]+self.rolling_window-1)+ dt*self.u_dim :self.x_dim*(x.shape[0]+self.rolling_window-1)  + dt*self.u_dim + self.u_dim*self.rolling_window] += np.eye(self.u_dim*self.rolling_window, self.u_dim*self.rolling_window)

        def dot1(A, B):
            np.einsum("")
        res = np.einsum( "gc,abcf->abgf", project_mat.T, np.einsum("abcd,df->abcf",hessian_np, project_mat))


        reshape_indexer = list(range(self.x_dim*(self.rolling_window-1) , self.x_dim*(x.shape[0]+self.rolling_window-1))) + \
            list(range(self.x_dim*(x.shape[0]+self.rolling_window-1) +  self.u_dim*(self.rolling_window-1) ,self.x_dim*(x.shape[0]+self.rolling_window-1)+ self.u_dim*(x.shape[0]+self.rolling_window-1)))

        res = np.take(res, reshape_indexer, axis=2)
        res = np.take(res, reshape_indexer, axis=3)


        return res
