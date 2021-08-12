import tensorflow as tf 
import numpy as np  

from .base import Model
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class KerasTFModel(Model):
    def __init__(self, model, x_dim: int, u_dim: int, p_dim=0, tvp_dim=0, standardScaler=None):

        if not standardScaler is None:
            raise NotImplementedError("This feature isn't supported yet !")

        if not isinstance(model, (tf.keras.Model)):
            raise ValueError("The provided model isn't a Keras Model object !")

        if len(model.input_shape) != 2:
            raise NotImplementedError("Recurrent neural network are not supported atm.")

        if model.output_shape[-1] != x_dim:
            raise ValueError("Your Keras model do not provide a suitable output dim ! \n It must get the same dim as the state dim.")

        if model.input_shape[-1] != sum((x_dim, u_dim, p_dim, tvp_dim)):
            raise ValueError("Your Keras model do not provide a suitable input dim ! \n It must get the same dim as the sum of all input vars (x, u, p, tvp).")
        
        super(KerasTFModel, self).__init__(x_dim, u_dim, p_dim, tvp_dim)

        self.model = model
        self.test = None

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

        jacobian_np = jacobian_tf.numpy().reshape(x.shape[0]*self.x_dim, (self.x_dim+self.u_dim)*x.shape[0])
        

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

        # TODO a better implem could be by spliting input BEFORE performing the hessian computation !
        hessian_np = hessian_np.reshape(x.shape[0], x.shape[1], input_np.shape[1]*input_np.shape[0], input_np.shape[1]*input_np.shape[0])
        
        reshape_indexer = sum([ list(np.arange(x.shape[1])+i*(x.shape[1]+u.shape[1])) for i in range(x.shape[0])  ], list()) + \
             sum([ list( x.shape[1]+np.arange(u.shape[1])+i*(x.shape[1]+u.shape[1])) for i in range(x.shape[0])  ], list())

        hessian_np = np.take(hessian_np, reshape_indexer, axis=2)
        hessian_np = np.take(hessian_np, reshape_indexer, axis=3)

        return hessian_np
