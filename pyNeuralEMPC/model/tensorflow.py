import tensorflow as tf 
import numpy as np  

from .base import Model


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
        # TODO check this projection

        jacobian_np = jacobian_tf.numpy()[:,:,:,0:self.x_dim+self.u_dim]
        jacobian_np = jacobian_tf.numpy().reshape(self.H*self.x_dim, (self.x_dim+self.u_dim)*self.H)
        
        return jacobian_np

    def hessian(self, x: np.ndarray, u: np.ndarray, p=None, tvp=None):
        input_np = self._gather_input(x, u, p=p, tvp=tvp)
        input_tf = tf.constant(input_np)

        hessian_tf_list = list()
        raise ValueError("Need to be re implemented")

        for i in range(self.output_dim):

            with tf.GradientTape(persistent=False) as tx_2:
                tx_2.watch(input_tf)
                with tf.GradientTape(persistent=False) as tx_1:
                    tx_1.watch(input_tf)
                    output_tf =  self.model(input_tf)[:,i]
                g = tx_1.gradient(output_tf, input_tf)
                    
            hessian_tf_list.append(tx_2.jacobian(g, input_tf))

        hessian_np_list = [ np.stack([hessian[i,:,i,:].numpy()  for i in range(input_tf.shape[0])]) for hessian in hessian_tf_list]

        return hessian_np_list
