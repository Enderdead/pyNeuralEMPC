import tensorflow as tf 
import numpy as np  

from .base import Model


class KerasTFModel(Model):
    def __init__(self, model, standardScaler=None):
        if not standardScaler is None:
            raise NotImplementedError("This feature isn't supported yet !")

        if not isinstance(model, (tf.keras.Model)):
            raise ValueError("The provided model isn't a Keras Model object !")

        if len(model.input_shape) != 2:
            raise NotImplementedError("Recurrent neural network are not supported atm.")

        super(KerasTFModel).__init__(model, model.output_shape[-1])

        self.model = model

    def forward(self, x):
        return self.model.predict(x)

    def jacobian(self, x):
        input_tf = tf.constant(x)

        with tf.GradientTape(persistent=False) as tx:
            tx.watch(input_tf)
            output_tf = self.model(tx)
        
        jacobian_tf = tx.jacobian(output_tf, input_tf)
        jacobian_np = np.stack([ jacobian_tf[i,:,i,:].numpy() for i in range(x.shape[0]) ]) 

        return jacobian_np

    def hessian(self, x):
        input_tf = tf.constant(x)

        hessian_tf_list = list()

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
