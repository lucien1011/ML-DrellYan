import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import regularizers

output_dim = 256
reg = 0.1

def Coupling(input_shape):
    input = tf.keras.layers.Input(shape=input_shape)

    t_layer_1 = tf.keras.layers.Dense(
            output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
            )(input)
    t_layer_2 = tf.keras.layers.Dense(
            2*output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
            )(t_layer_1)
    t_layer_3 = tf.keras.layers.Dense(
            2*output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
            )(t_layer_2)
    t_layer_4 = tf.keras.layers.Dense(
            input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
            )(t_layer_3)

    s_layer_1 = tf.keras.layers.Dense(
            output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
            )(input)
    s_layer_2 = tf.keras.layers.Dense(
            2*output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
            )(s_layer_1)
    s_layer_3 = tf.keras.layers.Dense(
            2*output_dim, activation="relu", kernel_regularizer=regularizers.l2(reg)
            )(s_layer_2)
    s_layer_4 = tf.keras.layers.Dense(
            input_shape, activation="tanh", kernel_regularizer=regularizers.l2(reg)
            )(s_layer_3)

    return tf.keras.Model(inputs=input, outputs=[s_layer_4, t_layer_4])

class RealNVP(tf.keras.Model):
    def __init__(self,num_coupling_layers,dim):
        super(RealNVP,self).__init__()
        
        self.num_coupling_layers = num_coupling_layers

        self.distribution = tfp.distributions.MultivariateNormalDiag(
                loc=[0.0 for _ in range(dim)],scale_diag=[1.0 for _ in range(dim)],
                )

        mask_list = [[0 for _ in range(dim)] for _ in range(dim)]
        for i in range(dim):
            if i < dim - 1:
                mask_list[i][i+1] = 1
            else:
                mask_list[i][0] = 1
        self.masks = np.array(
                mask_list * (num_coupling_layers // 2), dtype="float32",
                )
        self.layers_list = [Coupling(dim) for i in range(num_coupling_layers)]
        self.direction = 1

    def call(self,x,training=True):
        log_det_inv = 0
        for i in range(self.num_coupling_layers)[::self.direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (self.direction - 1) / 2
            x = (
                    reversed_mask
                    * (x * tf.exp(self.direction * s) + self.direction * t * tf.exp(gate * s))
                    + x_masked
                )
            log_det_inv += gate * tf.reduce_sum(s, [1])
        return x, log_det_inv

    def log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y) + logdet
        return -tf.reduce_mean(log_likelihood)
