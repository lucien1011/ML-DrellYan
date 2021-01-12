import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import regularizers

hidden_dim = 64
reg = 0.10

def ConditionalCoupling(input_shape,conditional_shape):
    input = tf.keras.layers.Input(shape=input_shape)
    condition = tf.keras.layers.Input(shape=conditional_shape)

    concat_layer_0 = tf.keras.layers.Lambda(lambda x: tf.concat(x,axis=1))([input,condition,])

    t_out = concat_layer_0
    for n in [1,2,4,4,2,1]:
        t_out = tf.keras.layers.Dense(n*hidden_dim,activation='relu',kernel_regularizer=regularizers.l2(reg))(t_out)
        t_out = tf.keras.layers.Lambda(lambda x: tf.concat(x,axis=1))([t_out,condition,])
    t_out = tf.keras.layers.Dense(
            input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
            )(t_out)

    s_out = concat_layer_0
    for n in [1,2,4,4,2,1]:
        s_out = tf.keras.layers.Dense(n*hidden_dim,activation='relu',kernel_regularizer=regularizers.l2(reg))(s_out)
        s_out = tf.keras.layers.Lambda(lambda x: tf.concat(x,axis=1))([s_out,condition,])
    s_out = tf.keras.layers.Dense(
            input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
            )(s_out)

    return tf.keras.Model(inputs=[input,condition,], outputs=[s_out, t_out])

class ConditionalRealNVP(tf.keras.Model):
    def __init__(self,num_coupling_layers,dim,ncond,epsilon=1e-3):
        super(ConditionalRealNVP,self).__init__()
        
        self.num_coupling_layers = num_coupling_layers

        self.distribution = tfp.distributions.MultivariateNormalDiag(
                loc=[0.0 for _ in range(dim)],
                scale_diag=[1.0 for _ in range(dim)],
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
        self.layers_list = [ConditionalCoupling(dim,ncond) for i in range(num_coupling_layers)]
        self.direction = 1
        self.epsilon = epsilon

    def call(self,inputs,training=True):
        x,condition = inputs
        log_det_inv = 0
        for i in range(self.num_coupling_layers)[::self.direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i]([x_masked,condition])
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

    def log_loss(self, inputs):
        x,condition = inputs
        y, logdet = self(inputs)
        log_likelihood = self.distribution.log_prob(y+self.epsilon) + logdet
        return -tf.reduce_mean(log_likelihood)

    def batch_log_loss(self, inputs):
        x,condition = inputs
        y, logdet = self(inputs)
        log_likelihood = self.distribution.log_prob(y+self.epsilon) + logdet
        return -log_likelihood
