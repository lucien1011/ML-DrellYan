import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import regularizers

def make_default_masks(num_coupling_layers,ndim):
    mask_list = [[0 for _ in range(ndim)] for _ in range(ndim)]
    for i in range(ndim):
        if i < ndim - 1:
            mask_list[i][i+1] = 1
        else:
            mask_list[i][0] = 1
    masks = np.array(
            mask_list * (num_coupling_layers // 2), dtype="float32",
            )
    return masks

def Coupling(input_shape,hidden_dim=16,reg=0.2,):
    input = tf.keras.layers.Input(shape=input_shape)

    t_out = input
    for n in [1,2,4,4,2,1]:
        t_out = tf.keras.layers.Dense(n*hidden_dim,activation='relu',kernel_regularizer=regularizers.l2(reg))(t_out)
    t_out = tf.keras.layers.Dense(
            input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
            )(t_out)

    s_out = input
    for n in [1,2,4,4,2,1]:
        s_out = tf.keras.layers.Dense(n*hidden_dim,activation='relu',kernel_regularizer=regularizers.l2(reg))(s_out)
    s_out = tf.keras.layers.Dense(
            input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
            )(s_out)

    return tf.keras.Model(inputs=input, outputs=[s_out, t_out])

class RealNVP(tf.keras.Model):
    def __init__(self,num_coupling_layers,ndim,masks=None,epsilon=1e-3):
        super(RealNVP,self).__init__()
        
        self.num_coupling_layers = num_coupling_layers

        self.distribution = tfp.distributions.MultivariateNormalDiag(
                loc=[0.0 for _ in range(ndim)],scale_diag=[1.0 for _ in range(ndim)],
                )
        self.masks = masks if masks else make_default_masks(num_coupling_layers,ndim)
        self.layers_list = [Coupling(ndim) for i in range(num_coupling_layers)]
        self.direction = 1
        self.epsilon = epsilon

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

    def batch_log_loss(self, x):
        y, logdet = self(x)
        log_likelihood = self.distribution.log_prob(y+self.epsilon) + logdet
        return -log_likelihood

def ConditionalCoupling(input_shape,conditional_shape,hidden_dim=128,reg=0.05,):
    input = tf.keras.layers.Input(shape=input_shape)
    condition = tf.keras.layers.Input(shape=conditional_shape)

    concat_layer_0 = tf.keras.layers.Lambda(lambda x: tf.concat(x,axis=1))([input,condition,])

    t_out = concat_layer_0
    for n in [1,2,4,4,2,1]:
        t_out = tf.keras.layers.Dense(n*hidden_dim,activation='relu',kernel_regularizer=regularizers.l2(reg))(t_out)
        #t_out = tf.keras.layers.Lambda(lambda x: tf.concat(x,axis=1))([t_out,condition,])
    t_out = tf.keras.layers.Dense(
            input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
            )(t_out)

    s_out = concat_layer_0
    for n in [1,2,4,4,2,1]:
        s_out = tf.keras.layers.Dense(n*hidden_dim,activation='relu',kernel_regularizer=regularizers.l2(reg))(s_out)
        #s_out = tf.keras.layers.Lambda(lambda x: tf.concat(x,axis=1))([s_out,condition,])
    s_out = tf.keras.layers.Dense(
            input_shape, activation="linear", kernel_regularizer=regularizers.l2(reg)
            )(s_out)

    return tf.keras.Model(inputs=[input,condition,], outputs=[s_out, t_out])

class ConditionalRealNVP(tf.keras.Model):
    def __init__(self,num_coupling_layers,ndim,ncond,masks=None,epsilon=1e-3):
        super(ConditionalRealNVP,self).__init__()
        
        self.num_coupling_layers = num_coupling_layers

        self.distribution = tfp.distributions.MultivariateNormalDiag(
                loc=[0.0 for _ in range(ndim)],
                scale_diag=[1.0 for _ in range(ndim)],
                )
        self.masks = masks if masks else make_default_masks(num_coupling_layers,ndim)
        self.layers_list = [ConditionalCoupling(ndim,ncond) for i in range(num_coupling_layers)]
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
        #print("s,log: ",gate,s[:10],log_det_inv[:10])
        return x, log_det_inv

    def log_loss(self, inputs):
        x,condition = inputs
        y, logdet = self(inputs)
        log_likelihood = self.distribution.log_prob(y+self.epsilon) + logdet
        #print("y: ",y[:10])
        #print("total log: ",self.distribution.log_prob(y+self.epsilon)[:10],logdet[:10],log_likelihood[:10])
        return -tf.reduce_mean(log_likelihood)

    def batch_log_loss(self, inputs):
        x,condition = inputs
        y, logdet = self(inputs)
        log_likelihood = self.distribution.log_prob(y+self.epsilon) + logdet
        return -log_likelihood
