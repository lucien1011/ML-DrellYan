import numpy

import tensorflow as tf
import tensorflow.keras.backend as K

def build_model(n_reco,n_param,disc_neuron_list,gen_neuron_list):
    x_input_layer = tf.keras.layers.Input(shape=(n_reco,),name="x_input",)
    condition_input_layer = tf.keras.layers.Input(shape=(n_param,),name="condition_input",)

    discriminator = Discriminator(disc_neuron_list,)
    generator = Generator(n_reco,gen_neuron_list,)
    return discriminator,generator,x_input_layer,condition_input_layer

class Discriminator(tf.keras.Model):
    def __init__(self,
            neuron_number_list,
            **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.neuron_number_list = neuron_number_list
        self._dense_layer_list = []
        self._dropout_layer_list = []
        self._leaky_layer_list = []
        self._batchNorm_layer_list = []
        for i,neuron_number in enumerate(neuron_number_list):
            self._dense_layer_list.append(tf.keras.layers.Dense(neuron_number,activation='relu',))
            self._leaky_layer_list.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self._dropout_layer_list.append(tf.keras.layers.Dropout(0.1))
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs, Training=True):
        x,condition = inputs
        x = tf.cast(x,tf.float32)
        condition = tf.cast(condition,tf.float32)
        disc_out = tf.concat([x,condition,], axis=-1)
        for i,neuron_number in enumerate(self.neuron_number_list):
            disc_out = self._dense_layer_list[i](disc_out)
            disc_out = self._leaky_layer_list[i](disc_out)
            disc_out = self._dropout_layer_list[i](disc_out)
            disc_out = tf.concat([disc_out,condition,], axis=-1)
        disc_out = self.output_layer(disc_out)
        return disc_out

class Generator(tf.keras.Model):
    def __init__(self,
        ndf,
        neuron_number_list,
        **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.neuron_number_list = neuron_number_list
        self._dense_layer_list = []
        self._dropout_layer_list = []
        self._leaky_layer_list = []
        
        for i,neuron_number in enumerate(neuron_number_list):
            self._dense_layer_list.append(tf.keras.layers.Dense(neuron_number,activation='relu',))
            self._leaky_layer_list.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self._dropout_layer_list.append(tf.keras.layers.Dropout(0.1))
        self._output_layer = tf.keras.layers.Dense(ndf,activation='linear',)

    def call(self, inputs, Training=True):
        gen_out = tf.concat(inputs,axis=-1)
        for i,neuron_number in enumerate(self.neuron_number_list):
            gen_out = self._dense_layer_list[i](gen_out)
            gen_out = self._leaky_layer_list[i](gen_out)
            gen_out = self._dropout_layer_list[i](gen_out)
        gen_out = self._output_layer(gen_out)
        return gen_out
