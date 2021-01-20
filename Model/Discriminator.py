import tensorflow as tf

reg = 0.20

class Discriminator(tf.keras.Model):
    def __init__(self,neuron_number_list,epsilon=1e-3,**kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.neuron_number_list = neuron_number_list
        self._dense_layer_list = []
        #self._dropout_layer_list = []
        #self._leaky_layer_list = []
        #self._batchNorm_layer_list = []
        for i,neuron_number in enumerate(neuron_number_list):
            self._dense_layer_list.append(tf.keras.layers.Dense(neuron_number,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(reg),))
            #self._leaky_layer_list.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            #self._dropout_layer_list.append(tf.keras.layers.Dropout(0.1))
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        self.epsilon = epsilon

    def call(self, inputs, Training=True):
        disc_out = tf.cast(inputs,tf.float32)
        for i,neuron_number in enumerate(self.neuron_number_list):
            disc_out = self._dense_layer_list[i](disc_out)
            #disc_out = self._leaky_layer_list[i](disc_out)
            #disc_out = self._dropout_layer_list[i](disc_out)
        disc_out = self.output_layer(disc_out)
        return disc_out+self.epsilon
