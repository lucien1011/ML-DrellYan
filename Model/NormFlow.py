import tensorflow as tf
import tensorflow.keras.backend as K

class NormFlow(tf.keras.Model):
    def __init__(self,dim,flow_length):
        super(NormFlow,self).__init__()
        self.dim = dim
        self.flow_length = flow_length
        self.activation = tf.keras.activations.tanh

        #self.flow_layers = [PlanarFlowLayer(self.dim,self.activation) for _ in range(self.flow_length)]
        self.dense_layers = [tf.keras.layers.Dense(32,activation='relu') for _ in range(5)]
        self.inter_layer = tf.keras.layers.Dense(2*self.dim,activation='linear')
        self.sampling_layer = NormalSampling(self.dim)

    def call(self,inputs):
        z = inputs
        for dl in self.dense_layers:
            z = dl(z)
        z = self.inter_layer(z)
        z = self.sampling_layer(z)
        #for fl in self.flow_layers:
        #    z = fl(z)
        return z

class NormalSampling(tf.keras.layers.Layer):
    def __init__(self,dim):
        super(NormalSampling,self).__init__()
        self.dim = dim

    def call(self, input):
        z_mean = input[:,:self.dim]
        z_log_var = input[:,self.dim:]
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, 1))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class PlanarFlowLayer(tf.keras.layers.Layer):
    def __init__(self,dim,activation,):
        super(PlanarFlowLayer,self).__init__()
        self.dim = dim
        self.activation = activation

    def build(self,input_shape):
        self.w = self.add_weight("w",shape=[self.dim,1])
        self.b = self.add_weight("b",shape=[1,])
        self.u = self.add_weight("u",shape=[self.dim,1])

    def call(self,input):
        activation = self.activation(tf.matmul(input,self.w)+self.b)
        psi = (1 - self.activation(activation) ** 2)
        det_grad = 1 + tf.matmul(tf.transpose(self.u),self.w) * psi
        self.add_loss(tf.reduce_mean(tf.math.log(tf.math.abs(det_grad))))
        return input + tf.matmul(activation,tf.transpose(self.u))

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
