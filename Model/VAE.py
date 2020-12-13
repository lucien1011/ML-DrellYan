import tensorflow as tf
import tensorflow.keras.backend as K

class VAE(tf.keras.Model):
    def __init__(self,dim,enc_number_list,dec_number_list):
        super(VAE,self).__init__()
        self.dim = dim
        self.enc_layer_list = [tf.keras.layers.Dense(n,activation='tanh') for n in enc_number_list[:-1]]
        self.dec_layer_list = [tf.keras.layers.Dense(n,activation='tanh') for n in dec_number_list]
        self.enc_out_layer = tf.keras.layers.Dense(enc_number_list[-1],activation='tanh')
        self.enc_sampling_layer = NormalSampling(int(enc_number_list[-1]/2))
        self.dec_out_layer = tf.keras.layers.Dense(self.dim,activation='linear')

    def call(self,input):
        x,condition = input
        out = self.predict_latent_vector(input)
        out = tf.concat([out,condition],axis=1)
        out = self.decode_latent_vector(out)
        return out

    def predict_latent_vector(self,input):
        x,condition = input
        out = tf.concat([x,condition],axis=1)
        for l in self.enc_layer_list:
            out = l(out)
        out = self.enc_out_layer(out)
        out = self.enc_sampling_layer(out)
        return out

    def decode_latent_vector(self,input):
        out = input
        for l in self.dec_layer_list:
            out = l(out)
        out = self.dec_out_layer(out)
        return out

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
        kl_loss = -0.5 * (1. + z_log_var - tf.exp(z_log_var) - z_mean**2)
        self.add_loss(tf.reduce_sum(kl_loss,axis=1))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
