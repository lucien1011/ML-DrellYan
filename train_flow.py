import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from KerasSimpleTrainer.MiniBatchTrainer import MiniBatchTrainer

from process_data import make_daflow_data
from Model.RealNVP import RealNVP
from Utils.ObjDict import ObjDict

# ____________________________________________________________ ||
input_csv_path      = "data/train.npy"
n_epoch             = 500
batch_size          = 512
print_per_point     = 50
plot_per_point      = 50
save_per_point      = 1000
output_path         = "output/train_flow_210104_v2/"
saved_model_name    = "saved_model.h5"
lambda_tran         = 1E-4

plot_cfgs           = [
        ObjDict(name="pt1",bins=100,range=[0.,4.],index=0,histtype='step',),
        ObjDict(name="eta1",bins=100,range=[-4.,4.],index=1,histtype='step',),
        ObjDict(name="pt2",bins=100,range=[0.,4.],index=2,histtype='step',),
        ObjDict(name="eta2",bins=100,range=[-4.,4.],index=3,histtype='step',),
        ObjDict(name="dphi",bins=100,range=[1.,5.],index=4,histtype='step',),
        ]

# ____________________________________________________________ ||
arr = np.load(input_csv_path)
x_orig,x_smear = make_daflow_data(arr)
x_orig_train,x_orig_test,x_smear_train,x_smear_test = train_test_split(x_orig,x_smear,test_size=0.1,random_state=42)

ndim = x_orig_train.shape[1]

# ____________________________________________________________ ||
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
        disc_out = tf.cast(inputs,tf.float32)
        for i,neuron_number in enumerate(self.neuron_number_list):
            disc_out = self._dense_layer_list[i](disc_out)
            disc_out = self._leaky_layer_list[i](disc_out)
            disc_out = self._dropout_layer_list[i](disc_out)
        disc_out = self.output_layer(disc_out)
        return disc_out

disc = Discriminator([128,128,128])

tran_model = RealNVP(num_coupling_layers=5,dim=ndim,)  
da_model = RealNVP(num_coupling_layers=10,dim=ndim,)
db_model = RealNVP(num_coupling_layers=10,dim=ndim,)

# ____________________________________________________________ ||
batch_trainer = MiniBatchTrainer()
nf_optimizer = tf.keras.optimizers.Adam()
disc_optimizer = tf.keras.optimizers.Adam()
tran_optimizer = tf.keras.optimizers.Adam()
bce = tf.keras.losses.BinaryCrossentropy()
for i_epoch in range(n_epoch):
    
    for _ in range(1):

        idx_train = np.random.randint(0, x_orig_train.shape[0], batch_size)
            
        da_model.direction = -1
        db_model.direction = -1
        with tf.GradientTape() as tape:
            da_loss = da_model.log_loss(x_orig_train[idx_train])
            db_loss = db_model.log_loss(x_smear_train[idx_train])
            nf_loss = da_loss + db_loss 
        grads = tape.gradient(nf_loss, da_model.trainable_weights + db_model.trainable_weights)
        nf_optimizer.apply_gradients(zip(grads, da_model.trainable_weights + db_model.trainable_weights))

        idx_train = np.random.randint(0, x_orig_train.shape[0], batch_size)
        
        ones = np.ones((batch_size,1))
        zeros = np.zeros((batch_size,1))
        
        with tf.GradientTape() as tape:
            real_loss = bce(ones,disc(x_smear_train[idx_train]))
            fake_loss = bce(zeros,disc(x_orig_train[idx_train]))
            disc_loss = real_loss + fake_loss
        grads = tape.gradient(disc_loss, disc.trainable_weights)
        disc_optimizer.apply_gradients(zip(grads, disc.trainable_weights))

    da_model.direction = -1
    db_model.direction = 1
    tran_model.direction = -1
    with tf.GradientTape() as tape:
        z_orig,_ = da_model(x_orig_train[idx_train])
        z_tran,logdet = tran_model(z_orig)
        log_likelihood = tran_model.distribution.log_prob(z_tran) + logdet
        x_tran,_ = db_model(z_tran)
        tran_loss = bce(ones,disc(x_tran)) - tf.reduce_mean(log_likelihood)
    grads = tape.gradient(tran_loss, tran_model.trainable_weights)
    tran_optimizer.apply_gradients(zip(grads, tran_model.trainable_weights))

    batch_trainer.add_loss("NF loss",nf_loss)
    batch_trainer.add_loss("Disc loss",disc_loss)
    batch_trainer.add_loss("Tran loss",tran_loss)

    batch_trainer.add_epoch()
    batch_trainer.print_loss(print_per_point)
    
    batch_trainer.make_history_plot(
            os.path.join(output_path,"loss.png"),
            log_scale=True,
            )

    if i_epoch % plot_per_point == 0 or i_epoch == n_epoch - 1:
        samples_da = da_model.distribution.sample(x_orig_test.shape[0])
        da_model.direction = 1
        x_gen_da,_ = da_model.predict(samples_da)

        samples_db = db_model.distribution.sample(x_orig_test.shape[0])
        db_model.direction = 1
        x_gen_db,_ = db_model.predict(samples_db)

        samples_tran,_ = tran_model.predict(samples_da)
        x_tran,_ = db_model.predict(samples_tran)

        fig,ax = plt.subplots(ndim,1,figsize=(15,50))
        for ip,p in enumerate(plot_cfgs):
            ax[ip].hist(x_orig_test[:,ip],bins=p.bins,label="orig",density=1.,histtype=p.histtype,range=p.range,)
            ax[ip].hist(x_smear_test[:,ip],bins=p.bins,label="smear",density=1.,histtype=p.histtype,range=p.range,)
            ax[ip].hist(x_gen_da[:,ip],bins=p.bins,label="flow A",density=1.,histtype=p.histtype,range=p.range,)
            ax[ip].hist(x_gen_db[:,ip],bins=p.bins,label="flow B",density=1.,histtype=p.histtype,range=p.range,)
            ax[ip].hist(x_tran[:,ip],bins=p.bins,label="flow tran",density=1.,histtype=p.histtype,range=p.range,)
            ax[ip].legend(loc='best')
        fig.savefig(os.path.join(output_path,"plot"+str(i_epoch)+".png"))

batch_trainer.save_weights(model,os.path.join(output_path,saved_model_name),)

