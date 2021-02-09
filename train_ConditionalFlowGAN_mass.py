import os,pickle
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from KerasSimpleTrainer.MiniBatchTrainer import MiniBatchTrainer

from process_data import preprocess_conditional_flow_data_mass,simuate_conditional_flow_data_mass
from Model.ConditionalRealNVP import ConditionalRealNVP
from Model.ConditionalDiscriminator import Discriminator
from Utils.ObjDict import ObjDict
from Utils.mkdir_p import mkdir_p

# ____________________________________________________________ ||
input_csv_path      = "data/train_mass.npy"
n_epoch             = 5000
batch_size          = 1
event_size          = 1000
print_per_point     = 50
plot_per_point      = 50
save_per_point      = 50
output_path         = "output/train_ConditionalFlowGAN_mass_210210_v1/"
saved_model_name    = "saved_model.h5"

plot_cfgs           = [
        ObjDict(name="pt1",bins=50,range=[-5.,5.],index=0,histtype='step',),
        ObjDict(name="pt2",bins=50,range=[-5.,5.],index=2,histtype='step',),
        ObjDict(name="mll",bins=50,range=[-5.,5.],index=2,histtype='step',),
        ]
flow_linestyle      = ':'
true_linestyle      = '-'

# ____________________________________________________________ ||
arr = np.load(input_csv_path)
arr_list = preprocess_conditional_flow_data_mass(arr)

ndim = 3
ncond = 1

# ____________________________________________________________ ||
nf_model = ConditionalRealNVP(num_coupling_layers=3,ndim=ndim,ncond=ncond)
disc_model = Discriminator([32,64,128,128,64,32,])

# ____________________________________________________________ ||
batch_trainer = MiniBatchTrainer()
nf_optimizer = tf.keras.optimizers.Adam()
disc_optimizer = tf.keras.optimizers.Adam()
mkdir_p(output_path)
bce = tf.keras.losses.BinaryCrossentropy()
for i_epoch in range(n_epoch):
    
    idx_train = np.random.randint(0, len(arr_list), batch_size)
    x_train = np.concatenate([arr_list[idx].x for idx in idx_train])
    condition_train = np.concatenate([arr_list[idx].condition for idx in idx_train])

    idx_batch = np.random.randint(0, x_train.shape[0], event_size)
    x_train = x_train[idx_batch]
    condition_train = condition_train[idx_batch]

    ones = np.ones((event_size,1))
    zeros = np.zeros((event_size,1))
    
    with tf.GradientTape() as tape:
        nf_model.direction = 1
        samples = nf_model.distribution.sample(condition_train.shape[0])
        x_pred,_ = nf_model([samples,condition_train,])
        
        real_disc_loss = bce(ones,disc_model([x_train,condition_train]))
        fake_disc_loss = bce(zeros,disc_model([x_pred,condition_train]))
        disc_loss = real_disc_loss + fake_disc_loss

    grads = tape.gradient(disc_loss, disc_model.trainable_weights)
    disc_optimizer.apply_gradients(zip(grads, disc_model.trainable_weights))
    
    #with tf.GradientTape() as tape:
    #    nf_model.direction = -1
    #    nf_loss = nf_model.log_loss([x_train,condition_train])
    #grads = tape.gradient(nf_loss, nf_model.trainable_weights)
    #nf_optimizer.apply_gradients(zip(grads, nf_model.trainable_weights))

    with tf.GradientTape() as tape:
        nf_model.direction = 1
        samples = nf_model.distribution.sample(condition_train.shape[0])
        x_pred,_ = nf_model([samples,condition_train,])
        gen_loss = bce(ones,disc_model([x_pred,condition_train]))
    grads = tape.gradient(gen_loss, nf_model.trainable_weights)
    nf_optimizer.apply_gradients(zip(grads, nf_model.trainable_weights))
    
    #batch_trainer.add_loss("NF loss",nf_loss)
    batch_trainer.add_loss("disc loss",disc_loss)
    batch_trainer.add_loss("gen loss",gen_loss)

    batch_trainer.add_epoch()
    batch_trainer.print_loss(print_per_point)
    
    batch_trainer.make_history_plot(
            os.path.join(output_path,"loss.png"),
            log_scale=True,
            )

    if i_epoch % plot_per_point == 0 or i_epoch == n_epoch - 1:

        nf_model.direction = 1
        
        samples = nf_model.distribution.sample(condition_train.shape[0])
        x_pred,_ = nf_model.predict([samples,condition_train,])     

        fig,ax = plt.subplots(ndim,1,figsize=(15,50))
        for ip,p in enumerate(plot_cfgs):
            ax[ip].hist(x_pred[:,ip],bins=p.bins,label="flow",density=1.,histtype=p.histtype,range=p.range,linestyle=flow_linestyle,color='b')
            ax[ip].hist(x_train[:,ip],bins=p.bins,label="true",density=1.,histtype=p.histtype,range=p.range,color='b')

            ax[ip].legend(loc='best')
        fig.savefig(os.path.join(output_path,"plot"+str(i_epoch)+".png"))

    batch_trainer.save_weights(nf_model,os.path.join(output_path,saved_model_name),save_per_point=save_per_point,)

batch_trainer.save_weights(nf_model,os.path.join(output_path,saved_model_name),)
