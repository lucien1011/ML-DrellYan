import os,pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from KerasSimpleTrainer.MiniBatchTrainer import MiniBatchTrainer

from process_data import preprocess_conditional_flow_data_cww
from Model.Discriminator import Discriminator
from Model.ConditionalRealNVP import ConditionalRealNVP,RealNVP
from Utils.ObjDict import ObjDict
from Utils.mkdir_p import mkdir_p

# ____________________________________________________________ ||
input_csv_path      = "data/train_cww.npy"
n_epoch             = 5000
batch_size          = 1
event_size          = 4096
print_per_point     = 10
plot_per_point      = 50
save_per_point      = 50
output_path         = "output/train_mixnlp_cww_210120_v2/"
saved_model_name    = "saved_model.h5"

plot_cfgs           = [
        ObjDict(name="m4l",bins=50,range=[-10.,10.],index=0,histtype='step',),
        ObjDict(name="mZ1",bins=50,range=[-5.,5.],index=1,histtype='step',),
        ObjDict(name="mZ2",bins=50,range=[-5.,5.],index=2,histtype='step',),
        #ObjDict(name="pT4l",bins=50,range=[-10.,10.],index=3,histtype='step',),
        #ObjDict(name="pTZ1",bins=50,range=[-10.,10.],index=4,histtype='step',),
        #ObjDict(name="pTZ2",bins=50,range=[-10.,10.],index=5,histtype='step',),
        ]
flow_linestyle      = ':'
true_linestyle      = '-'
plot_xdim           = 3
plot_ydim           = 2

# ____________________________________________________________ ||
arr = np.load(input_csv_path)
sigs,bkg = preprocess_conditional_flow_data_cww(arr)

ndim = 3
ncond = 1

# ____________________________________________________________ ||
disc_model = Discriminator([32,32,32,])
nf_sig_model = ConditionalRealNVP(num_coupling_layers=5,ndim=ndim,ncond=ncond,)  
nf_bkg_model = RealNVP(num_coupling_layers=5,ndim=ndim,)  

# ____________________________________________________________ ||
batch_trainer = MiniBatchTrainer()
nf_optimizer = tf.keras.optimizers.Adam()
disc_optimizer = tf.keras.optimizers.Adam()
bce = tf.keras.losses.BinaryCrossentropy()
mkdir_p(output_path)
for i_epoch in range(n_epoch):
    
    idx_sig_train = np.random.randint(0, len(sigs), batch_size)
    x_sig_train = np.concatenate([sigs[idx].x for idx in idx_sig_train])
    cond_sig_train = np.concatenate([sigs[idx].condition for idx in idx_sig_train])

    idx_sig_batch = np.random.randint(0, x_sig_train.shape[0], event_size)
    x_sig_train = x_sig_train[idx_sig_batch]
    cond_sig_train = cond_sig_train[idx_sig_batch]

    idx_bkg_batch = np.random.randint(0, bkg.x.shape[0], event_size)
    x_bkg_train = bkg.x[idx_bkg_batch]
    cond_bkg_train = bkg.condition[idx_bkg_batch]

    with tf.GradientTape() as tape:
        disc_sig_loss = bce(1.,disc_model(x_sig_train))
        disc_bkg_loss = bce(0.,disc_model(x_bkg_train))
        disc_loss = disc_sig_loss+disc_bkg_loss
    grads = tape.gradient(disc_loss,disc_model.trainable_weights)
    grads = [tf.clip_by_value(g,clip_value_min=-0.1,clip_value_max=0.1) for g in grads]
    disc_optimizer.apply_gradients(zip(grads,disc_model.trainable_weights))
    batch_trainer.add_loss("Disc loss",disc_loss)
    
    nf_sig_model.direction = -1
    nf_bkg_model.direction = -1
    with tf.GradientTape() as tape:
        p_sig = disc_model(x_sig_train)
        p_bkg = disc_model(x_bkg_train)
        nf_sig_sig_log_loss = nf_sig_model.batch_log_loss([x_sig_train,cond_sig_train])
        nf_sig_bkg_log_loss = nf_sig_model.batch_log_loss([x_bkg_train,cond_sig_train])
        nf_bkg_bkg_log_loss = nf_bkg_model.batch_log_loss(x_bkg_train)
        nf_bkg_sig_log_loss = nf_bkg_model.batch_log_loss(x_sig_train)
        batch_log_loss = p_sig * nf_sig_sig_log_loss + p_bkg * nf_sig_bkg_log_loss
        batch_log_loss += (1.-p_bkg) * nf_bkg_bkg_log_loss + (1-p_sig) * nf_bkg_sig_log_loss
        log_loss = tf.reduce_mean(batch_log_loss)
    grads = tape.gradient(log_loss,nf_sig_model.trainable_weights+nf_bkg_model.trainable_weights)
    nf_optimizer.apply_gradients(zip(grads,nf_sig_model.trainable_weights+nf_bkg_model.trainable_weights))
    batch_trainer.add_loss("NF loss",log_loss)

    batch_trainer.add_epoch()
    batch_trainer.print_loss(print_per_point)
    
    batch_trainer.make_history_plot(
            os.path.join(output_path,"loss.png"),
            log_scale=True,
            )

    if i_epoch % plot_per_point == 0 or i_epoch == n_epoch - 1:

        nf_sig_model.direction = 1
        nf_bkg_model.direction = 1
        
        sig_samples = nf_sig_model.distribution.sample(cond_sig_train.shape[0])
        x_sig_pred,_ = nf_sig_model.predict([sig_samples,cond_sig_train,])     

        bkg_samples = nf_bkg_model.distribution.sample(cond_bkg_train.shape[0])
        x_bkg_pred,_ = nf_bkg_model.predict(bkg_samples)     

        fig,ax = plt.subplots(plot_xdim,plot_ydim,figsize=(60,40))
        for ip,p in enumerate(plot_cfgs):
            px = int(ip % plot_xdim)
            py = int(ip / plot_xdim)
            ax[px,0].hist(x_sig_pred[:,p.index],bins=p.bins,label="flow",density=1.,histtype=p.histtype,range=p.range,linestyle=flow_linestyle,color='b')
            ax[px,0].hist(x_sig_train[:,p.index],bins=p.bins,label="true",density=1.,histtype=p.histtype,range=p.range,color='b')
            ax[px,0].set_title(p.name)
            ax[px,0].legend(loc='best')

            ax[px,1].hist(x_bkg_pred[:,p.index],bins=p.bins,label="flow",density=1.,histtype=p.histtype,range=p.range,linestyle=flow_linestyle,color='b')
            ax[px,1].hist(x_bkg_train[:,p.index],bins=p.bins,label="true",density=1.,histtype=p.histtype,range=p.range,color='b')
            ax[px,1].set_title(p.name)
            ax[px,1].legend(loc='best')

        fig.savefig(os.path.join(output_path,"plot"+str(i_epoch)+".png"))

    batch_trainer.save_weights(nf_sig_model,os.path.join(output_path,saved_model_name.replace(".h5","_sig.h5")),save_per_point=save_per_point,)
    batch_trainer.save_weights(nf_bkg_model,os.path.join(output_path,saved_model_name.replace(".h5","_bkg.h5")),save_per_point=save_per_point,)
    batch_trainer.save_weights(disc_model,os.path.join(output_path,saved_model_name.replace(".h5","_disc.h5")),save_per_point=save_per_point,)

batch_trainer.save_weights(nf_sig_model,os.path.join(output_path,saved_model_name.replace(".h5","_sig.h5")),)
batch_trainer.save_weights(nf_bkg_model,os.path.join(output_path,saved_model_name.replace(".h5","_bkg.h5")),)
batch_trainer.save_weights(disc_model,os.path.join(output_path,saved_model_name.replace(".h5","_disc.h5")),)
