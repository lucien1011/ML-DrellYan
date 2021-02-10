import os,pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from KerasSimpleTrainer.MiniBatchTrainer import MiniBatchTrainer

from process_data import preprocess_conditional_flow_data_cww,simuate_conditional_flow_data_mass
from Model.ConditionalRealNVP import ConditionalRealNVP
from Utils.ObjDict import ObjDict
from Utils.mkdir_p import mkdir_p

# ____________________________________________________________ ||
#tf.random.set_seed(123456)

# ____________________________________________________________ ||
input_csv_path      = "data/train_cww.npy"
n_epoch             = 4000
batch_size          = 1
event_size          = 1000
print_per_point     = 50
plot_per_point      = 50
save_per_point      = 50
output_path         = "output/train_condrealnvp_cww_210210_v1/"
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
plot_xdim           = 2
plot_ydim           = 2

# ____________________________________________________________ ||
arr = np.load(input_csv_path)
arr_list,_ = preprocess_conditional_flow_data_cww(arr)

ndim = 3
ncond = 1

# ____________________________________________________________ ||
nf_model = ConditionalRealNVP(num_coupling_layers=10,ndim=ndim,ncond=ncond)  

# ____________________________________________________________ ||
batch_trainer = MiniBatchTrainer()
nf_optimizer = tf.keras.optimizers.Adam()
mkdir_p(output_path)
for i_epoch in range(n_epoch):
    
    idx_train = np.random.randint(0, len(arr_list), batch_size)
    x_train = np.concatenate([arr_list[idx].x for idx in idx_train])
    condition_train = np.concatenate([arr_list[idx].condition for idx in idx_train])

    idx_batch = np.random.randint(0, x_train.shape[0], event_size)
    x_train = x_train[idx_batch]
    condition_train = condition_train[idx_batch]

    nf_model.direction = -1
    with tf.GradientTape() as tape:
        nf_loss = nf_model.log_loss([x_train,condition_train])
    grads = tape.gradient(nf_loss, nf_model.trainable_weights)
    grads = [tf.clip_by_value(g,clip_value_min=-0.1,clip_value_max=0.1) for g in grads]
    nf_optimizer.apply_gradients(zip(grads, nf_model.trainable_weights))
    
    batch_trainer.add_loss("NF loss",nf_loss)

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

        fig,ax = plt.subplots(plot_xdim,plot_ydim,figsize=(60,40))
        for ip,p in enumerate(plot_cfgs):
            px = int(ip % plot_xdim)
            py = int(ip / plot_xdim)
            ax[px,py].hist(x_pred[:,p.index],bins=p.bins,label="flow",density=1.,histtype=p.histtype,range=p.range,linestyle=flow_linestyle,color='b')
            ax[px,py].hist(x_train[:,p.index],bins=p.bins,label="true",density=1.,histtype=p.histtype,range=p.range,color='b')
            ax[px,py].set_title(p.name)
            ax[px,py].legend(loc='best')
        fig.savefig(os.path.join(output_path,"plot"+str(i_epoch)+".png"))

    batch_trainer.save_weights(nf_model,os.path.join(output_path,saved_model_name),save_per_point=save_per_point,)

batch_trainer.save_weights(nf_model,os.path.join(output_path,saved_model_name),)
