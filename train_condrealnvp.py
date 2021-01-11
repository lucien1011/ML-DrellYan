import os,pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from KerasSimpleTrainer.MiniBatchTrainer import MiniBatchTrainer

from process_data import simulate_conditional_flow_data_ptscale
from Model.ConditionalRealNVP import ConditionalRealNVP
from Utils.ObjDict import ObjDict
from Utils.mkdir_p import mkdir_p

# ____________________________________________________________ ||
input_csv_path      = "data/train.npy"
n_epoch             = 2000
batch_size          = 4
event_size          = 4096
print_per_point     = 5
plot_per_point      = 5
save_per_point      = 1000
output_path         = "output/train_condrealnvp_210111_v1/"
saved_model_name    = "saved_model.h5"

plot_cfgs           = [
        ObjDict(name="pt1",bins=50,range=[-5.,5.],index=0,histtype='step',),
        ObjDict(name="pt2",bins=50,range=[-5.,5.],index=2,histtype='step',),
        ObjDict(name="mll",bins=200,range=[-20.,20.],index=2,histtype='step',),
        ]
flow_linestyle      = ':'
true_linestyle      = '-'

# ____________________________________________________________ ||
arr = np.load(input_csv_path)
condition = (arr[:,-1] - 90.) 
arr = arr[np.squeeze(np.abs(condition) < 1)]

ndim = 3
ncond = 2
pt1_mean = np.mean(arr[:,0])
pt2_mean = np.mean(arr[:,3])

# ____________________________________________________________ ||
nf_model = ConditionalRealNVP(num_coupling_layers=5,dim=ndim,ncond=ncond)  

# ____________________________________________________________ ||
batch_trainer = MiniBatchTrainer()
nf_optimizer = tf.keras.optimizers.Adam()
mkdir_p(output_path)
for i_epoch in range(n_epoch):
    
    idx_train = np.random.randint(0, arr.shape[0], event_size)
    x_train,condition_train = simulate_conditional_flow_data_ptscale(arr[idx_train],pt1_mean=pt1_mean,pt2_mean=pt2_mean,batch_size=batch_size,event_size=event_size,)

    nf_model.direction = -1
    with tf.GradientTape() as tape:
        nf_loss = nf_model.log_loss([x_train,condition_train])
    grads = tape.gradient(nf_loss, nf_model.trainable_weights)
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

        fig,ax = plt.subplots(ndim,1,figsize=(15,50))
        for ip,p in enumerate(plot_cfgs):
            ax[ip].hist(x_pred[:,ip],bins=p.bins,label="flow",density=1.,histtype=p.histtype,range=p.range,linestyle=flow_linestyle,color='b')
            ax[ip].hist(x_train[:,ip],bins=p.bins,label="true",density=1.,histtype=p.histtype,range=p.range,color='b')

            ax[ip].legend(loc='best')
        fig.savefig(os.path.join(output_path,"plot"+str(i_epoch)+".png"))
