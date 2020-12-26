# ____________________________________________________________ ||
# Author: Kin Ho Lo
# The idea is to test how sensitive the mll spectrum is to the 
# embedded energy scale and uncertainty parameter. Assume two variable 
# (mu,scale) for the energy scale and uncertainty.
# Starting from a DY sample, the lepton pT is smeared given mu and scale,
# and a mll spectrum is recalculated. The training process is to find the 
# correct mu and scale.
# The loss to be minimized is a standard -log(Gaussian).
# ____________________________________________________________ ||
# Result:
# With mu as 1.1 and scale as 0.1, the best performance occurs at 
# epoch ~ 150.
# 3 points to note:
# 1) a gaussian pdf is not the best pdf to describe 
# mll shape, therefore one does not actually observe a perfect fit to the 
# smeared distribution. One can see this in the jupyter notebook train_mle_mll.ipynb.
# 2) Can consider code up a early stopping condition.
# 3) Can test more mu and scale parameter values.
# ____________________________________________________________ ||
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from KerasSimpleTrainer.MiniBatchTrainer import MiniBatchTrainer

# ____________________________________________________________ ||
tf.random.set_seed(1)

# ____________________________________________________________ ||
# Require function
def make_ptregression_data(input_arr,mu=1.10,scale=0.1):
    energy_norm = 50.
    angle_norm = 1.
    
    condition = (input_arr[:,-1] - 90.) 
    arr = input_arr[np.squeeze(np.abs(condition) < 1)]

    arr[:,0] = arr[:,0] / energy_norm
    arr[:,1] = arr[:,1] / angle_norm
    arr[:,2] = arr[:,2] / angle_norm
    arr[:,3] = arr[:,3] / energy_norm
    arr[:,4] = arr[:,4] / angle_norm
    arr[:,5] = arr[:,5] / angle_norm
        
    x_orig = np.concatenate(
            [
                np.expand_dims(arr[:,0],axis=1),
                np.expand_dims(arr[:,1],axis=1),
                np.expand_dims(arr[:,2],axis=1),
                np.expand_dims(arr[:,3],axis=1),
                np.expand_dims(arr[:,4],axis=1),
                np.expand_dims(arr[:,5],axis=1),
            ],
            axis=1,
            )
    
    eps1 = K.random_normal(shape=(x_orig.shape[0],1))
    sf1 = mu + scale * eps1

    eps2 = K.random_normal(shape=(x_orig.shape[0],1))
    sf2 = mu + scale * eps2

    smear_pt1 = tf.math.multiply(np.expand_dims(arr[:,0],axis=1),sf1) 
    eta1 = np.expand_dims(arr[:,1],axis=1)
    phi1 = np.expand_dims(arr[:,2],axis=1)
    smear_pt2 = tf.math.multiply(np.expand_dims(arr[:,3],axis=1),sf2) 
    eta2 = np.expand_dims(arr[:,4],axis=1)
    phi2 = np.expand_dims(arr[:,5],axis=1)

    x_smear = np.concatenate(
            [
                smear_pt1,
                np.expand_dims(arr[:,1],axis=1),
                np.expand_dims(arr[:,2],axis=1),
                smear_pt2,
                np.expand_dims(arr[:,4],axis=1),
                np.expand_dims(arr[:,5],axis=1),
            ],
            axis=1,
            )
    
    smear_mll = 2 * np.multiply(
            np.multiply(smear_pt1,smear_pt2),
            np.cosh(eta1-eta2) - np.cos(phi1-phi2), 
            )

    return x_orig,x_smear,smear_mll

# ____________________________________________________________ ||
input_csv_path      = "data/train.npy"
output_path         = "output/train_mle_ptreg_201226_v1/"
n_epoch             = 500
batch_size          = 10000
plot_per_epoch      = 10
print_per_epoch     = 50
plot_range          = [0.,5.]
nbin                = 50

# ____________________________________________________________ ||
mu = tf.Variable(1.0,dtype=tf.float32)
scale = tf.Variable(0.05,dtype=tf.float32)

# ____________________________________________________________ ||
arr = np.load(input_csv_path)
x_orig,x_smear,mll_smear = make_ptregression_data(arr)
x_orig_train,x_orig_test,mll_train,mll_test = train_test_split(x_orig,mll_smear,test_size=0.1,random_state=42)

# ____________________________________________________________ ||
optimizer = tf.keras.optimizers.Adam()
batch_trainer = MiniBatchTrainer()

for i_epoch in range(n_epoch):
    with tf.GradientTape() as tape:
        x_train = x_orig_train

        eps1 = K.random_normal(shape=(x_train.shape[0],))
        sf1 = mu + scale * eps1
        lep1_pt_pred = tf.math.multiply(x_train[:,0],sf1)

        eps2 = K.random_normal(shape=(x_train.shape[0],))
        sf2 = mu + scale * eps2
        lep2_pt_pred = tf.math.multiply(x_train[:,3],sf2)
       
        mll_true = mll_train.astype(np.float32)
        mll_pred = 2 * tf.math.multiply(
                tf.math.multiply(lep1_pt_pred,lep2_pt_pred),
                np.cosh(x_train[:,1]-x_train[:,4]) - np.cos(x_train[:,2] - x_train[:,5]), 
                )
        
        mean_est = tf.math.reduce_mean(mll_pred)
        beta_est = 1./tf.math.reduce_std(mll_pred)
        train_loss = -tf.math.log(beta_est) + 0.5 * tf.reduce_mean(tf.math.square(beta_est*(mll_true-mean_est)))
    
    batch_trainer.add_loss("train loss",train_loss)

    grads = tape.gradient(train_loss,[mu,scale,])
    optimizer.apply_gradients(zip(grads,[mu,scale,]))

    eps1 = K.random_normal(shape=(x_orig_test.shape[0],))
    sf1 = mu + scale * eps1
    lep1_pt_pred = tf.math.multiply(x_orig_test[:,0],sf1)

    eps2 = K.random_normal(shape=(x_orig_test.shape[0],))
    sf2 = mu + scale * eps2
    lep2_pt_pred = tf.math.multiply(x_orig_test[:,3],sf2)
    
    mll_true = mll_test.astype(np.float32)
    mll_pred = 2 * tf.math.multiply(
            tf.math.multiply(lep1_pt_pred,lep2_pt_pred),
            np.cosh(x_orig_test[:,1]-x_orig_test[:,4]) - np.cos(x_orig_test[:,2] - x_orig_test[:,5]), 
            )

    mean_est = tf.math.reduce_mean(mll_pred)
    beta_est = 1./tf.math.reduce_std(mll_pred)
    test_loss = -tf.math.log(beta_est) + 0.5 * tf.reduce_mean(tf.math.square(beta_est*(mll_true-mean_est)))

    batch_trainer.add_loss("test loss",test_loss)
    batch_trainer.add_epoch()
    batch_trainer.print_loss(print_per_epoch)
    batch_trainer.make_history_plot(
            os.path.join(output_path,"loss.png"),
            log_scale=True,
            )
    
    if i_epoch % plot_per_epoch == 0:

        print("Fitted parameters: ",mu,scale)
 
        plt.hist(mll_pred.numpy(),bins=nbin,label="pred",density=1.,histtype='step',range=plot_range,)
        plt.hist(mll_true,bins=nbin,label="true",density=1.,histtype='step',range=plot_range,)
        plt.title("Test loss: "+str(test_loss))
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_path,"mll_"+str(i_epoch)+".png"))
        plt.clf()
