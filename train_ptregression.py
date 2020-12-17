import os,uproot_methods,math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from KerasSimpleTrainer.MiniBatchTrainer import MiniBatchTrainer
from process_data import make_ptregression_data
from Model.BNN import Bayesian

tf.random.set_seed(1)

# ____________________________________________________________ ||
input_csv_path      = "data/train.npy"
output_path         = "output/plot_ptregression_201217_v1/"
n_epoch             = 5000
batch_size          = 512
plot_per_epoch      = 100
print_per_epoch     = 50
plot_range          = [0.,5.]
nbin                = 50
ndivide             = 10000

# ____________________________________________________________ ||
arr = np.load(input_csv_path)
x_orig,x_smear,mll_smear = make_ptregression_data(arr)
x_orig_train = x_orig[ndivide:]
x_orig_test = x_orig[:ndivide]

model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(8,activation='relu',),
            #Bayesian(8,activation=tf.keras.activations.relu),
            Bayesian(1,activation=lambda x: x),
            #tf.keras.layers.Dense(8,activation='relu',),
            #tf.keras.layers.Dense(8,activation='relu',),
            #tf.keras.layers.Dense(8,activation='relu',),
            #tf.keras.layers.Dense(2,activation='linear',),
            #tf.keras.layers.Dense(4,activation='linear',),
            #NormalSampling(2),
        ]
        )

optimizer = tf.keras.optimizers.Adam()
batch_trainer = MiniBatchTrainer()
for i_epoch in range(n_epoch):
    with tf.GradientTape() as tape:
        idx_train = np.random.randint(0, x_orig_train.shape[0], batch_size)
        x_train = x_orig_train[idx_train]

        out1 = model(x_train[:,:3])
        epsilon1 = K.random_normal(shape=(batch_size,))
        lep1_pt_pred = tf.math.multiply(x_train[:,0],tf.math.exp(-0.5 * out1[:,0]))
        #lep1_pt_pred = tf.math.multiply(x_train[:,0],out1[:,0])

        out2 = model(x_train[:,3:])
        epsilon2 = K.random_normal(shape=(batch_size,))
        lep2_pt_pred = tf.math.multiply(x_train[:,3],tf.math.exp(-0.5 * out2[:,0]))
        #lep2_pt_pred = tf.math.multiply(x_train[:,3],out2[:,0])
       
        mll_pred = 2 * tf.math.multiply(
                tf.math.multiply(lep1_pt_pred,lep2_pt_pred),
                np.cosh(x_train[:,1]-x_train[:,4]) - np.cos(x_train[:,2] - x_train[:,5]), 
                )
        
        print("MSE: ",tf.reduce_sum((mll_pred - mll_smear[idx_train])**2))
        print("Reg: ",tf.reduce_sum(model.losses))
        train_loss = tf.reduce_sum((mll_pred - mll_smear[idx_train])**2) / batch_size + tf.reduce_sum(model.losses)

    batch_trainer.add_loss("train loss",train_loss)

    grads = tape.gradient(train_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    out1 = model(x_orig_test[:,:3])
    epsilon1 = K.random_normal(shape=(batch_size,))
    lep1_pt_pred = tf.math.multiply(x_orig_test[:,0],tf.math.exp(-0.5 * out1[:,0]))
    #lep1_pt_pred = tf.math.multiply(x_orig_test[:,0],out1[:,0])

    out2 = model(x_orig_test[:,3:])
    epsilon2 = K.random_normal(shape=(batch_size,))
    lep2_pt_pred = tf.math.multiply(x_orig_test[:,3],tf.math.exp(-0.5 * out2[:,0]))
    #lep2_pt_pred = tf.math.multiply(x_orig_test[:,3],out2[:,0])
    
    mll_pred = 2 * tf.math.multiply(
            tf.math.multiply(lep1_pt_pred,lep2_pt_pred),
            np.cosh(x_orig_test[:,1]-x_orig_test[:,4]) - np.cos(x_orig_test[:,2] - x_orig_test[:,5]), 
            )
    test_loss = tf.reduce_sum((mll_pred - mll_smear[:ndivide])**2) / x_orig_test.shape[0]

    batch_trainer.add_loss("test loss",test_loss)
    batch_trainer.print_loss(print_per_epoch)
    batch_trainer.make_history_plot(
            os.path.join(output_path,"loss.png"),
            log_scale=True,
            )
    
    if i_epoch % plot_per_epoch == 0:
 
        plt.hist(mll_pred.numpy(),bins=nbin,label="pred",density=1.,histtype='step',range=plot_range,)
        plt.hist(mll_smear,bins=nbin,label="true",density=1.,histtype='step',range=plot_range,)
        plt.title("Test loss: "+str(test_loss))
        plt.legend(loc='best')
        plt.savefig(os.path.join(output_path,"mll_"+str(i_epoch)+".png"))
        plt.clf()
