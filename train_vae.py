import os,uproot_methods,math
import numpy as np
import tensorflow as tf

from KerasSimpleTrainer.MiniBatchTrainer import MiniBatchTrainer

from process_data import make_flowgan_data
from Model.VAE import VAE

# ____________________________________________________________ ||
input_csv_path      = "data/train.npy"
n_epoch             = 500
batch_size          = 128
print_per_point     = 100
save_per_point      = 1000
enc_number_list     = [32,128,256,512,128,64,10,] 
dec_number_list     = [32,128,256,512,256,128,64,] 
latent_dim          = 5
output_path         = "output/train_vae_201213_v1/"
gen_model_name      = "saved_model.h5"

# ____________________________________________________________ ||
arr = np.load(input_csv_path)
x_train,condition_train,n_reco,n_param = make_flowgan_data(arr)

# ____________________________________________________________ ||
model = VAE(n_reco,enc_number_list,dec_number_list)

# ____________________________________________________________ ||
mse = tf.keras.losses.MSE
disc_optimizer = tf.keras.optimizers.Adam()
gen_optimizer = tf.keras.optimizers.Adam()
batch_trainer = MiniBatchTrainer()
for i_epoch in range(n_epoch):
    
    idx_train = np.random.randint(0, x_train.shape[0], batch_size)
        
    with tf.GradientTape() as tape:
        out = model([x_train[idx_train],condition_train[idx_train],])
        loss = tf.reduce_mean(mse(x_train[idx_train],out)+model.losses)
    grads = tape.gradient(loss, model.trainable_weights)
    disc_optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    batch_trainer.add_loss("loss",loss)

    batch_trainer.add_epoch()
    batch_trainer.print_loss(print_per_point)
    
    batch_trainer.make_history_plot(
            os.path.join(output_path,"loss.png"),
            log_scale=True,
            )
    batch_trainer.save_weights(model,os.path.join(output_path,gen_model_name),n_per_point=save_per_point,)
batch_trainer.save_weights(model,os.path.join(output_path,gen_model_name),)
