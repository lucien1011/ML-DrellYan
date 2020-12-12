import os
import numpy as np
import tensorflow as tf

from KerasSimpleTrainer.MiniBatchTrainer import MiniBatchTrainer

from Model.NormFlow import NormFlow,Discriminator

# ____________________________________________________________ ||
input_csv_path      = "/Users/lucien/CMS/PyLooper-DrellYan/output/2020_12_11_make_csv_DYToLL_cfg/train.npy"
energy_norm         = 100.
condition_norm      = 10.
n_epoch             = 1000
batch_size          = 128
print_per_point     = 100
save_per_point      = 1000
n_flow              = 10
latent_dim          = 6 
output_path         = "output/train_flowgan_201212_v1/"
gen_model_name      = "saved_model.h5"

# ____________________________________________________________ ||
arr = np.load(input_csv_path)
x_train = arr[:,:-1]
x_train[:,0] /= energy_norm 
x_train[:,3] /= energy_norm 
condition_train = (arr[:,-1] - 90.) / condition_norm
condition_train = np.expand_dims(condition_train,axis=1)

n_reco = x_train.shape[1]
n_param = condition_train.shape[1]

# ____________________________________________________________ ||
gen = NormFlow(n_reco,n_flow)
disc = Discriminator([16,16,16])

# ____________________________________________________________ ||
bce = tf.keras.losses.BinaryCrossentropy()
disc_optimizer = tf.keras.optimizers.Adam()
gen_optimizer = tf.keras.optimizers.Adam()
batch_trainer = MiniBatchTrainer()
for i_epoch in range(n_epoch):
    
    y_real_train = np.ones((batch_size,1))
    y_fake_train = np.zeros((batch_size,1))

    idx_train = np.random.randint(0, x_train.shape[0], batch_size)
    x_real_train = x_train[idx_train]
    condition_real_train = condition_train[idx_train]
    
    with tf.GradientTape() as tape:
        out = disc([x_real_train,condition_real_train])
        disc_real_loss = bce(y_real_train,out)
    grads = tape.gradient(disc_real_loss, disc.trainable_weights)
    disc_optimizer.apply_gradients(zip(grads, disc.trainable_weights))
    
    with tf.GradientTape() as tape:
        noise = tf.random.normal((batch_size,latent_dim))
        x_fake_train = gen(tf.concat([noise,condition_real_train],axis=1))
        out = disc([x_fake_train,condition_real_train])
        disc_fake_loss = bce(y_fake_train,out)
    grads = tape.gradient(disc_fake_loss, disc.trainable_weights)
    disc_optimizer.apply_gradients(zip(grads, disc.trainable_weights))  
    
    with tf.GradientTape() as tape:
        noise = tf.random.normal((batch_size,latent_dim))
        x_fake_train = gen(tf.concat([noise,condition_real_train],axis=1))
        out = disc([x_fake_train,condition_real_train])
        gen_loss = bce(y_real_train,out)
    grads = tape.gradient(gen_loss, gen.trainable_weights)
    disc_optimizer.apply_gradients(zip(grads, gen.trainable_weights))
    
    batch_trainer.add_loss("disc real loss",disc_real_loss)
    batch_trainer.add_loss("disc fake loss",disc_fake_loss)
    batch_trainer.add_loss("gen loss",gen_loss)

    batch_trainer.add_epoch()
    batch_trainer.print_loss(print_per_point)
    
    batch_trainer.make_history_plot(
            os.path.join(output_path,"loss.png"),
            log_scale=True,
            )
    batch_trainer.save_weights(gen,os.path.join(output_path,gen_model_name),n_per_point=save_per_point,)
batch_trainer.save_weights(gen,os.path.join(output_path,gen_model_name),)
