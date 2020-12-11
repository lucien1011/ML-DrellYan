import os
import numpy as np
import tensorflow as tf

from KerasSimpleTrainer.MiniBatchTrainer import MiniBatchTrainer

from Model.GAN import build_model

input_csv_path      = "/Users/lucien/CMS/PyLooper-DrellYan/output/2020_12_11_make_csv_DYToLL_cfg/train.npy"
energy_norm         = 100.
latent_dim          = 8
activation          = 'relu'
n_epoch             = 1000
batch_size          = 128
print_per_point     = 100
save_per_point      = 1000
disc_neuron_list    = [128,256,512,512,256,128]
gen_neuron_list     = [128,256,512,512,256,128]
output_path         = "output/train_gan_201211_v2/"
gen_model_name      = "saved_model.h5"

arr = np.load(input_csv_path)

x_train = arr[:,:-1]
x_train[:,0] /= energy_norm 
x_train[:,3] /= energy_norm 
condition_train = arr[:,-1] / energy_norm
condition_train = np.expand_dims(condition_train,axis=1)

n_reco = x_train.shape[1]
n_param = condition_train.shape[1]

discriminator,generator,x_input_layer,condition_input_layer = build_model(n_reco,n_param,disc_neuron_list,gen_neuron_list)

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
    
    with tf.GradientTape() as real_disc_tape:
        real_disc_output = discriminator([x_real_train,condition_real_train,])
        real_loss_disc = bce(y_real_train, real_disc_output)
    real_grads = real_disc_tape.gradient(real_loss_disc, discriminator.trainable_weights)
    disc_optimizer.apply_gradients(zip(real_grads, discriminator.trainable_weights))

    condition_fake_train = condition_real_train
    with tf.GradientTape() as fake_disc_tape:
        noise_train = np.random.uniform(-1,1,(batch_size,latent_dim))
        fake_disc_output = discriminator([generator([noise_train,condition_fake_train]),condition_fake_train,],)
        fake_loss_disc = bce(y_fake_train, fake_disc_output)
    fake_grads = fake_disc_tape.gradient(fake_loss_disc, discriminator.trainable_weights)
    disc_optimizer.apply_gradients(zip(fake_grads, discriminator.trainable_weights))
 
    condition_fake_train = condition_real_train
    with tf.GradientTape() as gen_tape:
        noise_train = np.random.uniform(-1,1,(batch_size,latent_dim))
        gen_out = discriminator([generator([noise_train,condition_fake_train]),condition_fake_train,],)
        loss_gen = bce(y_real_train, gen_out)
    gen_grads = gen_tape.gradient(loss_gen,generator.trainable_weights)
    gen_optimizer.apply_gradients(zip(gen_grads, generator.trainable_weights))

    batch_trainer.add_loss("real disc loss",real_loss_disc)
    batch_trainer.add_loss("fake disc loss",fake_loss_disc)
    batch_trainer.add_loss("gen loss",loss_gen)

    batch_trainer.add_epoch()
    batch_trainer.print_loss(print_per_point)
    
    batch_trainer.make_history_plot(
            os.path.join(output_path,"loss.png"),
            log_scale=True,
            )
    batch_trainer.save_weights(generator,os.path.join(output_path,gen_model_name),n_per_point=save_per_point,)
batch_trainer.save_weights(generator,os.path.join(output_path,gen_model_name),)
