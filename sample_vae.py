import os,uproot_methods
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from KerasSimpleTrainer.MiniBatchTrainer import MiniBatchTrainer

from process_data import make_flowgan_data
from Model.VAE import VAE

# ____________________________________________________________ ||
input_csv_path      = "data/train.npy"
batch_size          = 382547
saved_model_path    = "output/train_vae_201213_v1/saved_model.h5"
enc_number_list     = [32,128,256,512,128,64,10,] 
dec_number_list     = [32,128,256,512,256,128,64,] 
zmass               = 0.
output_path         = os.path.dirname(saved_model_path)

# ____________________________________________________________ ||
arr = np.load(input_csv_path)
x_train,condition_train,n_reco,n_param = make_flowgan_data(arr)

# ____________________________________________________________ ||
print("Loading weights")
model = VAE(n_reco,enc_number_list,dec_number_list)
_ = model.predict([x_train,condition_train,])
model.load_weights(saved_model_path)

print("Sampling latent space")
latent_vec = tf.random.normal(shape=(batch_size,int(enc_number_list[-1]/2)))
chosen_mass = np.broadcast_to(zmass,(batch_size,1))
gen_vars = model.decode_latent_vector(tf.concat([latent_vec,chosen_mass,],axis=1)).numpy()

print("Plot l1pt")
plt.hist(x_train[:,0],bins=100,label="mc",density=1.,histtype='step',)
plt.hist(gen_vars[:,0],bins=100,label="gan",density=1.,histtype='step',)
plt.legend(loc='best')
plt.savefig(os.path.join(output_path,"sample_l1pt.png"))
plt.clf()

print("Plot l1eta")
plt.hist(x_train[:,1],bins=100,label="mc",density=1.,histtype='step',)
plt.hist(gen_vars[:,1],bins=100,label="gan",density=1.,histtype='step',)
plt.legend(loc='best')
plt.savefig(os.path.join(output_path,"sample_l1eta.png"))
plt.clf()

print("Plot l2pt")
plt.hist(x_train[:,2],bins=100,label="mc",density=1.,histtype='step',)
plt.hist(gen_vars[:,2],bins=100,label="gan",density=1.,histtype='step',)
plt.legend(loc='best')
plt.savefig(os.path.join(output_path,"sample_l2pt.png"))
plt.clf()

print("Plot l2eta")
plt.hist(x_train[:,3],bins=100,label="mc",density=1.,histtype='step',)
plt.hist(gen_vars[:,3],bins=100,label="gan",density=1.,histtype='step',)
plt.legend(loc='best')
plt.savefig(os.path.join(output_path,"sample_l2eta.png"))
plt.clf()

print("Plot l21phi")
plt.hist(x_train[:,4],bins=100,label="mc",density=1.,histtype='step',)
plt.hist(gen_vars[:,4],bins=100,label="gan",density=1.,histtype='step',)
plt.legend(loc='best')
plt.savefig(os.path.join(output_path,"sample_l21phi.png"))
plt.clf()
