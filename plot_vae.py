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

output_path         = os.path.dirname(saved_model_path)

# ____________________________________________________________ ||
arr = np.load(input_csv_path)
x_train,condition_train,n_reco,n_param = make_flowgan_data(arr)

# ____________________________________________________________ ||
model = VAE(n_reco,enc_number_list,dec_number_list)
_ = model.predict([x_train,condition_train,])
model.load_weights(saved_model_path)

out = model.predict_latent_vector([x_train,condition_train,])
plt.hist(out[:,0].numpy(),bins=100,label="latent 1",histtype='step',density=1.)
plt.hist(out[:,1].numpy(),bins=100,label="latent 2",histtype='step',density=1.)
plt.hist(out[:,2].numpy(),bins=100,label="latent 3",histtype='step',density=1.)
plt.savefig(os.path.join(output_path,"latent.png"))
plt.clf()

gen_vars = model.predict([x_train,condition_train,])

plt.hist(x_train[:,0],bins=100,label="mc",density=1.,histtype='step',)
plt.hist(gen_vars[:,0],bins=100,label="gan",density=1.,histtype='step',)
plt.legend(loc='best')
plt.savefig(os.path.join(output_path,"l1pt.png"))
plt.clf()

plt.hist(x_train[:,1],bins=100,label="mc",density=1.,histtype='step',)
plt.hist(gen_vars[:,1],bins=100,label="gan",density=1.,histtype='step',)
plt.legend(loc='best')
plt.savefig(os.path.join(output_path,"l1eta.png"))
plt.clf()

plt.hist(x_train[:,2],bins=100,label="mc",density=1.,histtype='step',)
plt.hist(gen_vars[:,2],bins=100,label="gan",density=1.,histtype='step',)
plt.legend(loc='best')
plt.savefig(os.path.join(output_path,"l2pt.png"))
plt.clf()

plt.hist(x_train[:,3],bins=100,label="mc",density=1.,histtype='step',)
plt.hist(gen_vars[:,3],bins=100,label="gan",density=1.,histtype='step',)
plt.legend(loc='best')
plt.savefig(os.path.join(output_path,"l2eta.png"))
plt.clf()

plt.hist(x_train[:,4],bins=100,label="mc",density=1.,histtype='step',)
plt.hist(gen_vars[:,4],bins=100,label="gan",density=1.,histtype='step',)
plt.legend(loc='best')
plt.savefig(os.path.join(output_path,"l21phi.png"))
plt.clf()
