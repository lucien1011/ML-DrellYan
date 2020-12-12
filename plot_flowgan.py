import os,uproot_methods
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from KerasSimpleTrainer.MiniBatchTrainer import MiniBatchTrainer

from Model.NormFlow import NormFlow,Discriminator

# ____________________________________________________________ ||
input_csv_path      = "/Users/lucien/CMS/PyLooper-DrellYan/output/2020_12_11_make_csv_DYToLL_cfg/train.npy"
energy_norm         = 100.
condition_norm      = 10.
batch_size          = 3874
n_flow              = 10
latent_dim          = 6 
zmass               = (95. - 90.) / condition_norm
saved_model_path    = "output/train_flowgan_201212_v1/saved_model.h5"

output_path         = os.path.dirname(saved_model_path)

# ____________________________________________________________ ||
arr = np.load(input_csv_path)
x_train = arr[:,:-1]
x_train[:,0] /= energy_norm 
x_train[:,3] /= energy_norm 
condition_train = (arr[:,-1] - 90.) / condition_norm
condition_train = np.expand_dims(condition_train,axis=1)

n_reco = x_train.shape[1]
n_param = condition_train.shape[1]

noise = tf.random.normal((batch_size,latent_dim))
condition_real_plot = np.broadcast_to(zmass,(batch_size,1))
#input = tf.concat([noise,condition_real_plot],axis=1)
input = tf.concat([noise,condition_train],axis=1)

# ____________________________________________________________ ||
gen = NormFlow(n_reco,n_flow)
_ = gen.predict(input)
gen.load_weights(saved_model_path)

gen_vars = gen.predict(input)
arr_lep1_vec = uproot_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(gen_vars[:,0],gen_vars[:,1],gen_vars[:,2],0.)
arr_lep2_vec = uproot_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(gen_vars[:,3],gen_vars[:,4],gen_vars[:,5],0.)
arr_lep12_vec = arr_lep1_vec + arr_lep2_vec

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
plt.savefig(os.path.join(output_path,"l1phi.png"))
plt.clf()

plt.hist(x_train[:,3],bins=100,label="mc",density=1.,histtype='step',)
plt.hist(gen_vars[:,3],bins=100,label="gan",density=1.,histtype='step',)
plt.legend(loc='best')
plt.savefig(os.path.join(output_path,"l2pt.png"))
plt.clf()

plt.hist(x_train[:,4],bins=100,label="mc",density=1.,histtype='step',)
plt.hist(gen_vars[:,4],bins=100,label="gan",density=1.,histtype='step',)
plt.legend(loc='best')
plt.savefig(os.path.join(output_path,"l2eta.png"))
plt.clf()

plt.hist(x_train[:,5],bins=100,label="mc",density=1.,histtype='step',)
plt.hist(gen_vars[:,5],bins=100,label="gan",density=1.,histtype='step',)
plt.legend(loc='best')
plt.savefig(os.path.join(output_path,"l2phi.png"))
plt.clf()

plt.hist(arr_lep12_vec.mass,bins=100,histtype='step',)
plt.savefig(os.path.join(output_path,"mll.png"))
