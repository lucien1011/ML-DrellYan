import os,uproot_methods
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from Model.GAN import build_model

saved_model_path    = "output/train_mlp_201211_v1/saved_model.h5"
input_csv_path      = "/Users/lucien/CMS/PyLooper-DrellYan/output/2020_12_11_make_csv_DYToLL_cfg/train.npy"
energy_norm         = 100.
latent_dim          = 16
disc_neuron_list    = [128,256,512,512,256,128]
gen_neuron_list     = [128,256,512,512,256,128]
n_events            = 100000
zmass               = 0.90

output_path = os.path.dirname(saved_model_path)
arr = np.load(input_csv_path)

x_train = arr[:,:-1]
x_train[:,0] /= energy_norm 
x_train[:,3] /= energy_norm 
condition_train = arr[:,-1] / energy_norm
condition_train = np.expand_dims(condition_train,axis=1)

n_reco = x_train.shape[1]
n_param = condition_train.shape[1]

discriminator,generator,x_input_layer,condition_input_layer = build_model(n_reco,n_param,disc_neuron_list,gen_neuron_list)

condition_real_plot = np.broadcast_to(zmass,(n_events,1)) 
gen_vars = generator.predict(condition_real_plot)
arr_lep1_vec = uproot_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(gen_vars[:,0],gen_vars[:,1],gen_vars[:,2],0.)
arr_lep2_vec = uproot_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(gen_vars[:,3],gen_vars[:,4],gen_vars[:,5],0.)
arr_lep12_vec = arr_lep1_vec + arr_lep2_vec
plt.hist(arr_lep12_vec.mass,bins=100,)
plt.savefig(os.path.join(output_path,"mll.png"))
