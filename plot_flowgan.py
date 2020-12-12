import os,uproot_methods
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from KerasSimpleTrainer.MiniBatchTrainer import MiniBatchTrainer

from process_data import make_flowgan_data
from Model.NormFlow import NormFlow,Discriminator

# ____________________________________________________________ ||
input_csv_path      = "data/train.npy"
batch_size          = 382547
n_flow              = 10
latent_dim          = 5
saved_model_path    = "output/train_flowgan_201212_v1/saved_model.h5"

output_path         = os.path.dirname(saved_model_path)

# ____________________________________________________________ ||
arr = np.load(input_csv_path)
x_train,condition_train,n_reco,n_param = make_flowgan_data(arr)

# ____________________________________________________________ ||
gen = NormFlow(n_reco,n_flow)
noise = tf.random.normal((condition_train.shape[0],latent_dim))
input = tf.concat([noise,condition_train],axis=1)
_ = gen.predict(input)
gen.load_weights(saved_model_path)

gen_vars = gen.predict(input)

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
