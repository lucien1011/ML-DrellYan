import os,uproot_methods,math
import numpy as np

def make_flowgan_data(arr):
    energy_norm = 20.
    condition_norm = 10.
    
    arr_lep1_vec = uproot_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(arr[:,0],arr[:,1],arr[:,2],0.)
    arr_lep2_vec = uproot_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(arr[:,3],arr[:,4],arr[:,5],0.)
    arr_lep21_vec = arr_lep2_vec - arr_lep1_vec

    arr[:,0] = arr[:,0] / energy_norm - 2.
    arr[:,1] = arr[:,1] / 2.4
    arr[:,3] = arr[:,3] / energy_norm - 2.
    arr[:,4] = arr[:,4] / 2.4
        
    x_train = np.concatenate(
            [
                np.expand_dims(arr[:,0],axis=1),
                np.expand_dims(arr[:,1],axis=1),
                np.expand_dims(arr[:,3],axis=1),
                np.expand_dims(arr[:,4],axis=1),
                np.expand_dims(arr_lep21_vec.phi,axis=1),
            ],
            axis=1,
            )
    condition_train = (arr[:,-1] - 90.) / condition_norm
    condition_train = np.expand_dims(condition_train,axis=1)
    
    n_reco = x_train.shape[1]
    n_param = condition_train.shape[1]

    return x_train,condition_train,n_reco,n_param
