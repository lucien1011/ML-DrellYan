import os,uproot_methods,math
import numpy as np

def normalizePhi(res):
    '''Computes delta phi, handling periodic limit conditions.'''
    while res > math.pi:
        res -= 2*math.pi
    while res < -math.pi:
        res += 2*math.pi
    return res

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

def make_pxpypze_data(arr):
    energy_norm = 20.
    condition_norm = 10.
    
    arr_lep1_vec = uproot_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(arr[:,0],arr[:,1],arr[:,2],0.)
    arr_lep2_vec = uproot_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(arr[:,3],arr[:,4],arr[:,5],0.)
    #phi_func = np.vectorize(normalizePhi)
    #arr_dphi = phi_func(arr_lep1_vec.phi - arr_lep2_vec.phi)
    arr_dphi = np.abs(arr_lep1_vec.phi - arr_lep2_vec.phi)

    x_train = np.concatenate(
            [
                np.expand_dims(arr_lep1_vec.pt,axis=1) / energy_norm,
                np.expand_dims(arr_dphi,axis=1),
                np.expand_dims(arr_lep1_vec.z,axis=1) / energy_norm,
                np.expand_dims(arr_lep1_vec.t,axis=1) / energy_norm,
                np.expand_dims(arr_lep2_vec.pt,axis=1) / energy_norm,
                np.expand_dims(arr_lep2_vec.z,axis=1) / energy_norm,
                np.expand_dims(arr_lep2_vec.t,axis=1) / energy_norm,
            ],
            axis=1,
            )
    condition_train = (arr[:,-1] - 90.) / condition_norm
    condition_train = np.expand_dims(condition_train,axis=1)
    
    n_reco = x_train.shape[1]
    n_param = condition_train.shape[1]

    return x_train,condition_train,n_reco,n_param

def make_ptregression_data(input_arr):
    energy_norm = 20.
    angle_norm = 2.4
    
    condition = (input_arr[:,-1] - 90.) 
    arr = input_arr[np.squeeze(np.abs(condition) < 1)]

    arr_lep1_vec = uproot_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(arr[:,0],arr[:,1],arr[:,2],0.)
    arr_lep2_vec = uproot_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(arr[:,3],arr[:,4],arr[:,5],0.)
    arr_lep21_vec = arr_lep2_vec - arr_lep1_vec

    arr[:,0] = arr[:,0] / energy_norm
    arr[:,1] = arr[:,1] / angle_norm
    arr[:,2] = arr[:,2] / angle_norm
    arr[:,3] = arr[:,3] / energy_norm
    arr[:,4] = arr[:,4] / angle_norm
    arr[:,5] = arr[:,5] / angle_norm
        
    x_orig = np.concatenate(
            [
                np.expand_dims(arr[:,0],axis=1),
                np.expand_dims(arr[:,1],axis=1),
                np.expand_dims(arr[:,2],axis=1),
                np.expand_dims(arr[:,3],axis=1),
                np.expand_dims(arr[:,4],axis=1),
                np.expand_dims(arr[:,5],axis=1),
            ],
            axis=1,
            )

    smear_func = np.random.normal
    mean = 1.02
    sigma = 0.05
    x_smear = np.concatenate(
            [
                np.expand_dims(arr[:,0],axis=1) * smear_func(mean,sigma,(x_orig.shape[0],1)),
                np.expand_dims(arr[:,1],axis=1),
                np.expand_dims(arr[:,2],axis=1),
                np.expand_dims(arr[:,3],axis=1) * smear_func(mean,sigma,(x_orig.shape[0],1)),
                np.expand_dims(arr[:,4],axis=1),
                np.expand_dims(arr[:,5],axis=1),
            ],
            axis=1,
            )
    
    smear_lep1_vec = uproot_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(x_smear[:,0],x_smear[:,1],x_smear[:,2],0.)
    smear_lep2_vec = uproot_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray(x_smear[:,3],x_smear[:,4],x_smear[:,5],0.)
    smear_lep12_vec = smear_lep2_vec + smear_lep1_vec

    return x_orig,x_smear,smear_lep12_vec.mass
