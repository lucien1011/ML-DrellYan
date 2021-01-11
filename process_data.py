import os,uproot_methods,math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

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

def make_ptregression_data(input_arr,mu=1.10,scale=0.1):
    energy_norm = 50.
    angle_norm = 1.
    
    condition = (input_arr[:,-1] - 90.) 
    arr = input_arr[np.squeeze(np.abs(condition) < 1)]

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
    
    eps1 = K.random_normal(shape=(x_orig.shape[0],1))
    sf1 = mu + scale * eps1

    eps2 = K.random_normal(shape=(x_orig.shape[0],1))
    sf2 = mu + scale * eps2

    smear_pt1 = tf.math.multiply(np.expand_dims(arr[:,0],axis=1),sf1) 
    eta1 = np.expand_dims(arr[:,1],axis=1)
    phi1 = np.expand_dims(arr[:,2],axis=1)
    smear_pt2 = tf.math.multiply(np.expand_dims(arr[:,3],axis=1),sf2) 
    eta2 = np.expand_dims(arr[:,4],axis=1)
    phi2 = np.expand_dims(arr[:,5],axis=1)

    x_smear = np.concatenate(
            [
                smear_pt1,
                np.expand_dims(arr[:,1],axis=1),
                np.expand_dims(arr[:,2],axis=1),
                smear_pt2,
                np.expand_dims(arr[:,4],axis=1),
                np.expand_dims(arr[:,5],axis=1),
            ],
            axis=1,
            )
    
    smear_mll = 2 * np.multiply(
            np.multiply(smear_pt1,smear_pt2),
            np.cosh(eta1-eta2) - np.cos(phi1-phi2), 
            )

    return x_orig,x_smear,smear_mll

def make_daflow_data(input_arr,mu=1.5,scale=0.05):
    energy_norm = 50.
    angle_norm = 1.
    
    condition = (input_arr[:,-1] - 90.) 
    arr0 = np.copy(input_arr[np.squeeze(np.abs(condition) < 1)])

    arr0[:,0] = arr0[:,0] / energy_norm
    arr0[:,1] = arr0[:,1] / angle_norm
    arr0[:,2] = arr0[:,2] / angle_norm
    arr0[:,3] = arr0[:,3] / energy_norm
    arr0[:,4] = arr0[:,4] / angle_norm
    arr0[:,5] = arr0[:,5] / angle_norm
    dphi = np.abs(arr0[:,5] - arr0[:,2])
        
    x0 = np.concatenate(
            [
                np.expand_dims(arr0[:,0],axis=1),
                np.expand_dims(arr0[:,1],axis=1),
                np.expand_dims(arr0[:,3],axis=1),
                np.expand_dims(arr0[:,4],axis=1),
                np.expand_dims(dphi,axis=1),
            ],
            axis=1,
            )

    arr1 = np.copy(input_arr[np.squeeze(np.abs(condition) < 1)])
    arr1[:,0] = arr1[:,0] / energy_norm
    arr1[:,1] = arr1[:,1] / angle_norm
    arr1[:,2] = arr1[:,2] / angle_norm
    arr1[:,3] = arr1[:,3] / energy_norm
    arr1[:,4] = arr1[:,4] / angle_norm
    arr1[:,5] = arr1[:,5] / angle_norm

    eps1 = K.random_normal(shape=(arr0.shape[0],1))
    sf1 = mu + scale * eps1

    eps2 = K.random_normal(shape=(arr0.shape[0],1))
    sf2 = mu + scale * eps2

    smear_pt1 = tf.math.multiply(np.expand_dims(arr1[:,0],axis=1),sf1) 
    arr1[:,1] = arr1[:,1] / angle_norm
    arr1[:,2] = arr1[:,2] / angle_norm
    smear_pt2 = tf.math.multiply(np.expand_dims(arr1[:,3],axis=1),sf2) 
    arr1[:,4] = arr1[:,4] / angle_norm
    arr1[:,5] = arr1[:,5] / angle_norm
    dphi = np.abs(arr1[:,5] - arr1[:,2])
        
    x1 = np.concatenate(
            [
                smear_pt1,
                np.expand_dims(arr1[:,1],axis=1),
                smear_pt2,
                np.expand_dims(arr1[:,4],axis=1),
                np.expand_dims(dphi,axis=1),
            ],
            axis=1,
            )
 
    return x0,x1

def make_conditional_flow_data(input_arr,mu1=0.0,mu2=0.0,scale1=0.10,scale2=0.10):
    energy_norm = 10.
    condition_norm = 0.5
    angle_norm = 1.
    
    eps1 = K.random_normal(shape=(input_arr.shape[0],1))
    sf1 = np.exp(mu1 + scale1 * eps1)

    eps2 = K.random_normal(shape=(input_arr.shape[0],1))
    sf2 = np.exp(mu2 + scale1 * eps2)

    smear_pt1 = tf.math.multiply(np.expand_dims(input_arr[:,0],axis=1),sf1) 
    smear_pt2 = tf.math.multiply(np.expand_dims(input_arr[:,3],axis=1),sf2)
    pt1 = np.expand_dims(input_arr[:,0],axis=1)
    eta1 = np.expand_dims(input_arr[:,1],axis=1)
    phi1 = np.expand_dims(input_arr[:,2],axis=1)
    pt2 = np.expand_dims(input_arr[:,3],axis=1)
    eta2 = np.expand_dims(input_arr[:,4],axis=1)
    phi2 = np.expand_dims(input_arr[:,5],axis=1)

    orig_mll = np.sqrt(2 * np.multiply(
            np.multiply(pt1,pt2),
            np.cosh(eta1-eta2) - np.cos(phi1-phi2), 
            ))

    smear_mll = np.sqrt(2 * np.multiply(
            np.multiply(smear_pt1,smear_pt2),
            np.cosh(eta1-eta2) - np.cos(phi1-phi2), 
            ))
        
    x = np.concatenate(
            [
                ( smear_pt1 - np.mean(pt1) ) / energy_norm,
                ( smear_pt2 - np.mean(pt2) ) / energy_norm,
                ( smear_mll - np.mean(orig_mll) ) / energy_norm,
            ],
            axis=1,
            )
    condition = np.concatenate(
            [
                sf1,
                sf2,
            ],
            axis=1,
            )
 
    return x,condition

def simuate_conditional_flow_data_mass(x,ms,batch_size=10,evt_size=1,energy_norm = 10.): 
    idx_batch = np.random.randint(0, len(ms), batch_size)
    idx_evt = np.random.randint(0, x.shape[1], evt_size)
    x_batch = x[idx_batch]
    x_batch = x_batch[:,idx_evt,:]

    mll = np.sqrt(2 * np.multiply(
            np.multiply(x_batch[:,:,0],x_batch[:,:,3]),
            np.cosh(x_batch[:,:,1]-x_batch[:,:,4]) - np.cos(x_batch[:,:,2]-x_batch[:,:,5]), 
            ))
        
    x = np.concatenate(
            [
                np.expand_dims(x_batch[:,:,0],axis=2) / energy_norm,
                np.expand_dims(x_batch[:,:,3],axis=2) / energy_norm,
                np.expand_dims(mll,axis=2) / energy_norm,
            ],
            axis=2,
            )
    x = np.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]))
    condition = ms[idx_batch] / energy_norm 
 
    return x,condition

def simulate_conditional_flow_data_ptscale(x_arr,pt1_mean=0.,pt2_mean=0.,energy_norm=10.,batch_size=512,event_size=2048,):

    ncond = 2
    sf1 = 2.*np.random.random_sample((batch_size,1))-1.
    sf2 = 2.*np.random.random_sample((batch_size,1))-1.

    cond_arr = np.concatenate([sf1,sf2],axis=1)
    cond_arr = np.expand_dims(cond_arr,axis=1)
    cond_arr = np.broadcast_to(cond_arr,(batch_size,event_size,ncond))
    condition = np.reshape(cond_arr,(batch_size*event_size,ncond))
    
    x_orig = np.expand_dims(x_arr,axis=0)
    x_orig = np.broadcast_to(x_orig,(batch_size,x_orig.shape[1],x_orig.shape[2]))
    x_orig = np.reshape(x_orig,(batch_size*x_orig.shape[1],x_orig.shape[2]))

    smear_pt1 = (1.+condition[:,0])*x_orig[:,0]
    smear_pt2 = (1.+condition[:,1])*x_orig[:,3]

    smear_mll = np.sqrt(2 * np.multiply(
        np.multiply(smear_pt1,smear_pt2),
        np.cosh(x_orig[:,1]-x_orig[:,4]) - np.cos(x_orig[:,2]-x_orig[:,5]), 
        ))
    
    x = np.concatenate(
            [
                (np.expand_dims(smear_pt1,axis=1) - pt1_mean) / energy_norm,
                (np.expand_dims(smear_pt2,axis=1) - pt2_mean) / energy_norm,
                np.expand_dims(smear_mll-1.,axis=1) / energy_norm,
            ],
            axis=1,
            )
    
    return x,condition
