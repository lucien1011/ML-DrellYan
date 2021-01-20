import os,uproot_methods,math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from Utils.ObjDict import ObjDict

def preprocess_conditional_flow_data_mass(x,energy_norm=10.,condition_norm=5.,mll0=90.,m0=90.):
    out = []
    mass_arr = x[:,-1]
    for m in np.unique(mass_arr):
        idx_mass = x[:,-1] == m
        
        x_arr = np.copy(x[idx_mass])
        
        mll = np.sqrt(2 * np.multiply(np.multiply(x_arr[:,0],x_arr[:,3]),np.cosh(x_arr[:,1]-x_arr[:,4]) - np.cos(x_arr[:,2]-x_arr[:,5]),))

        reco = np.concatenate(
                [
                    np.expand_dims(x_arr[:,0],axis=1) / energy_norm,
                    np.expand_dims(x_arr[:,3],axis=1) / energy_norm,
                    np.expand_dims(mll-mll0,axis=1) / condition_norm,
                ],
                axis=1,
                )

        condition = np.ones((x_arr.shape[0],1)) * (m-m0) / condition_norm
        
        out.append(ObjDict(x=reco,condition=condition))
    return out

def simuate_conditional_flow_data_mass(x,ms,batch_size=10,evt_size=1,energy_norm=10.): 
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

def simulate_conditional_flow_data_ptscale(x_arr,pt1_mean=0.,pt2_mean=0.,energy_norm=10.,batch_size=512,event_size=2048,sf1=None,sf2=None):

    ncond = 2
    if not sf1:
        sf1 = 2.*np.random.random_sample((batch_size,1))-1.
    else:
        sf1 = sf1 * np.ones((batch_size,1))
    if not sf2:
        sf2 = 2.*np.random.random_sample((batch_size,1))-1.
    else:
        sf2 = sf2 * np.ones((batch_size,1))

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

def preprocess_conditional_flow_data_cww(x,energy_norm=10.,):
    out = []
    param_arr = x[:,-1]
    for param in np.unique(param_arr):
        idx_param = x[:,-1] == param
        
        x_arr = np.copy(x[idx_param])
        
        reco = np.concatenate(
                [
                    np.expand_dims(x_arr[:,0]-125.,axis=1) / energy_norm,
                    np.expand_dims(x_arr[:,1]-90.,axis=1) / energy_norm,
                    np.expand_dims(x_arr[:,2],axis=1) / energy_norm,
                    #np.expand_dims(x_arr[:,3],axis=1) / energy_norm,
                    #np.expand_dims(x_arr[:,5],axis=1) / energy_norm,
                    #np.expand_dims(x_arr[:,6],axis=1) / energy_norm,
                ],
                axis=1,
                )

        condition = np.ones((x_arr.shape[0],1)) * param
        
        if param != -1.:
            out.append(ObjDict(x=reco,condition=condition,param=param))
        else:
            bkgObj = ObjDict(x=reco,condition=condition,param=param)
    return out,bkgObj

