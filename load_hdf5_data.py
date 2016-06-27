import os
import h5py
import numpy as np


def load_representations(context_folder, ds_name, noise=0, suffix='s', mask=None):
    if noise == 0:
        x_name = ds_name + '.x' + suffix + '.hdf5'
        y_name = ds_name + '.y' + suffix + '.npy'
    else:    
        x_name = ds_name + '.' + str(int(100*noise)) + '.x' + suffix + '.hdf5'
        y_name = ds_name + '.' + str(int(100*noise)) + '.y' + suffix + '.npy'
    
    y = np.load( os.path.join(context_folder, ds_name, y_name) )
    
    x_data = h5py.File(os.path.join(context_folder, ds_name, x_name))
    
    if mask is None:
        x = x_data['x'][:,:]
    else:
        x = x_data['x'][mask,:]
        y = y[mask]
        
    x_data.close()
    return x, y


