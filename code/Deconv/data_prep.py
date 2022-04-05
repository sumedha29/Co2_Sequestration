import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from read_data import read
import os
import numpy as np
import glob
import collections

from keras.layers import Dense
from keras.layers import LSTM
from keras import optimizers
from sklearn.metrics import mean_squared_error
# from read_model import LSTM_single
# from data_to_numpy import numpy_single

import cv2 as cv
import pickle as pkl


save_data_pkl = 'inputs&outputs2net.pkl'

all_pressures,all_saturations,all_permeabilities,all_porosities,all_surf_inj_rate_series,all_surf_prod_rate_series,Ks,Rs = read()


# In python:
# sio.savemat('/Users/anerudhraina/Documents/pennstate_sp22/CO2_seq/sample_pressure_1.mat',{'pres':all_pressures[0,0,:,:,0]})
# In MATLAB:
# load('/Users/anerudhraina/Documents/pennstate_sp22/CO2_seq/sample_pressure_1.mat')


# all_pressures_3D = np.zeros((all_pressures.shape[0],all_pressures.shape[1],1),np.float64);
# all_pressures_3D[:,:,0] = all_pressures[:,:];

# all_saturations_3D = np.zeros((all_saturations.shape[0],all_saturations.shape[1],1),np.float64);
# all_saturations_3D[:,:,0] = all_saturations[:,:];

# all_permeabilities_3D = np.zeros((all_permeabilities.shape[0],all_permeabilities.shape[1],1),np.float64);
# all_permeabilities_3D[:,:,0] = all_permeabilities[:,:];

# all_porosities_3D = np.zeros((all_porosities.shape[0],all_porosities.shape[1],1),np.float64);
# all_porosities_3D[:,:,0] = all_porosities[:,:];

all_pressures_3D = all_pressures;
all_saturations_3D = all_saturations;
all_permeabilities_3D = all_permeabilities;
all_porosities_3D = all_porosities;


grid_dims = [25,25,3];
inj_coordinates = [13,13];
prod_coordinates = [23,23];


all_surf_inj_rate_series_2D = np.zeros((all_surf_inj_rate_series.shape[0], \
										all_surf_inj_rate_series.shape[1], \
										grid_dims[0], grid_dims[1], 1), np.float64);
all_surf_inj_rate_series_2D[:,:,inj_coordinates[0]-1,inj_coordinates[1]-1,0] = all_surf_inj_rate_series[:,:];


all_surf_prod_rate_series_2D = np.zeros((all_surf_prod_rate_series.shape[0], \
										 all_surf_prod_rate_series.shape[1], \
										 grid_dims[0], grid_dims[1], 1), np.float64);
all_surf_prod_rate_series_2D[:,:,prod_coordinates[0]-1,prod_coordinates[1]-1,0] = all_surf_prod_rate_series[:,:];

net_inputs = np.concatenate((all_permeabilities_3D,all_porosities_3D,all_surf_inj_rate_series_2D), axis=4);

net_outputs = np.concatenate((all_pressures_3D, all_saturations_3D, all_surf_prod_rate_series_2D), axis=4);

print(net_inputs.shape)
print(net_outputs.shape)

# To save data
with open(save_data_pkl, 'wb+') as file:
	pkl.dump([net_inputs, net_outputs], file)


# To load data
# with open(save_data_pkl, 'rb') as file:
# 	inputs, outputs = pkl.load(file)



