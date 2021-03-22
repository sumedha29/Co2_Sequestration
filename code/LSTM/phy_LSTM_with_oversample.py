#for phywics model lstm (LSTM)
# In[3]:
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from read_data import read
import os
import numpy as np
import glob
import collections

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error
from read_model import LSTM_physics
from data_to_numpy import numpy_read_physics
from sklearn.utils import shuffle


#read data
all_pressures,all_saturations,all_permeabilities,all_porosities,all_surf_inj_rate_series,all_surf_prod_rate_series,Ks,Rs = read()


#data to numpy
oversampling_rate=200
training = [i for i in range(all_pressures.shape[0]) if i not in[2,3,11]]
testing = [2,3,11]

features1_tr,features2_tr,features3_tr,target1_tr,target2_tr,target3_tr,target31_tr,target_inj_tr=numpy_read_physics(all_pressures,all_saturations,all_permeabilities,all_porosities,all_surf_inj_rate_series,all_surf_prod_rate_series,Ks,Rs,oversampling_rate,training)

features1_te,features2_te,features3_te,target1_te,target2_te,target3_te,target31_te,target_inj_te=numpy_read_physics(all_pressures,all_saturations,all_permeabilities,all_porosities,all_surf_inj_rate_series,all_surf_prod_rate_series,Ks,Rs,1,testing)

features1_tr,features2_tr,target1_tr,target2_tr,target3_tr,target31_tr,target_inj_tr = shuffle(features1_tr,features2_tr,target1_tr,target2_tr,target3_tr, target31_tr, target_inj_tr, random_state=0)


#train model
batch_size = 250
model=LSTM_physics()
history = model.fit([features1_tr[:,:,4:7],features1_tr[:,:,3:4][:,:3,0],features1_tr[:,:,3:4][:,3,0],features1_tr[:,:,0:1][:,:3,0],features1_tr[:,:,0:1][:,3,0],features1_tr[:,:,1:2][:,:3,0],features1_tr[:,:,1:2][:,3,0],features1_tr[:,:,2:3][:,:3,0],features1_tr[:,:,2:3][:,3,0],features3_tr,features2_tr],[target1_tr,target2_tr,target3_tr,target31_tr,target_inj_tr], epochs=500, batch_size=250,shuffle=True, verbose=1)

print(model.evaluate([features1_te[:,:,4:7],features1_te[:,:,3:4][:,:3,0],features1_te[:,:,3:4][:,3,0],features1_te[:,:,0:1][:,:3,0],features1_te[:,:,0:1][:,3,0],features1_te[:,:,1:2][:,:3,0],features1_te[:,:,1:2][:,3,0], features1_te[:,:,2:3][:,:3,0],features1_te[:,:,2:3][:,3,0],features3_te,features2_te],[target1_te,target2_te,target3_te,target31_te,target_inj_te]))
