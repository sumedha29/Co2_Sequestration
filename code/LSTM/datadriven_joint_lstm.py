#for mtt (LSTM)
# In[3]:
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
from read_model import LSTM_multiple
from data_to_numpy import numpy_multi


#read data
all_pressures,all_saturations,all_permeabilities,all_porosities,all_surf_inj_rate_series,all_surf_prod_rate_series,Ks,Rs = read()

#convert to numpy
features1_tr,features2_tr,target1_tr,target2_tr,target3_tr,features1_te,features2_te,target1_te,target2_te,target3_te = numpy_multi(all_pressures,all_saturations,all_permeabilities,all_porosities,all_surf_inj_rate_series,all_surf_prod_rate_series,Ks,Rs)


#train model LSTM
batch_size = 250
model = LSTM_multiple()


history = model.fit([features1_tr[:,:,4:7],features1_tr[:,:,3:4][:,:3,0],features1_tr[:,:,3:4][:,3,0],features1_tr[:,:,0:1][:,:3,0],features1_tr[:,:,0:1][:,3,0],features1_tr[:,:,1:2][:,:3,0],features1_tr[:,:,1:2][:,3,0],features1_tr[:,:,2:3][:,:3,0],features1_tr[:,:,2:3][:,3,0],features2_tr], [target1_tr,target2_tr,target3_tr], epochs=100, batch_size=250,shuffle=True, verbose=1)

#evaluate on dataset
print(model.evaluate([features1_te[:,:,4:7],features1_te[:,:,3:4][:,:3,0],features1_te[:,:,3:4][:,3,0],features1_te[:,:,0:1][:,:3,0],features1_te[:,:,0:1][:,3,0],features1_te[:,:,1:2][:,:3,0],features1_te[:,:,1:2][:,3,0], features1_te[:,:,2:3][:,:3,0],features1_te[:,:,2:3][:,3,0],features2_te],[target1_te,target2_te,target3_te]))
