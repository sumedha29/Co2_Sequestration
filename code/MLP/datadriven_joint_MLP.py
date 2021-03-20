#for mtt (MLP)
# In[3]:
from read_data import read
import os
import numpy as np
import glob
import collections

from sklearn.metrics import mean_squared_error
from read_model import MLP_multiple
from data_to_numpy import numpy_multi


#read data
all_pressures,all_saturations,all_permeabilities,all_porosities,all_surf_inj_rate_series,all_surf_prod_rate_series,Ks,Rs = read()

#convert to numpy
features1_tr,features2_tr,target1_tr,target2_tr,target3_tr,features1_te,features2_te,target1_te,target2_te,target3_te = numpy_multi(all_pressures,all_saturations,all_permeabilities,all_porosities,all_surf_inj_rate_series,all_surf_prod_rate_series,Ks,Rs)


#train model LSTM
batch_size = 250
model = MLP_multiple()


history = model.fit([features1_tr,features2_tr], [target1_tr,target2_tr,target3_tr], epochs=100, batch_size=250,shuffle=True, verbose=1)

#evaluate on dataset
print(model.evaluate([features1_te,features2_te],[target1_te,target2_te,target3_te]))
