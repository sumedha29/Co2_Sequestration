#for stt (MLP) gas
# In[3]:
from read_data import read
import os
import numpy as np
import glob
import collections
from read_model import MLP_single
from data_to_numpy import numpy_single


#read data
all_pressures,all_saturations,all_permeabilities,all_porosities,all_surf_inj_rate_series,all_surf_prod_rate_series,Ks,Rs = read()

#convert to numpy
features1_tr,target1_tr,features1_te,target1_te = numpy_single(all_pressures,all_saturations,all_permeabilities,all_porosities,all_surf_inj_rate_series,all_surf_prod_rate_series,Ks,Rs,all_saturations)



#train model MLP
batch_size = 250
model = MLP_single()

history = model.fit(features1_tr, target1_tr, epochs=500, batch_size=250,shuffle=True, verbose=1)

model.evaluate(features1_te,target1_te)