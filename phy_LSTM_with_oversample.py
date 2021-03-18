base_dir = '/home/mll/sjp6046/ToyModel2/'

import os
import numpy as np
import glob
import collections
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras import optimizers
from sklearn.metrics import mean_squared_error

os.environ["CUDA_VISIBLE_DEVICES"]="1"
all_pressures=[]
all_saturations=[]
all_permeabilities=[]
all_porosities = []
all_surf_inj_rate_series = []
all_surf_prod_rate_series  = []
Ks = []
Rs = []

class GEM_File:

    grid_search_keywords = ['*GRID','*CART']
    time_search_keywords = ['Time','=','hr']

    def __init__(self, file_name):

        self.file_name = file_name
        self.input_list = self.read_file(self.file_name)
        self.current = 0 # pointer to current line number in file
        self.nx, self.ny, self.nz = self.get_grid(self.grid_search_keywords)

    # read file into list
    def read_file(self, file_name):
        input_list = []
        with open(file_name) as f:
            input_list = f.readlines()
        return input_list

    # read grid dimensions
    def get_grid(self, search_strings):
        for m,line in enumerate(self.input_list[self.current:]):
            if all(elem in line for elem in search_strings):
                line_list = line.split()
                self.current = self.current + m
                return int(line_list[-3]), int(line_list[-2]), int(line_list[-1])
        self.current = -1
        return -1, -1, -1

    # find next time step
    def get_time(self, search_strings):
        for m,line in enumerate(self.input_list[self.current+1:]):
            if all(elem in line for elem in search_strings):
                line_list = line.split()
                i = line_list.index('=')
                # save line number, return time and units
                self.current = self.current + m + 1
                return float(line_list[i+1]), line_list[i+2]
        self.current = -1 # end of file reached
        return -1.0, 'days'   

    # move file pointer to line containing all search strings in list
    def find_it(self, search_strings):
        if (self.current < 0):
            return
        for m,line in enumerate(self.input_list[self.current+1:]):
            if all(elem in line for elem in search_strings):
                self.current = self.current + m + 1
                return
        return

    # read model layer with constant value into 2D numpy array
    def read_constant_layer(self):
        line = self.input_list[self.current]
        line_list = line.split()
        var = np.ones((self.nx,self.ny))*float(line_list[-1])
        return var       

    # read entire grid with constant value into 3D numpy array
    def read_constant_block(self):
        line = self.input_list[self.current]
        line_list = line.split()
        var = np.ones((self.nx,self.ny,self.nz))*float(line_list[-1])
        return var

    # read a model layer with variable values into 2D numpy array
    def read_variable_layer(self):

        # converts string to float, fills missing values
        def my_float(s):
            s = s.strip()
            missing = 0.0
            return float(s) if s else missing

        line = self.input_list[self.current]
        var = np.zeros((self.nx,self.ny))
        while True:
            line_list = line.split()
            int_list = list(map(int, line_list[2:]))
            for j in range(self.nx):
                self.current = self.current + 1
                line = self.input_list[self.current]
                skip = len(str(self.nx))+4
                chunk = int((len(line)-skip)/len(int_list))
                line_list = [line[i:i+chunk] for i in range(skip, len(line), chunk)]
                float_list = list(map(my_float, line_list[:-1]))
                for m,n in enumerate(int_list):
                    i = n - 1
                    var[i][j] = float_list[m]
            if int_list[-1] == self.ny:
                break
            self.current = self.current + 2
            line = self.input_list[self.current]
        return var

    # read output variable into 2D or 3D numpy array
    def read_variable(self):
        self.current = self.current + 3
        line = self.input_list[self.current]

        if self.nz == 1: # 2D
            if all(elem in line for elem in ['All','values','are']):
                var = self.read_constant_layer()
            else:
                var = self.read_variable_layer() 
        else: # 3D
            if 'Plane' in line:
                var = np.zeros((self.nx,self.ny,self.nz))
                for k in range(self.nz):
                    if all(elem in line for elem in ['All','values','are']):
                        layer = self.read_constant_layer()
                    else:
                        self.current = self.current + 1                 
                        layer = self.read_variable_layer()
                    var[:,:,k] = layer
                    self.current = self.current + 2
                    line = self.input_list[self.current]
            else:
                 var = self.read_constant_block()
        return var

    # get output variable at all time steps
    def get_variable(self, search_strings):
        self.current = 0 # go to beginning of file
        variables = {}
        while True:
            # look for next time step
            time, units = self.get_time(self.time_search_keywords)
            if (self.current == -1.0): return variables # end of file
            # get the selected variable values
            self.find_it(search_strings)
            var = self.read_variable()
            variables.update({time : var})
        return variables

    # get list of well coordinates
    def get_well_coords(self, well_num, location_strings):
        self.current = 0 # go to beginning of file
       # find well coordinates
        well_str = str(well_num)
        location_strings.append(well_str)
        self.find_it(location_strings)
        self.current = self.current + 1
        line = self.input_list[self.current]
        line_list = line.split()
        i = int(line_list[0])
        j = int(line_list[1])
        k_string = line_list[2]
        if ':' in k_string:
            k_list = k_string.split(':')
            k1 = int(k_list[0])
            k2 = int(k_list[1])
            coords = [(i, j, k) for k in range(k1,k2+1)]
        else:
            k = [int(line_list[2])]
            coords = [(i, j, k)]
        return coords

    # get well operating parameters
    def get_well_params(self, well_num, search_strings, subsearch_strings):
        self.current = 0 # go to beginning of file
        well_str = str(well_num)
        search_strings.append(well_str)
        self.find_it(search_strings)
        self.find_it(subsearch_strings)
        line = self.input_list[self.current]
        line_list = line.split()
        return float(line_list[-1])

    # get well surface rate at a particular time
    def get_well_surface_rate(self, well_num, search_strings, subsearch_strings):
            # look for well rate
            self.find_it(search_strings)
            self.find_it(subsearch_strings)
            line = self.input_list[self.current]
            line_list = line.split()[len(subsearch_strings):]
            return float(line_list[well_num*2-1])        

    # get well surface rates as a dictionary of rate vs time
    def get_well_surface_rates(self, well_num, search_strings, subsearch_strings):
        self.current = 0
        values = {}
        # look for time zero
        time, units = self.get_time(self.time_search_keywords)
        # no injection at time zero
        values.update({time : 0.0})
        while True:
            # look for next time step
            time, units = self.get_time(self.time_search_keywords)
            if (self.current == -1.0): return values # end of file
            rate = self.get_well_surface_rate(well_num, search_strings, subsearch_strings)
            values.update({time : rate})
        return values

    # get well surface rates as a dictionary of 2D numpy arrays shaped like the top surface of the model grid
    def get_well_surface_maps(self, well_num, location_strings, search_strings, subsearch_strings):
        self.current = 0
        variables = {}
        i,j,k = self.get_well_coords(well_num, location_strings)[0]
        rates = self.get_well_surface_rates(well_num, search_strings, subsearch_strings)
        for time,rate in rates.items():
            var = np.zeros((self.nx,self.ny))
            var[i-1][j-1] = rate
            variables.update({time : var})
        return variables

files = sorted(glob.glob('/home/mll/sjp6046/ToyModel2/*.out')) # 3D Grid with missing values

print('All files:', files)

for fil in files:
    print('Processing file: ', fil)
    permeability = int(fil.split('k')[1][0])
    injection_rate = int(fil.split('r')[1][0])
    Ks.append(int(permeability)-1)
    Rs.append(int(injection_rate)-1)
    
    file = GEM_File(fil)

    # get a dictionary of pressures in numpy array vs time
    pressures = file.get_variable(['Pressure','(psia)'])

    # get a list of all time steps
    times = list(pressures.keys())

    # get a dictionary of saturations in numpy array vs time
    saturations = file.get_variable(['Gas','Saturation'])

    # get a dictionary of I-direction permeabilities in numpy array vs time
    permeabilities = file.get_variable(['I-direction','Permeabilities'])

    # get a dictionary of porosities in numpy array vs time
    porosities = file.get_variable(['Current','Porosity'])

    # get a dictionary of surface CO2 injection rates vs time for well #1
    surf_inj_rate_series = file.get_well_surface_rates(1,['Inst','Surface','Injection','Rates'],['Gas','MSCF/day'])
    
    # # get a dictionary of surface water production rates vs time for well #2
    surf_prod_rate_series = file.get_well_surface_rates(2,['Inst','Surface','Production','Rates'],['Water','STB/day'])
    
    if fil=='k1r1-h.out':
        surf_prod_rate_series[60] = surf_prod_rate_series[31]
        surf_prod_rate_series = collections.OrderedDict(sorted(surf_prod_rate_series.items()))

    # get a dictionary of surface CO2 injection rates in 2D numpy array vs time for well #1
    surf_inj_rate_maps = file.get_well_surface_maps(1,['*PERF','*GEO'], ['Inst','Surface','Injection','Rates'], ['Gas','MSCF/day'])

    # get a dictionary of surface water production rates in 2D numpy array vs time for well #2
    surf_prod_rate_maps = file.get_well_surface_maps(2,['*PERF','*GEO'], ['Inst','Surface','Production','Rates'], ['Water','STB/day'])

    pressures_np = np.array(list(pressures.values()))
    all_pressures.append(pressures_np)
    
    saturations_np = np.array(list(saturations.values()))
    all_saturations.append(saturations_np)
    
    permeabilities_np = np.array(list(permeabilities.values()))
    all_permeabilities.append(permeabilities_np)
    
    porosities_np = np.array(list(porosities.values()))
    all_porosities.append(porosities_np)
    
    surf_inj_rate_series_np = np.array(list(surf_inj_rate_series.values()))
    print(surf_inj_rate_series_np.shape)
    all_surf_inj_rate_series.append(surf_inj_rate_series_np)
    
    surf_prod_rate_series_np = np.array(list(surf_prod_rate_series.values()))
    all_surf_prod_rate_series.append(surf_prod_rate_series_np)


Ks = np.reshape(Ks, (27,1))

Rs = np.reshape(Rs, (27,1))

all_pressures = np.array(all_pressures)
print(all_pressures.shape)

all_saturations = np.array(all_saturations)
print(all_saturations.shape)

all_permeabilities = np.array(all_permeabilities)
print(all_permeabilities.shape)

all_porosities = np.array(all_porosities)
print(all_porosities.shape)

all_surf_inj_rate_series = np.array(all_surf_inj_rate_series)
print(all_surf_inj_rate_series.shape)

all_surf_prod_rate_series = np.array(all_surf_prod_rate_series)
print(all_surf_prod_rate_series.shape)


#normalizing pressures
max_ = np.amax(all_pressures)
min_ = np.amin(all_pressures)
all_pressures = (all_pressures-min_)/(max_-min_)


#normalizing saturation
max_ = np.amax(all_saturations)
min_ = np.amin(all_saturations)
all_saturations = (all_saturations-min_)/(max_-min_)

#normalizing permeabilities
max_ = np.amax(all_permeabilities)
min_ = np.amin(all_permeabilities)
all_permeabilities = (all_permeabilities-min_)/(max_-min_)

#normalizing porosities
max_ = np.amax(all_porosities)
min_ = np.amin(all_porosities)
all_porosities = (all_porosities-min_)/(max_-min_)

#normalizing surf_inj_rate_series
max_ = np.amax(all_surf_inj_rate_series)
min_ = np.amin(all_surf_inj_rate_series)
all_surf_inj_rate_series = (all_surf_inj_rate_series-min_)/(max_-min_)

#normalizing 
max_ = np.amax(all_surf_prod_rate_series)
min_ = np.amin(all_surf_prod_rate_series)
all_surf_prod_rate_series = (all_surf_prod_rate_series-min_)/(max_-min_)

print(all_permeabilities.shape, all_saturations.shape, all_pressures.shape, all_surf_inj_rate_series.shape)

oversampling_rate=200

training = [i for i in range(all_pressures.shape[0]) if i not in[2,3,11]]
testing = [2,3,11]


features1_tr = []
features2_tr = []
features3_tr = []
target1_tr = []
target2_tr = []
target3_tr = []
target31_tr = []
target_inj_tr = []

for i in training:
    for j in range(all_pressures.shape[1]):
        for k in range(all_pressures.shape[2]):
            for l in range(all_pressures.shape[3]):
                for m in range(all_pressures.shape[4]):
                    if (k,l) not in [(13,13),(23,23)]:
                        features1_tr.append([[(k/24),(l/24),(m/2),(max(j-3,0)/71),(all_permeabilities[i][max(j-3,0)][k][l][m]),(all_porosities[i][max(j-3,0)][k][l][m]),(all_surf_inj_rate_series[i][max(j-3,0)])], 
                                        [(k/24),(l/24),(m/2),(max(j-2,0)/71),(all_permeabilities[i][max(j-2,0)][k][l][m]),(all_porosities[i][max(j-2,0)][k][l][m]),(all_surf_inj_rate_series[i][max(j-2,0)])],
                                        [(k/24),(l/24),(m/2),(max(j-2,0)/71),(all_permeabilities[i][max(j-1,0)][k][l][m]),(all_porosities[i][max(j-1,0)][k][l][m]),(all_surf_inj_rate_series[i][max(j-1,0)])],
                                        [(k/24),(l/24),(m/2),(j/71),(all_permeabilities[i][j][k][l][m]),(all_porosities[i][j][k][l][m]),(all_surf_inj_rate_series[i][j])]])
                        features2_tr.append([[(max(j-3,0)/71),(Ks[i]/2),(Rs[i]/7)],
                                        [(max(j-2,0)/71),(Ks[i]/2),(Rs[i]/7)],
                                        [(max(j-1,0)/71),(Ks[i]/2),(Rs[i]/7)],
                                        [(j/71),(Ks[i]/2),(Rs[i]/7)]])

                        features3_tr.append([[all_permeabilities[i][j//10][k//10][l//10][0],0,0],[0,all_permeabilities[i][j//10][k//10][l//10][1],0],[0,0,all_permeabilities[i][j//10][k//10][l//10][2]]])
                        target1_tr.append(all_saturations[i][j][k][l][m])
                        target2_tr.append(all_pressures[i][j][k][l][m])
                        target3_tr.append(all_surf_prod_rate_series[i][j])
                        target31_tr.append(0)
                        target_inj_tr.append(0)
                    else:
                        features1_tr+=[[[(k/24),(l/24),(m/2),(max(j-3,0)/71),(all_permeabilities[i][max(j-3,0)][k][l][m]),(all_porosities[i][max(j-3,0)][k][l][m]),(all_surf_inj_rate_series[i][max(j-3,0)])], 
                                        [(k/24),(l/24),(m/2),(max(j-2,0)/71),(all_permeabilities[i][max(j-2,0)][k][l][m]),(all_porosities[i][max(j-2,0)][k][l][m]),(all_surf_inj_rate_series[i][max(j-2,0)])],
                                        [(k/24),(l/24),(m/2),(max(j-2,0)/71),(all_permeabilities[i][max(j-1,0)][k][l][m]),(all_porosities[i][max(j-1,0)][k][l][m]),(all_surf_inj_rate_series[i][max(j-1,0)])],
                                        [(k/24),(l/24),(m/2),(j/71),(all_permeabilities[i][j][k][l][m]),(all_porosities[i][j][k][l][m]),(all_surf_inj_rate_series[i][j])]]] * oversampling_rate
                        
                        features2_tr+=[[[(max(j-3,0)/71),(Ks[i]/2),(Rs[i]/7)],
                                        [(max(j-2,0)/71),(Ks[i]/2),(Rs[i]/7)],
                                        [(max(j-1,0)/71),(Ks[i]/2),(Rs[i]/7)],
                                        [(j/71),(Ks[i]/2),(Rs[i]/7)]]] * oversampling_rate
                        
                        features3_tr+=[[[all_permeabilities[i][j//10][k//10][l//10][0],0,0],[0,all_permeabilities[i][j//10][k//10][l//10][1],0],[0,0,all_permeabilities[i][j//10][k//10][l//10][2]]]] * oversampling_rate
                        target1_tr += [all_saturations[i][j][k][l][m]] * oversampling_rate
                        target2_tr += [all_pressures[i][j][k][l][m]] * oversampling_rate
                        target3_tr += [all_surf_prod_rate_series[i][j]] * oversampling_rate
                        target31_tr += [all_surf_prod_rate_series[i][j]] * oversampling_rate
                        target_inj_tr += [all_surf_inj_rate_series[i][j]] * oversampling_rate
                        

features1_tr = np.array(features1_tr)
features2_tr = np.array(features2_tr)
features3_tr = np.array(features3_tr)
target1_tr = np.array(target1_tr)
target1_tr = np.expand_dims(target1_tr, axis=1)
target2_tr = np.array(target2_tr)
target2_tr = np.expand_dims(target2_tr, axis=1)
target3_tr = np.array(target3_tr)
target3_tr = np.expand_dims(target3_tr, axis=1)
target31_tr = np.array(target31_tr)
target31_tr = np.expand_dims(target31_tr, axis=1)
target_inj_tr = np.array(target_inj_tr)
target_inj_tr = np.expand_dims(target_inj_tr, axis=1)
print(features1_tr.shape)
print(features2_tr.shape)
print(features3_tr.shape)
print(target1_tr.shape)
print(target2_tr.shape)
print(target3_tr.shape)
print(target31_tr.shape)
print(target_inj_tr.shape)


features1_te = []
features2_te = []
features3_te = []
target1_te = []
target2_te = []
target3_te = []
target31_te = []
target_inj_te = []

for i in testing:
    for j in range(all_pressures.shape[1]):
        for k in range(all_pressures.shape[2]):
            for l in range(all_pressures.shape[3]):
                for m in range(all_pressures.shape[4]):
                    features1_te.append([[(k/24),(l/24),(m/2),(max(j-3,0)/71),(all_permeabilities[i][max(j-3,0)][k][l][m]),(all_porosities[i][max(j-3,0)][k][l][m]),(all_surf_inj_rate_series[i][max(j-3,0)])], 
                                    [(k/24),(l/24),(m/2),(max(j-2,0)/71),(all_permeabilities[i][max(j-2,0)][k][l][m]),(all_porosities[i][max(j-2,0)][k][l][m]),(all_surf_inj_rate_series[i][max(j-2,0)])],
                                    [(k/24),(l/24),(m/2),(max(j-2,0)/71),(all_permeabilities[i][max(j-1,0)][k][l][m]),(all_porosities[i][max(j-1,0)][k][l][m]),(all_surf_inj_rate_series[i][max(j-1,0)])],
                                    [(k/24),(l/24),(m/2),(j/71),(all_permeabilities[i][j][k][l][m]),(all_porosities[i][j][k][l][m]),(all_surf_inj_rate_series[i][j])]])
                    features2_te.append([[(max(j-3,0)/71),(Ks[i]/2),(Rs[i]/7)],
                                    [(max(j-2,0)/71),(Ks[i]/2),(Rs[i]/7)],
                                    [(max(j-1,0)/71),(Ks[i]/2),(Rs[i]/7)],
                                    [(j/71),(Ks[i]/2),(Rs[i]/7)]])
                    features3_te.append([[all_permeabilities[i][j//10][k//10][l//10][0],0,0],[0,all_permeabilities[i][j//10][k//10][l//10][1],0],[0,0,all_permeabilities[i][j//10][k//10][l//10][2]]])
                    target1_te.append(all_saturations[i][j][k][l][m])
                    target2_te.append(all_pressures[i][j][k][l][m])
                    target3_te.append(all_surf_prod_rate_series[i][j])
                    if (k,l) not in [(13,13),(23,23)]:
                        target31_te.append(0)
                        target_inj_te.append(0)
                    else:
                        target31_te.append(all_surf_prod_rate_series[i][j])
                        target_inj_te.append(all_surf_inj_rate_series[i][j])
                       

features1_te = np.array(features1_te)
features2_te = np.array(features2_te)
features3_te = np.array(features3_te)
target1_te = np.array(target1_te)
target1_te = np.expand_dims(target1_te, axis=1)
target2_te = np.array(target2_te)
target2_te = np.expand_dims(target2_te, axis=1)
target3_te = np.array(target3_te)
target3_te = np.expand_dims(target3_te, axis=1)
target31_te = np.array(target31_te)
target31_te = np.expand_dims(target31_te, axis=1)
target_inj_te = np.array(target_inj_te)
target_inj_te = np.expand_dims(target_inj_te, axis=1)
print(features1_te.shape)
print(features2_te.shape)
print(features3_te.shape)
print(target1_te.shape)
print(target2_te.shape)
print(target3_te.shape)
print(target31_te.shape)
print(target_inj_te.shape)


from sklearn.utils import shuffle
features1_tr,features2_tr,target1_tr,target2_tr,target3_tr,target31_tr,target_inj_tr = shuffle(features1_tr,features2_tr,target1_tr,target2_tr,target3_tr, target31_tr, target_inj_tr, random_state=0)



batch_size = 250
input_1 = layers.Input(shape=(4,3),name='first_input')
prev_time = layers.Input(shape=(3,),name='prev_time')
cur_time = layers.Input(shape=(1,),name='cur_time')
time_temp = layers.Concatenate(axis=1)([prev_time, cur_time])
time = layers.Reshape((4,1))(time_temp)
prev_x_input = layers.Input(shape=(3,),name='prev_x_input')
cur_x_input = layers.Input(shape=(1,),name='cur_x_input')
x_input_temp = layers.Concatenate(axis=1)([prev_x_input, cur_x_input])
x_input = layers.Reshape((4,1))(x_input_temp)
prev_y_input = layers.Input(shape=(3,),name='prev_y_input')
cur_y_input = layers.Input(shape=(1,),name='cur_y_input')
y_input_temp = layers.Concatenate(axis=1)([prev_y_input, cur_y_input])
y_input = layers.Reshape((4,1))(y_input_temp)
prev_z_input = layers.Input(shape=(3,),name='prev_z_input')
cur_z_input = layers.Input(shape=(1,),name='cur_z_input')
z_input_temp = layers.Concatenate(axis=1)([prev_z_input, cur_z_input])
z_input = layers.Reshape((4,1))(z_input_temp)
perm_input = layers.Input(shape=(3,3),name='perm_input')

input_int = layers.Concatenate(axis=2)([input_1, time, x_input, y_input, z_input])
lstm_1 = LSTM(units=128, activation='relu', return_sequences=True,unroll=True)(input_int) 
lstm_2 = LSTM(units=64, activation='relu',unroll=True)(lstm_1) 
hidden_1 = Dense(32, activation='relu')(lstm_2)
hidden_2 = Dense(16, activation='relu')(hidden_1)
hidden_3 = Dense(8, activation='relu')(hidden_2) 
out1 = Dense(1, activation='relu', name='gas')(hidden_3)
print('out1 done ~')

lstm_1 = LSTM(units=128, activation='relu', return_sequences=True,unroll=True)(input_int) 
lstm_2 = LSTM(units=64, activation='relu',unroll=True)(lstm_1) 
hidden_1 = Dense(32, activation='relu')(lstm_2)
hidden_2 = Dense(16, activation='relu')(hidden_1)
hidden_3 = Dense(8, activation='relu')(hidden_2) 
out2 = Dense(1, activation='relu', name='pressure')(hidden_3)
print('out2 done ~')

input_2 = layers.Input(shape=(4, 3), name='second_input')
lstm_1 = LSTM(units=128, activation='relu', return_sequences=True,unroll=True)(input_2) 
lstm_2 = LSTM(units=64, activation='relu',unroll=True)(lstm_1) 
hidden_1 = Dense(32, activation='relu')(lstm_2)
hidden_2 = Dense(16, activation='relu')(hidden_1)
hidden_3 = Dense(8, activation='relu')(hidden_2) 
out3 = Dense(1, activation='relu', name='water')(hidden_3)
print('out3 done ~')
    
def gradient_1(params):
    out1, out2, cur_time, cur_x_input, cur_y_input, cur_z_input, perm_input = params     
    gradient_with_time = tf.keras.backend.gradients(out1,cur_time)[0]
    bias = tf.expand_dims(tf.convert_to_tensor([0.,0.,tf.keras.backend.variable(0.)]),0)
    bias = tf.expand_dims(bias,2)
    pressure_grad_x = tf.keras.backend.gradients(out2,cur_x_input)[0]
    pressure_grad_y = tf.keras.backend.gradients(out2,cur_y_input)[0]
    pressure_grad_z = tf.keras.backend.gradients(out2,cur_z_input)[0]
    
    pressure_grad = tf.convert_to_tensor([pressure_grad_x,pressure_grad_y,pressure_grad_z])
    pressure_grad = tf.keras.backend.permute_dimensions(pressure_grad,(1,0,2))
    coeff = (1-out1)/tf.keras.backend.variable(1e-8)   
    m = tf.multiply(perm_input,(pressure_grad - bias))
    m_grad_x = tf.keras.backend.gradients(m,cur_x_input)[0]
    m_grad_y = tf.keras.backend.gradients(m,cur_y_input)[0]
    m_grad_z = tf.keras.backend.gradients(m,cur_z_input)[0]
    
    m_grad = m_grad_x + m_grad_y + m_grad_z
    m_final = tf.multiply(coeff, m_grad)
    eqn = -gradient_with_time - m_final
    return eqn

def gradient_2(params):
    out1, out2, cur_time, cur_x_input, cur_y_input, cur_z_input, perm_input = params     
    gradient_with_time = tf.keras.backend.gradients(out1,cur_time)[0]
    bias = tf.expand_dims(tf.convert_to_tensor([0.,0.,tf.keras.backend.variable(0.)]),0)
    bias = tf.expand_dims(bias,2)
    pressure_grad_x = tf.keras.backend.gradients(out2,cur_x_input)[0]
    pressure_grad_y = tf.keras.backend.gradients(out2,cur_y_input)[0]
    pressure_grad_z = tf.keras.backend.gradients(out2,cur_z_input)[0]
    
    pressure_grad = tf.convert_to_tensor([pressure_grad_x,pressure_grad_y,pressure_grad_z])
    pressure_grad = tf.keras.backend.permute_dimensions(pressure_grad,(1,0,2))
    coeff = (out1)/tf.keras.backend.variable(1e-8)     
    m = tf.multiply(perm_input,(pressure_grad - bias))
    m_grad_x = tf.keras.backend.gradients(m,cur_x_input)[0]
    m_grad_y = tf.keras.backend.gradients(m,cur_y_input)[0]
    m_grad_z = tf.keras.backend.gradients(m,cur_z_input)[0]
    
    m_grad = m_grad_x + m_grad_y + m_grad_z
    m_final = tf.multiply(coeff, m_grad)
    eqn = gradient_with_time - m_final
    return eqn


from tensorflow.python.keras.layers import Lambda;    
grad_out_1 = Lambda(gradient_1)([out1, out2, cur_time, cur_x_input, cur_y_input, cur_z_input, perm_input]) 
grad_out_2 = Lambda(gradient_2)([out1, out2, cur_time, cur_x_input, cur_y_input, cur_z_input, perm_input])    

model = keras.Model(inputs=[input_1,prev_time,cur_time,prev_x_input,cur_x_input,prev_y_input,cur_y_input,prev_z_input,cur_z_input,perm_input,input_2],outputs=[out1,out2,out3,grad_out_1,grad_out_2])
model = model

model.compile(loss='mse', optimizer=keras.optimizers.Adam(1e-5), metrics=['mae'])
    
print(model.summary())

checkpoint_filepath = '/home/mll/sjp6046/models_new/joint_{val_water_loss:.8f}__{val_pressure_loss:.8f}__{val_gas_loss:.8f}.h5'
callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_water_loss',
    mode='min',
    save_best_only=True)


history = model.fit([features1_tr[:,:,4:7],features1_tr[:,:,3:4][:,:3,0],features1_tr[:,:,3:4][:,3,0],features1_tr[:,:,0:1][:,:3,0],features1_tr[:,:,0:1][:,3,0],features1_tr[:,:,1:2][:,:3,0],features1_tr[:,:,1:2][:,3,0],
                     features1_tr[:,:,2:3][:,:3,0],features1_tr[:,:,2:3][:,3,0],features3_tr,features2_tr], [target1_tr,target2_tr,target3_tr,target31_tr,target_inj_tr], epochs=100, batch_size=250,
                    validation_data=([features1_te[:,:,4:7],features1_te[:,:,3:4][:,:3,0],features1_te[:,:,3:4][:,3,0],features1_te[:,:,0:1][:,:3,0],features1_te[:,:,0:1][:,3,0],features1_te[:,:,1:2][:,:3,0],features1_te[:,:,1:2][:,3,0], features1_te[:,:,2:3][:,:3,0],features1_te[:,:,2:3][:,3,0],features3_te,features2_te],[target1_te,target2_te,target3_te,target31_te,target_inj_te]),
                    shuffle=True, verbose=1,callbacks=[callback])
