
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

features1 = []
features2 = []
target1 = []
target2 = []
target3 = []
print('start')
for i in range(all_pressures.shape[0]):
    for j in range(all_pressures.shape[1]):
        for k in range(all_pressures.shape[2]):
            for l in range(all_pressures.shape[3]):
                for m in range(all_pressures.shape[4]):
                    if j == 0 or j == 1 or j == 2:
                        features1.append([[(k/24),(l/24),(m/2),(j/71),(all_permeabilities[i][j][k][l][m]),(all_porosities[i][j][k][l][m]),(all_surf_inj_rate_series[i][j])], 
                                     [(k/24),(l/24),(m/2),(j/71),(all_permeabilities[i][j][k][l][m]),(all_porosities[i][j][k][l][m]),(all_surf_inj_rate_series[i][j])],
                                     [(k/24),(l/24),(m/2),(j/71),(all_permeabilities[i][j][k][l][m]),(all_porosities[i][j][k][l][m]),(all_surf_inj_rate_series[i][j])],
                                     [(k/24),(l/24),(m/2),(j/71),(all_permeabilities[i][j][k][l][m]),(all_porosities[i][j][k][l][m]),(all_surf_inj_rate_series[i][j])]])
                        features2.append([[(j/71),(Ks[i]/2),(Rs[i]/7)],
                                     [(j/71),(Ks[i]/2),(Rs[i]/7)],
                                     [(j/71),(Ks[i]/2),(Rs[i]/7)],
                                     [(j/71),(Ks[i]/2),(Rs[i]/7)]])
                        target1.append(all_saturations[i][j][k][l][m])
                        target2.append(all_pressures[i][j][k][l][m])
                        target3.append(all_surf_prod_rate_series[i][j])
                    else:
                        features1.append([[(k/24),(l/24),(m/2),(j-3/71),(all_permeabilities[i][j-3][k][l][m]),(all_porosities[i][j-3][k][l][m]),(all_surf_inj_rate_series[i][j-3])], 
                                        [(k/24),(l/24),(m/2),(j-2/71),(all_permeabilities[i][j-2][k][l][m]),(all_porosities[i][j-2][k][l][m]),(all_surf_inj_rate_series[i][j-2])],
                                        [(k/24),(l/24),(m/2),(j-1/71),(all_permeabilities[i][j-1][k][l][m]),(all_porosities[i][j-1][k][l][m]),(all_surf_inj_rate_series[i][j-1])],
                                        [(k/24),(l/24),(m/2),(j/71),(all_permeabilities[i][j][k][l][m]),(all_porosities[i][j][k][l][m]),(all_surf_inj_rate_series[i][j])]])
                        features2.append([[(j-3/71),(Ks[i]/2),(Rs[i]/7)],
                                        [(j-2/71),(Ks[i]/2),(Rs[i]/7)],
                                        [(j-1/71),(Ks[i]/2),(Rs[i]/7)],
                                        [(j/71),(Ks[i]/2),(Rs[i]/7)]])
                        target1.append(all_saturations[i][j][k][l][m])
                        target2.append(all_pressures[i][j][k][l][m])
                        target3.append(all_surf_prod_rate_series[i][j])

features1 = np.array(features1)
features2 = np.array(features2)
target1 = np.array(target1)
target1 = np.expand_dims(target1, axis=1)
target2 = np.array(target2)
target2 = np.expand_dims(target2, axis=1)
target3 = np.array(target3)
target3 = np.expand_dims(target3, axis=1)
print('end')
print(features1.shape)
print(features2.shape)
print(target1.shape)
print(target2.shape)
print(target3.shape)

features1_tr1 = features1[:2*72*25*25*3]
features1_te1 = features1[2*72*25*25*3:4*72*25*25*3]
features1_tr2 = features1[4*72*25*25*3:11*72*25*25*3]
features1_te2 = features1[11*72*25*25*3:12*72*25*25*3]
features1_tr3 = features1[12*72*25*25*3:]

features1_tr = np.concatenate((features1_tr1,features1_tr2,features1_tr3))
features1_te = np.concatenate((features1_te1,features1_te2))

features2_tr1 = features2[:2*72*25*25*3]
features2_te1 = features2[2*72*25*25*3:4*72*25*25*3]
features2_tr2 = features2[4*72*25*25*3:11*72*25*25*3]
features2_te2 = features2[11*72*25*25*3:12*72*25*25*3]
features2_tr3 = features2[12*72*25*25*3:]

features2_tr = np.concatenate((features2_tr1,features2_tr2,features2_tr3))
features2_te = np.concatenate((features2_te1,features2_te2))

target1_tr1 = target1[:2*72*25*25*3]
target1_te1 = target1[2*72*25*25*3:4*72*25*25*3]
target1_tr2 = target1[4*72*25*25*3:11*72*25*25*3]
target1_te2 = target1[11*72*25*25*3:12*72*25*25*3]
target1_tr3 = target1[12*72*25*25*3:]

target1_tr = np.concatenate((target1_tr1,target1_tr2,target1_tr3))
target1_te = np.concatenate((target1_te1,target1_te2))

target2_tr1 = target2[:2*72*25*25*3]
target2_te1 = target2[2*72*25*25*3:4*72*25*25*3]
target2_tr2 = target2[4*72*25*25*3:11*72*25*25*3]
target2_te2 = target2[11*72*25*25*3:12*72*25*25*3]
target2_tr3 = target2[12*72*25*25*3:]

target2_tr = np.concatenate((target2_tr1,target2_tr2,target2_tr3))
target2_te = np.concatenate((target2_te1,target2_te2))

target3_tr1 = target3[:2*72*25*25*3]
target3_te1 = target3[2*72*25*25*3:4*72*25*25*3]
target3_tr2 = target3[4*72*25*25*3:11*72*25*25*3]
target3_te2 = target3[11*72*25*25*3:12*72*25*25*3]
target3_tr3 = target3[12*72*25*25*3:]

target3_tr = np.concatenate((target3_tr1,target3_tr2,target3_tr3))
target3_te = np.concatenate((target3_te1,target3_te2))

from sklearn.utils import shuffle
features1_tr,features2_tr,target1_tr,target2_tr,target3_tr = shuffle(features1_tr,features2_tr,target1_tr,target2_tr,target3_tr, random_state=0)

print(features1_tr.shape)
print(features2_tr.shape)
print(target1_tr.shape)
print(target2_tr.shape)
print(target3_tr.shape)
print(features1_te.shape)
print(features2_te.shape)
print(target1_te.shape)
print(target2_te.shape)
print(target3_te.shape)

checkpoint_filepath = '/home/mll/sjp6046/models_new/gas_ind/joint_{val_water_loss:.8f}__{val_pressure_loss:.8f}__{val_gas_loss:.8f}.h5'
callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_water_loss',
    mode='min',
    save_best_only=True)


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

input_int = layers.Concatenate(axis=2)([input_1, time, x_input, y_input, z_input])
lstm_1 = LSTM(units=128, activation='relu', return_sequences=True)(input_int) 
lstm_2 = LSTM(units=64, activation='relu')(lstm_1) 
hidden_1 = Dense(32, activation='relu')(lstm_2)
hidden_2 = Dense(16, activation='relu')(hidden_1)
hidden_3 = Dense(8, activation='relu')(hidden_2) 
out1 = Dense(1, activation='relu', name='gas')(hidden_3)
print('out1 done ~')

lstm_1 = LSTM(units=128, activation='relu', return_sequences=True)(input_int) 
lstm_2 = LSTM(units=64, activation='relu')(lstm_1) 
hidden_1 = Dense(32, activation='relu')(lstm_2)
hidden_2 = Dense(16, activation='relu')(hidden_1)
hidden_3 = Dense(8, activation='relu')(hidden_2) 
out2 = Dense(1, activation='relu', name='pressure')(hidden_3)
print('out2 done ~')

input_2 = layers.Input(shape=(4, 3), name='second_input')
lstm_1 = LSTM(units=128, activation='relu', return_sequences=True)(input_2) 
lstm_2 = LSTM(units=64, activation='relu')(lstm_1) 
hidden_1 = Dense(32, activation='relu')(lstm_2)
hidden_2 = Dense(16, activation='relu')(hidden_1)
hidden_3 = Dense(8, activation='relu')(hidden_2) 
out3 = Dense(1, activation='relu', name='water')(hidden_3)
print('out3 done ~')

model = keras.Model(inputs=[input_1,prev_time,cur_time,prev_x_input,cur_x_input,prev_y_input,cur_y_input,prev_z_input,cur_z_input,input_2],outputs=[out1,out2,out3])
model.compile(loss='mse', optimizer=keras.optimizers.Adam(1e-4), metrics=['mae'])
print(model.summary())

history = model.fit([features1_tr[:,:,4:7],features1_tr[:,:,3:4][:,:3,0],features1_tr[:,:,3:4][:,3,0],features1_tr[:,:,0:1][:,:3,0],features1_tr[:,:,0:1][:,3,0],features1_tr[:,:,1:2][:,:3,0],features1_tr[:,:,1:2][:,3,0],
                     features1_tr[:,:,2:3][:,:3,0],features1_tr[:,:,2:3][:,3,0],features2_tr], [target1_tr,target2_tr,target3_tr], epochs=100, batch_size=250,
                    validation_data=([features1_te[:,:,4:7],features1_te[:,:,3:4][:,:3,0],features1_te[:,:,3:4][:,3,0],features1_te[:,:,0:1][:,:3,0],features1_te[:,:,0:1][:,3,0],features1_te[:,:,1:2][:,:3,0],features1_te[:,:,1:2][:,3,0], features1_te[:,:,2:3][:,:3,0],features1_te[:,:,2:3][:,3,0],features2_te],[target1_te,target2_te,target3_te]),
                    shuffle=True,callbacks=[callback], verbose=1)
