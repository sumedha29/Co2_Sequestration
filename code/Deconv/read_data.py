import os
import numpy as np
import glob
import collections



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
    
    
def read():
    all_pressures=[]
    all_saturations=[]
    all_permeabilities=[]
    all_porosities = []
    all_surf_inj_rate_series = []
    all_surf_prod_rate_series  = []
    Ks = []
    Rs = []

    files = sorted(glob.glob('../../data/*.out')) # 3D Grid with missing values
    print('All files:', files)

    for fil in files:
        root = fil.split("/")[-1]
        print('Processing file: ', fil)
        permeability = int(root[1])
        injection_rate = int(root[3])
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

        # get a dictionary of surface water production rates vs time for well #2
        surf_prod_rate_series = file.get_well_surface_rates(2,['Inst','Surface','Production','Rates'],['Water','STB/day'])

        if fil=='../../data/k1r1-h.out':
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
    
    return all_pressures,all_saturations,all_permeabilities,all_porosities,all_surf_inj_rate_series,all_surf_prod_rate_series,Ks,Rs