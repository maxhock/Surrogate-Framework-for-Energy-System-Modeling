"""
direct conversion of HD5 importer by Max Schulze
"""

import pandas as pd
import os
from pathlib import Path

class ResultContainer(object):
    """ Result/input data container for reporting functions. """
    def __init__(self, data, result):
        self._data = data
        self._result = result

def load(filename):
    """Load a urbs model result container from a HDF5 store file.

    Args:
        filename: an existing HDF5 store file

    Returns:
        prob: the modified instance containing the result cache
    """
    with pd.HDFStore(filename, mode='r') as store:
        data_cache = {}
        for group in store.get_node('data'):
            data_cache[group._v_name] = store[group._v_pathname]

        result_cache = {}
        for group in store.get_node('result'):
            result_cache[group._v_name] = store[group._v_pathname]

    return ResultContainer(data_cache, result_cache)


# variation(x, y, z): x = pv, y = battery, z = heatpump 

# define variations of pv (v1), battery (v2) and heatpump (v3) and create list of filenames:
# if not all variations of the possible combinations of v1, v2 and v3 are existent, missing
# files will be skipped.

v1 = [0.6, 0.8, 1.0]
v2 = [0.8, 1.0, 1.2]
v3 = [0.8, 1.0, 1.2]



h5_files = []
for x in v1:
    for y in v2:
        for z in v3:
            file = "variation(" + str(x) + ", " + str(y) + ", " + str(z) + ")"
            h5_files.append(file)


# Define process and commodity variables to be included in dataframe. 
# If labels are not existent in the data, an empty column will be inserted.
# This can be necessary if ML model is supposed to be applied on data with these 
# technologies such that the dimensions are maintained.


demand_data = ['electricity', 'water_heat', 'space_heat', 'mobility']
storage_capacity_data = ['battery_private', 'heat_storage']
inst_technology_data = ['Rooftop PV', 'heatpump_air']


# Define wheter or not to include previous datasteps (t-n) in each row for timeseries data

add_previous_timesteps = True
number_previous_timesteps = 8


# Data-Frame to store whole Dataset
complete_data_original = pd.DataFrame()


# Load desired data from h5 files into Dataframe iterating over all scenarios
for file_name in h5_files:
    fullinname=os.path.join('./scenario_data' + '/Original' + '/flex_all_{}.h5'.format(file_name))
    if Path(fullinname).is_file():
        hdf = load(fullinname)
        print("{} FOUND".format(file_name))
    else: 
        print("{} MISSING".format(file_name))
        continue

    data = hdf._data
    results = hdf._result
    
    
    # Data-Frame to store data from the current scenario
    scenario_data = pd.DataFrame()
    
    # load demand data as defined above
    for demand in demand_data:
        label = demand + '_demand'
        scenario_data[label] = data['demand'].T[data['demand'].keys().droplevel(level = 0) == demand].sum()[1:]
        
        
        # If desired, add previous timestep (t-n) data for timeseries data in each row
        if add_previous_timesteps:
            for idx in range(number_previous_timesteps):
                new_label = label + ' [t-' + str((idx+1)) + ']'
                scenario_data[new_label] = scenario_data[label].shift(idx+1, fill_value = 0.0)
                  
    # load storage data as defined above    
    for storage_type in storage_capacity_data:
        label = storage_type + '_capacity'
        scenario_data[label] = results['cap_sto_c'].loc[:,:, storage_type].sum()
        
    # add capacities as defined above   
    for technology in inst_technology_data:
        label = technology + '_capacity'
        scenario_data[label] = results['cap_pro'].loc[:,:, technology].sum() if technology in results['cap_pro'].keys().droplevel(level = [0,1]) else 0.0
        
              
    scenario_data['solar_irradiation'] = data['supim'].T.iloc[0][1:]
    
    # add mean cop relative to installed heatpump capacity at each building
    heatpump_capacity = results['cap_pro'][:,:,'heatpump_air']
    relative_capacity = heatpump_capacity.div(heatpump_capacity.sum())
    cop = data['eff_factor'].T[data['eff_factor'].keys().get_level_values(level=1).str.startswith('heatpump')]
    scenario_data['relative_cop'] = cop.T.mul(relative_capacity.values).T.sum()
    
    # add availability of cars relative to total charging capacity (57 charging stations available)
    scenario_data['charging_availability'] = data['eff_factor'].T[data['eff_factor'].keys().get_level_values(level=1).str.startswith('charging_station')].sum()/57
    
    # add net demand as import - export
    scenario_data['net_demand'] = results['e_pro_out'].loc[:,:,:,'import'].values - results['e_pro_out'].loc[:,:,:, 'feed_in'].values

    
    # concatenate data from current scenario to dataframe for complete dataset
    complete_data_original = pd.concat([complete_data_original, scenario_data])


    # variation(x, y, z): x = pv, y = battery, z = heatpump 

# define variations of pv (v1), battery (v2) and heatpump (v3) and create list of filenames:
# if not all variations of the possible combinations of v1, v2 and v3 are existent, missing
# files will be skipped.

v1 = [0.8, 0.9, 1.0]
v2 = [0.9, 1.0, 1.1]
v3 = [0.9, 1.0, 1.1]


h5_files = []
for x in v1:
    for y in v2:
        for z in v3:
            file = "variation(" + str(x) + ", " + str(y) + ", " + str(z) + ")"
            h5_files.append(file)


# Define process and commodity variables to be included in dataframe. 
# If labels are not existent in the data, an empty column will be inserted.
# This can be necessary if ML model is supposed to be applied on data with these 
# technologies such that the dimensions are maintained.

# Due to nomenclature, charging_station, PV and mobility have to be addressed seperately.

demand_data = ['electricity', 'water_heat', 'space_heat']
storage_capacity_data = ['battery_private', 'heat_storage']
inst_technology_data = ['heatpump_air']

# Define wheter or not to include previous datasteps (t-n) in each row for timeseries data

add_previous_timesteps = False
number_previous_timesteps = 8

# Data-Frame to store whole Dataset
complete_data_forchheim = pd.DataFrame()

# Load desired data from h5 files into Dataframe iterating over all scenarios
for file_name in h5_files:
    fullinname=os.path.join('./scenario_data' + '/Forchheim' + '/flex_all_{}.h5'.format(file_name))
    if Path(fullinname).is_file():
        hdf = load(fullinname)
        print("{} FOUND".format(file_name))
    else: 
        print("{} MISSING".format(file_name))
        continue

    data = hdf._data
    results = hdf._result
    
    # Data-Frame to store data from the current scenario
    scenario_data = pd.DataFrame()
    
    #load demand data defined above
    for demand in demand_data:
        label = demand + '_demand'
        scenario_data[label] = data['demand'].T[data['demand'].keys().droplevel(level = 0) == demand].sum()[1:]
        
        # If desired, add previous timestep (t-n) data for timeseries data in each row
        if add_previous_timesteps:
            for idx in range(number_previous_timesteps):
                new_label = label + ' [t-' + str((idx+1)) + ']'
                scenario_data[new_label] = scenario_data[label].shift(idx+1, fill_value = 0.0)
                  
    #load mobility demand aggregated over all vehicles and buildings
    label = 'mobility_demand'
    mobility_columns = [(site, commodity) for (site, commodity) in data['demand'].columns if commodity.startswith('mobility')]
    scenario_data[label] = data['demand'][mobility_columns].sum(axis=1).values[1:]
    
    # If desired, add previous timestep (t-n) data for timeseries data in each row
    if add_previous_timesteps:
            for idx in range(number_previous_timesteps):
                new_label = label + ' [t-' + str((idx+1)) + ']'
                scenario_data[new_label] = scenario_data[label].shift(idx+1, fill_value = 0.0)
    
    # load storage data as defined above
    for storage_type in storage_capacity_data:
        label = storage_type + '_capacity'
        scenario_data[label] = results['cap_sto_c'].loc[:,:, storage_type].sum()
        
    # load PV data
    scenario_data['Rooftop PV_capacity'] = results['cap_pro'][results['cap_pro'].keys().get_level_values(level=2).str.startswith('Rooftop PV')].sum()
    
    # load capacities as defined above 
    for technology in inst_technology_data:
        label = technology + '_capacity'
        scenario_data[label] = results['cap_pro'].loc[:,:, technology].sum() if technology in results['cap_pro'].keys().droplevel(level = [0,1]) else 0.0
        
        
      
    # add solar irradiation weighted in relation to relative installed PV capacity at building and orientation:
  
    # filter for cap_pro starting with Rooftop PV to get all orientations
    pv = results['cap_pro'][results['cap_pro'].keys().get_level_values(level=2).str.startswith('Rooftop PV')]
    
    #remove unnecessary indices
    pv = pv.reset_index(level=0, drop = True)
    pv = pv.reset_index(level=1)
    
    #slice 'Rooftop PV' from index and add '_solar' to fit to solar irradiation index
    pv['pro'] = pv['pro'].str.slice(11)
    pv.index = pv.index + 'solar_' + pv['pro'].str[:]
    
    #drop orientation column 
    pv = pv.drop('pro', axis=1).sort_index()
    
    #compute capacity relative to total installed capacity per building and orientation
    relative_pv = pv.div(pv.sum())   
    
    #remove unnecessary index
    radiation = data['supim'].reset_index(level=0, drop=True).T
    
    #merge indices to fit to PV
    radiation = radiation.reset_index(level=1, drop = False)
    radiation.index = radiation.index + radiation['level_1'].str[:]
    
    #replace ; by . in index
    radiation.index = radiation.index.str.replace(';', '.')
    
    #drop orientation column
    radiation = radiation.drop('level_1', axis=1)
    
    #filter entries that fit the pv indices
    radiation = radiation.loc[radiation.index.intersection(pv.index)].sort_index()
    
    #scale radiation per building and orientation with the relative pv capacity
    scenario_data['solar_irradiation'] = radiation.mul(relative_pv.values).sum()[1:].values

    
    
    # add mean cop relative to installed heatpump capacity at each building
    heatpump_capacity = results['cap_pro'][:,:,'heatpump_air']
    relative_capacity = heatpump_capacity.div(heatpump_capacity.sum())
    cop = data['eff_factor'].T[data['eff_factor'].keys().get_level_values(level=1).str.startswith('heatpump')]
    scenario_data['relative_cop'] = cop.T.mul(relative_capacity.values).T.sum()
    
    
    # add availability of cars relative to total charging capacity (213 charging stations available)
    scenario_data['charging_availability'] = data['eff_factor'].T[data['eff_factor'].keys().get_level_values(level=1).str.startswith('charging_station')].sum()/213
    
    # add net demand as import - export
    scenario_data['net_demand'] = results['e_pro_out'].loc[:,:,:,'import'].values - results['e_pro_out'].loc[:,:,:, 'feed_in'].values

    # concatenate data from current scenario to dataframe for complete dataset
    complete_data_forchheim = pd.concat([complete_data_forchheim, scenario_data])


complete_data_original.reset_index(drop=True, inplace = True)
complete_data_forchheim.reset_index(drop=True, inplace = True)

compression_opts_original = dict(method='zip', archive_name='energydata_original.csv')
compression_opts_forchheim = dict(method='zip', archive_name='energydata_forchheim.csv')

complete_data_original.to_csv('energydata_original.zip', index=False, compression=compression_opts_original)
complete_data_forchheim.to_csv('energydata_forchheim.zip', index=False, compression=compression_opts_forchheim)