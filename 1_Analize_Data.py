# %% 
import numpy as np
import mat73
from os.path import join
from proj_viz.argo_viz import plot_ts_profiles, plot_profiles_sorted_by_SH1950

from configs.RunConfig import RunConfig

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Disable GPU

# %%  Read the data
input_folder = RunConfig.data_folder.value

# ADT_1 = mat73.loadmat(          'ADT_20220920.mat'          , use_attrdict=True) 
# ADT_noseason = mat73.loadmat(   'ADT_noseason_20220920.mat' , use_attrdict=True) 
ARGO = mat73.loadmat( join(input_folder,'ARGO_GoM_20220920.mat')     , use_attrdict=True) 
# LoopCur = loadmat(              'LoopCurrentRings_edges.mat'    )

# %% All fields
# Separate and display data from ARGO file
print('ADT_loc', np.shape(ARGO.ADT_loc))# ADT_loc (4890,)
print('ADTnoseason_loc', np.shape(ARGO.ADTnoseason_loc)) # ADTnoseason_loc (4890,)
print('LAT', np.shape(ARGO.LAT))        # LAT (4890,)
print('LON', np.shape(ARGO.LON))        # LON (4890,)
print('PRES', np.shape(ARGO.PRES))      # PRES (2001, 4890)
print('RHO', np.shape(ARGO.RHO))        # RHO (2001, 4890)
print('SAL', np.shape(ARGO.SAL))        # SAL (2001, 4890)
print('SH1950', np.shape(ARGO.SH1950))  # SH1950 (4890,)
print('SIG', np.shape(ARGO.SIG))        # SIG (2001, 4890)
print('SPICE', np.shape(ARGO.SPICE))    # SPICE (2001, 4890)
print('TEMP', np.shape(ARGO.TEMP))      # TEMP (2001, 4890)
print('TIME', np.shape(ARGO.TIME))      # TIME (4890,)

# %% Plot some profiles
start_profile = 10
plot_ts_profiles(ARGO.TEMP, ARGO.SAL, start_profile)

# %% Plot all profiles aligned by depth
# Sort the temperature profiles by SH1950 values
plot_profiles_sorted_by_SH1950(ARGO)

# %%
