# %%
# External
import os
from pandas import DataFrame
import pandas as pd
import time
from os.path import join
import numpy as np
import xarray as xr
# Torch
import torch
from torch.utils.data import DataLoader

# Common
import sys
# AI common
sys.path.append("ai_common_torch")
from proj_ai.Training import train_model
from proj_io.Generators import ProjDataset, ProjDatasetPCA
from proj_ai.proj_models import BasicDenseModel
from configs.RunConfig import RunConfig

# EOAS Utils
sys.path.append("eoas_pyutils")
from eoas_pyutils.viz_utils.eoa_viz import EOAImageVisualizer
from eoas_pyutils.io_utils.io_common import create_folder, all_files_in_folder

# Proj
from proj_viz.argo_viz import plot_single_ts_profile, compare_profiles

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
:param config:
:return:
"""
# *********** Reads the parameters ***********
output_folder = "/unity/f1/ozavala/OUTPUTS/GEM_SubSurface/outputs"
summary_folder = "/unity/f1/ozavala/OUTPUTS/GEM_SubSurface/summary"
summary_file = join(summary_folder, "summary.csv")
data_folder = RunConfig.data_folder.value
df = pd.read_csv(summary_file)

save_predictions = False
save_imgs = True

temp_componets = 10
sal_components = 10
seed = 42
workers = 20

with_pca = True
if with_pca:
    dataset = ProjDatasetPCA(data_folder, temp_components=temp_componets, 
                             sal_components=sal_components, test=True) 
else:
    dataset = ProjDataset(data_folder)

test_size = len(dataset)
batch_size = 10

print("Total number of test samples: ", test_size)

# Create DataLoaders for training and validation
test_loader = DataLoader(dataset, batch_size=batch_size,  
                        shuffle=False, num_workers=workers)

#%% Initialize the model, loss, and optimizer
inout_dims = dataset.get_inout_dims()
input_size = inout_dims[0]  
output_size = inout_dims[1]  
hidden_layers = 4
neurons_per_layer = 100
activation_hidden = 'relu'
activation_output = 'linear'
batch_norm = True
model = BasicDenseModel(input_size, output_size, hidden_layers, neurons_per_layer, 
                      activation_hidden, activation_output, batch_norm)

model.to(device)

# %%

# Iterates over all the models in the file
for model_id in range(len(df)):
    c_model = df.iloc[model_id]
    model_name = c_model["Name"]
    model_weights_file = c_model["Path"]
    print(F"Model model_name: {model_name}")

    c_output_folder =join(output_folder,model_name)
    create_folder(c_output_folder)

    # *********** Chooses the proper model ***********
    print('Reading model ....')

    # *********** Reads the weights***********
    print('Reading weights ....')
    model.load_state_dict(torch.load(model_weights_file))
    # model.eval()  # Why?

    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)

        # We do inverse pca on each sal and temp
        for idx, c_profile in enumerate(output[:10,:]):
            print(c_profile.shape)

            c_temp = c_profile[:temp_componets].to('cpu').detach().numpy()
            c_sal = c_profile[temp_componets:].to('cpu').detach().numpy()
            t, s = dataset.inverse_pca(c_temp, c_sal)

            # Raw/Original profiles
            raw_t, raw_s = dataset.get_original_profile(idx)

            # Target profiles
            target_t_pca = target[idx,:temp_componets].to('cpu').detach().numpy()
            target_sal_pca = target[idx,temp_componets:].to('cpu').detach().numpy()

            target_t, target_s = dataset.inverse_pca(target_t_pca, target_sal_pca)

            # plot_single_ts_profile(raw_t, raw_s,  depths=range(0, -2001, -1), title=f'Profile_{idx}',
                        #    labelone='Raw T', labeltwo='Raw S')

            # plot_single_ts_profile(target_t, target_s,  depths=range(0, -2001, -1), title=f'Profile_{idx}',
                        #    labelone='Temperature', labeltwo='Salinity')

            compare_profiles(raw_t, target_t,  depths=range(0, -2001, -1), title=f'Profile_{idx}',
                           labelone='Raw', labeltwo='Target', figsize=10)

            compare_profiles(target_t, t,  depths=range(0, -2001, -1), title=f'Profile_{idx}',
                           labelone='Target T', labeltwo='NN T', figsize=10)

            compare_profiles(target_s, s,  depths=range(0, -2001, -1), title=f'Profile_{idx}',
                           labelone='Target S', labeltwo='NN S', figsize=10)
            if idx > 5:
                break

        break
# %%
