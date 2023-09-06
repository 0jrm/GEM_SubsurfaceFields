# %% External
import os
from pandas import DataFrame
import pandas as pd
import time
from os.path import join
import numpy as np
import xarray as xr
import torch
import pickle
import sys

# AI common
# sys.path.append("ai_common_torch")
from ai_common.constants.AI_params import TrainingParams, ModelParams
from ai_common.models.modelSelector import select_2d_model
import ai_common.training.trainingutils as utilsNN
from proj_ai.proj_models import BasicDenseModel

# EOAS Utils
# sys.path.append("eoas_pyutils")
from eoas_pyutils.io_utils.io_common import create_folder, all_files_in_folder
from eoas_pyutils.viz_utils.eoa_viz import EOAImageVisualizer

"""
:param config:
:return:
"""
# *********** Reads the parameters *********** TODO: get paths automatically
output_folder = "/home/jmiranda/SubsurfaceFields/training"
summary_file = join(output_folder, "summary/summary.csv")
df = pd.read_csv(summary_file)

save_predictions = False
save_imgs = True

# # load model parameters from the file **** TODO: get paths automatically
# mod_param = "/home/jmiranda/SubsurfaceFields/training/BasicDenseModel_NoPCA_2023-08-23_09:10/model_params.pkl"
# input_size, output_size, hidden_layers, neurons_per_layer, activation_hidden, activation_output, batch_norm = pickle.load(open(mod_param, "rb"))

# # declare and load model
# model = BasicDenseModel(input_size, output_size, hidden_layers, neurons_per_layer, 
#                       activation_hidden, activation_output, batch_norm)
# model.load_state_dict(torch.load("/home/jmiranda/SubsurfaceFields/training/BasicDenseModel_NoPCA_2023-08-23_09:10/models/BasicDenseModel_NoPCA_2023-08-23_09:10_epoch:037_0.00002671.pt"))

# # loaded_original_model = torch.load('/home/jmiranda/SubsurfaceFields/training/BasicDenseModel_WithPCA_2023-08-21_11:50/BasicDenseModel_WithPCA_2023-08-21_11:50.pkl')
# # loaded_best_model = torch.load('/home/jmiranda/SubsurfaceFields/training/BasicDenseModel_NoPCA_2023-08-21_16:31/models/BasicDenseModel_NoPCA_2023-08-21_16:31_epoch:039_0.00002763.pt')
# # model = select_2d_model() # not needed if loading the model directly

# %% Iterates over all the models in the file
for model_id in range(len(df)):
    mod_series = df.iloc[model_id]
    model_name = mod_series["Name"]
    model_weights_file = mod_series["Path"]
    this_folder = join(output_folder, model_name)
    mod_param = join(this_folder, "model_params.pkl")
    mod_data = join(this_folder, "train_val.pkl")
    input_size, output_size, hidden_layers, neurons_per_layer, activation_hidden, activation_output, batch_norm = pickle.load(open(mod_param, "rb"))
    train_dataset, val_dataset = pickle.load(open(mod_data, "rb"))
    model = BasicDenseModel(input_size, output_size, hidden_layers, neurons_per_layer, 
                      activation_hidden, activation_output, batch_norm)
    model.load_state_dict(torch.load(model_weights_file))
    
    print("shape pf train_dataset:", train_dataset.shape)
    print("shape pf val_dataset:", val_dataset.shape)
    
    
    # TODO: test and validation

    print(F"Model model_name: {model_name}")

    # c_output_folder =join(output_folder,model_name)
    # create_folder(c_output_folder)
    # print(c_output_folder)
    # # *********** Chooses the proper model ***********
    # print('Reading model ....')
    # model = select_2d_model()

    # # *********** Reads the weights***********
    # print('Reading weights ....')
    # model.load_weights(model_weights_file)
    
    # WHAT TESTS DO i WANT TO DO? WHAT VISUALIZATIONS, METRICS?
    # WHAT FUNCTIONS DO I NEED TO WRITE? WHICH ONES ARE ALREADY WRITTEN I CAN USE?

    # **************** Split definition *****************
    [train_ids, val_ids, test_ids] = utilsNN.split_train_validation_and_test(tot_examples,
                                                                             val_percentage=val_perc,
                                                                             test_percentage=test_perc)

    # Working only with the test indexes
    for c_id in range(len(test_ids)):
        file_id = test_ids[c_id]
        # TODO here you could test in batches also
        X, Y = generateXY(hr_path=hr_paths[file_id],
                          lr_path=lr_paths[file_id],
                          inc_factor=inc_factor,
                          in_dims=in_dims)

        X = np.expand_dims(X, axis=0)
        nn_raw_output = model.predict(X, verbose=1)
        #
        lr_ds = xr.open_dataset(lr_paths[file_id])
        lr_ds_crop = lr_ds.isel(Latitude=slice(0, in_dims[0]), Longitude=slice(0, in_dims[1]), MT=0)  # Cropping by value
        lats = lr_ds_crop.Latitude
        lons = lr_ds_crop.Longitude
        vizobj = EOAImageVisualizer(disp_images=False, output_folder=c_output_folder, lats=[lats],lons=[lons], show_var_names=True)
        vizobj.plot_2d_data_np(np.swapaxes(X[0], 0, 2), ['U', 'V'], F'X_{lr_names[file_id]}_lr', file_name_prefix=F'X_{lr_names[file_id]}_lr', flip_data=True, rot_90=True)
        vizobj.plot_2d_data_np(np.swapaxes(nn_raw_output[0], 0, 2), ['U', 'V'], F'Y_{hr_names[file_id]}_hr', file_name_prefix=F'Y_{hr_names[file_id]}_hr', flip_data=True, rot_90=True)
# %%
