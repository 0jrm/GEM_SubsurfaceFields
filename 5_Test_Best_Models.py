# External
import os
from pandas import DataFrame
import pandas as pd
import time
from os.path import join
import numpy as np
import xarray as xr

# Common
import sys
# AI common
sys.path.append("ai_common_torch")
from ai_common.constants.AI_params import TrainingParams, EvaluationParams, ModelParams
from ai_common.models.modelSelector import select_2d_model
import ai_common.training.trainingutils as utilsNN
# EOAS Utils
sys.path.append("eoas_pyutils")
from eoas_pyutils.viz_utils.eoa_viz import EOAImageVisualizer
from eoas_pyutils.io_utils.io_common import create_folder, all_files_in_folder

"""
:param config:
:return:
"""
# *********** Reads the parameters ***********
output_folder = ""
summary_file = join(output_folder, "summary.csv")
df = pd.read_csv(summary_file)

save_predictions = False
save_imgs = True

# Iterates over all the models in the file
for model_id in range(len(df)):
    model = df.iloc[model_id]
    model_name = model["Name"]
    model_weights_file = model["Path"]
    print(F"Model model_name: {model_name}")

    c_output_folder =join(output_folder,model_name)
    create_folder(c_output_folder)

    # *********** Chooses the proper model ***********
    print('Reading model ....')
    model = select_2d_model()

    # *********** Reads the weights***********
    print('Reading weights ....')
    model.load_weights(model_weights_file)

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