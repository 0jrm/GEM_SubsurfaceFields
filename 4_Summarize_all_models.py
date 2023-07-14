import os
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# AI common
import sys
sys.path.append("ai_common_torch")

# EOAS Utils
sys.path.append("eoas_pyutils")
from eoas_pyutils.io_utils.io_common import create_folder

NET = "Network Type"
LOSS  = "Loss value"

# Read folders for all the experiments
trained_models_folder = ""
output_folder = ""

all_folders = os.listdir(trained_models_folder)
all_folders.sort()

experiments = []

# Iterate over all the experiments
for experiment in all_folders:
    models_folder = join(trained_models_folder, experiment , "models")
    if os.path.exists(models_folder):
        print(F"Working with experiment: {experiment}")
        all_models = os.listdir(models_folder)
        min_loss = 100000.0
        best_model = {}
        # Iterate over the saved models for each experiment and obtain the best of them
        for model in all_models:
            model_split = model.split("-")
            loss = float((model_split[-1]).replace(".hdf5",""))
            if loss < min_loss:
                min_loss = loss
                id = model_split[0]
                name = model_split[1]
                network_type = model_split[2]
                loss = np.around(min_loss,5)
                path = join(trained_models_folder, experiment, "models", model)
                best_model = [id, experiment, network_type, loss, path]
        experiments.append(best_model)

# Build a dictionary from data
df = {
    "ID": [x[0] for x in experiments],
    "Name": [x[1] for x in experiments],
    "Network Type": [x[2] for x in experiments],
    "Loss value": [x[3] for x in experiments],
    "Path": [x[4] for x in experiments],
}
summary = pd.DataFrame.from_dict(df)
print(F"Models summary: {summary}")

create_folder(output_folder)
summary.to_csv(join(output_folder,"summary.csv"))