# External
import sys
from datetime import datetime
import xarray as xr
import numpy as np
from multiprocessing import Pool
from shapely.geometry import Polygon
import os
from os.path import join
import h5py
# EOAS Library
sys.path.append("eoas_pyutils")
from io_utils.io_common import create_folder
from viz_utils.eoa_viz import EOAImageVisualizer
# Local
from proj_io.contours import read_contours_mask_and_polygons

# %%  This code should process/clean you data before training, using functions at proj_proc
input_folder = ""
output_folder = ""

def preproc_data(proc_id):
    '''Here we preprocess the data before training'''
    create_folder(output_folder)
    all_files = os.listdir(input_folder)
    all_files.sort()
    for i, c_file in enumerate(all_files):
        if i % NUM_PROC == proc_id:
            file_name = join(input_folder, c_file)
            print(f"Processing file {file_name}")
            # Save preprocessed file

if __name__ == '__main__':
    # ----------- Parallel -------
    NUM_PROC = 20
    p = Pool(NUM_PROC)
    p.map(preproc_data, range(NUM_PROC))