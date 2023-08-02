# %% DONE!
import numpy as np
from netCDF4 import Dataset
import xarray as xr
import os
import pandas as pd
import dask.dataframe as dd
import datetime
import matplotlib.pyplot as plt
import sys
from shapely.geometry import LineString, Polygon, MultiPoint, Point
import gsw

sys.path.append("/home/jmiranda/GEM_SubsurfaceFields/torch_folder/eoas-pyutils/")
from io_utils.coaps_io_data import get_aviso_by_date, get_sst_by_date, get_sss_by_date
# from io_utils.io_netcdf import read_netcdf_xr, read_multiple_netcdf_xarr # read_multiple not working, using my own
from viz_utils.eoa_viz import select_colormap, EOAImageVisualizer
from viz_utils.eoas_viz_3d import ImageVisualizer3D
from viz_utils.constants import PlotMode

def load_join_clean(path, files, variables, num_files=None):
    """Load, joins and remove NaNs from multiple NetCDF files.

    Parameters:
    path (str): The path to the directory containing the files.
    files (list of str): The list of files to load.
    variables (list of str): The variables to extract from each file.
    num_files (int, optional): The maximum number of files to load.

    Returns:
    pandas.DataFrame: The concatenated clean dataset as a DataFrame.
    """
    # Initialize an empty list to store the datasets
    datasets = []

    # Loop over the files
    for file in files[:num_files]:  # Limit the number of files if specified
        # Only process .nc files
        if not file.endswith('.nc'):
            continue

        # Define the file path
        file_path = os.path.join(path, file)

        # Open the dataset without decoding times
        ds = xr.open_dataset(file_path, decode_times=False)

        # Extract the variables of interest
        ds = ds[variables]

        # Append the dataset to the list
        datasets.append(ds)

    # Concatenate all datasets
    ds_combined = xr.concat(datasets, dim='time')

    # Convert the dataset to a DataFrame
    df = ds_combined.to_dataframe()

    # Remove rows with missing values
    df_clean = df.dropna()
    
    # Convert the index to a DatetimeIndex
    df_clean.index = pd.to_datetime(df_clean.index, unit='s')

    # Extract the date for each measurement
    df_clean['date'] = df_clean.index.date

    return df_clean

def plot_profiles(df, day):
    """Plot temperature, salinity, and density profiles for a specific day.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    day (datetime.date): The day to plot profiles for.
    """
    # Filter the DataFrame for the chosen day
    df_day = df[df['date'] == day]

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    variables = ['temperature', 'salinity', 'density']
    titles = ['Temperature Profile', 'Salinity Profile', 'Density Profile']
    ylabels = ['Temperature (°C)', 'Salinity (psu)', 'Density (kg/m³)']
    colormaps = ['RdBu_r', 'Blues', 'viridis']
    
    for i, var in enumerate(variables):
        sc = axs[i].scatter(df_day[var], df_day['depth'], alpha=0.5, s=1)
        axs[i].set_title(f'{titles[i]} on {day}')
        axs[i].set_xlabel(ylabels[i])
        axs[i].set_ylabel('Depth (m)')
        axs[i].invert_yaxis()

    plt.tight_layout()
    plt.show()

# %% 
# Define the path to the directory containing the files
path = "/home/jmiranda/GEM_SubsurfaceFields/input_data/LCE_Poseidon/"
# Get a list of all files and directories in the folder
all_files = os.listdir(path)
# Filter only the files with the .nc extension
nc_files = [file for file in all_files if file.endswith('.nc')]

variables = ['pressure', 'depth', 'time', 'lat', 'lon', 'salinity', 'temperature', 'density']

# Load and join data
ds = load_join_clean(path, nc_files, variables, num_files=len(nc_files))

# Plot profiles for a specific day
day = pd.to_datetime('2016-08-06').date()  # Modify this to the day you're interested in
plot_profiles(ds, day)

# %%
# Calculate the pressure difference for each row compared to the previous row
ds['pres_diff'] = ds['pressure'].diff()

# Determine whether each row is part of an ascent or descent
ds['direction'] = np.where(ds['pres_diff'] > 0, 'descent', 'ascent')

# Assign a unique profile ID to each ascent/descent
ds['profile_id'] = (ds['direction'] != ds['direction'].shift()).cumsum()

# Get median latitude and longitude for each profile
medians = ds.groupby('profile_id')[['lat', 'lon']].median()

# Get the global minimum and maximum values for temperature, salinity, and depth
temp_min, temp_max = ds['temperature'].min()-1, ds['temperature'].max()+1
salinity_min, salinity_max = ds['salinity'].min()-1, ds['salinity'].max()+1
depth_min, depth_max = ds['depth'].min(), ds['depth'].max()+50

# Function to plot individual profiles
def plot_individual_profiles(df, profile_id):
    df_profile = df[df['profile_id'] == profile_id]
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    
    variables = ['temperature', 'salinity']
    titles = ['Temperature Profile', 'Salinity Profile']
    ylabels = ['Temperature (°C)', 'Salinity (psu)']
    x_lims = [(temp_min, temp_max), (salinity_min, salinity_max)]
    colors = ['r', 'b']
    
    for i, var in enumerate(variables):
        axs[i].plot(df_profile[var], df_profile['depth'], color=colors[i])
        axs[i].set_title(f'{titles[i]} for Profile ID {profile_id}')
        axs[i].set_xlabel(ylabels[i])
        axs[i].set_ylabel('Depth (m)')
        axs[i].set_xlim(x_lims[i])
        axs[i].set_ylim(depth_max, depth_min) # Inverted to display depth correctly
        # axs[i].invert_yaxis()
    
    plt.tight_layout()
    plt.show()

# # Test the function with a specific profile_id
# for i in range(max(ds['profile_id'])):
#     plot_individual_profiles(ds, i+1)

# %% figure out points candidates (median of the track)

# load aviso data (2016-08-06 - 2016-09-05) 
aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
bbox = [17.5, 32.5, -110, -80]
x = datetime.datetime(2016, 8, 6)
# test. this should be done in a
aviso_data, lats, lons = get_aviso_by_date(aviso_folder, x, bbox)
print(aviso_data["time"]) # check date

# Filter the medians DataFrame for profiles from the desired date
date_of_interest = x.date()  # x is the date used to load the AVISO data
profiles_on_date = ds[ds.index.date == date_of_interest]['profile_id'].unique()
medians_on_date = medians.loc[profiles_on_date]

# TO-DO: Plot the AVISO data AND the location of the profiles
# %%
viz_obj = EOAImageVisualizer(lats=lats, lons=lons, disp_images=True, output_folder="outputs",  show_var_names=True, eoas_pyutils_path=".")
viz_obj.plot_2d_data_np(aviso_data.adt, ['adt'], 'LC', 'filepref', plot_mode=PlotMode.RASTER)

# # Plot the locations of the profiles
# Assuming medians_on_date contains the 'lat' and 'lon' data
latitudes = medians_on_date['lat']
longitudes = medians_on_date['lon']
# %%

# Create the multipoint variable by combining latitudes and longitudes
# mymultipoint = MultiPoint(list(zip(latitudes, longitudes)))
mymultipoint = MultiPoint(list(zip(latitudes, longitudes)))
# mymultipoint = MultiPoint((latitudes[i], longitudes[i]) for i in range(latitudes.size))

# %%
viz_obj.__setattr__('additional_polygons', mymultipoint)
viz_obj.plot_2d_data_np(aviso_data.adt, ['adt'], 'With external polygon', 'filepref', plot_mode=PlotMode.CONTOUR)


# # # Add the steric height as text above each point
# # fig, ax = plt.gca()  # Get the current figure and axes
# # for idx, row in medians_on_date.iterrows():
# #     ax.text(row['lon'], row['lat'], f"{row['steric_height']:.2f}", fontsize=9, ha='center', va='bottom')



# %% Check if correct
def calculate_steric_height(dataset):
    # Calculate Absolute Salinity (g/kg)
    SA = gsw.SA_from_SP(dataset['salinity'], dataset['pressure'], 0, dataset['latitude'])
    
    # Calculate Conservative Temperature (°C)
    CT = gsw.CT_from_t(SA, dataset['temperature'], dataset['pressure'])

    # Calculate specific volume anomaly
    delta_v = gsw.specvol_anom_standard(SA, CT, dataset['pressure'])

    # Integrate specific volume anomaly from the bottom to the surface to get dynamic height
    dynamic_height = gsw.geo_strf_dyn_height(SA, CT, dataset['pressure'], p_ref=0)

    # Convert dynamic height to steric height (m)
    steric_height = dynamic_height / 9.81

    return steric_height

