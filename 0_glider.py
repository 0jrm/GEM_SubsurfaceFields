# %%
import numpy as np
import xarray as xr
import os
import sys
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from shapely.geometry import Point
from scipy.interpolate import RegularGridInterpolator

from proj_viz.argo_viz import plot_single_ts_profile, compare_profiles
sys.path.append("/home/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/eoas_pyutils/")
from io_utils.coaps_io_data import get_aviso_by_date, get_sst_by_date, get_sss_by_date
# from io_utils.io_netcdf import read_netcdf_xr, read_multiple_netcdf_xarr # read_multiple not working, using my own
from viz_utils.eoa_viz import select_colormap, EOAImageVisualizer
from viz_utils.eoas_viz_3d import ImageVisualizer3D
from viz_utils.constants import PlotMode

processed = True

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
    datasets = [xr.open_dataset(os.path.join(path, file), decode_times=False)[variables] for file in files[:num_files] if file.endswith('.nc')]
    ds_combined = xr.concat(datasets, dim='time')
    df_clean = ds_combined.to_dataframe().dropna()
    df_clean.index = pd.to_datetime(df_clean.index, unit='s')
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
    variables, titles, ylabels, colormaps = ['temperature', 'salinity', 'density'], ['Temperature Profile', 'Salinity Profile', 'Density Profile'], ['Temperature (°C)', 'Salinity (psu)', 'Density (kg/m³)'], ['RdBu_r', 'Blues', 'viridis']
    for i, var in enumerate(variables):
        axs[i].scatter(df_day[var], df_day['depth'], alpha=0.5, s=1)
        axs[i].set_title(f'{titles[i]} on {day}')
        axs[i].set_xlabel(ylabels[i])
        axs[i].set_ylabel('Depth (m)')
        axs[i].invert_yaxis()
        axs[i].grid(True)
    plt.tight_layout()
    plt.show()


# %%
# Data Loading and Cleaning
path = "/home/jmiranda/SubsurfaceFields/Data/LCE_Poseidon/"
nc_files = [file for file in os.listdir(path) if file.endswith('.nc')]
variables = ['pressure', 'depth', 'time', 'lat', 'lon', 'salinity', 'temperature', 'density']
if processed:
    ds = pd.read_pickle("/home/jmiranda/SubsurfaceFields/Data/glider_processed.pkl")
else:
    ds = load_join_clean(path, nc_files, variables, num_files=len(nc_files))


# %%
# Separate profiles, get position for each profile
ds['pres_diff'] = ds['pressure'].diff()
ds['direction'] = np.where(ds['pres_diff'] > 0, 'descent', 'ascent')
ds['profile_id'] = (ds['direction'] != ds['direction'].shift()).cumsum()
position = ds.groupby('profile_id')[['lat', 'lon']].first()

# AVISO Data
aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
bbox = [17.5, 32.5, -110, -80]
start_date = datetime.datetime(2016, 8, 6)
end_date = datetime.datetime(2016, 9, 5)
delta = datetime.timedelta(days=1)
date_of_interest = start_date

def get_prof_ssh_from_aviso(lats, lons, aviso_data, grid_lats, grid_lons):
    """Interpolates AVISO altimetry to profile locations.

    Parameters:
    lats (array-like): Latitudes of the profile locations.
    lons (array-like): Longitudes of the profile locations.
    aviso_data (array-like): AVISO altimetry data on a regular grid.
    grid_lats (array-like): Latitudes of the regular grid.
    grid_lons (array-like): Longitudes of the regular grid.

    Returns:
    numpy.ndarray: Interpolated altimetry values at the profile locations.
    """
    interpolator = RegularGridInterpolator((grid_lats, grid_lons), aviso_data.values, bounds_error=False, fill_value=None)
    coordinates = np.array([lats, lons]).T
    return interpolator(coordinates)


# %%
# Initialize an empty DataFrame to store additional data
additional_data = []

# Get AVISO, plot daily profiles
while date_of_interest <= end_date:
    date_string = date_of_interest.strftime("%Y-%m-%d")
    aviso_data, lats, lons = get_aviso_by_date(aviso_folder, date_of_interest, bbox)
    profiles_on_date = ds[ds.index.date == date_of_interest.date()]['profile_id'].unique()
    position_on_date = position.loc[profiles_on_date]
    prof_lat, prof_lon = position_on_date['lat'].values, position_on_date['lon'].values
    profile_ssh = get_prof_ssh_from_aviso(prof_lat, prof_lon, aviso_data.adt, lats, lons)
    
    plot_points = [Point(lon, lat) for lat, lon in zip(prof_lat, prof_lon)]
    viz_obj = EOAImageVisualizer(lats=lats, lons=lons, disp_images=True, output_folder="outputs", show_var_names=True)
    viz_obj.__setattr__('plot_pointsgons', plot_points)
    
    viz_obj.plot_2d_data_np(aviso_data.adt, ['adt'], f'- glider profiles on {date_string}', 'filepref')
    print("date\t\tlat\t\t\tlon\t\t\tAVISO SSH")
    for lat, lon, altimetry_value in zip(prof_lat, prof_lon, profile_ssh):
        print(f"{date_string}\t{lat}\t{lon}\t{altimetry_value}")
    plot_profiles(ds, date_of_interest.date())
    
     # Create a temporary DataFrame with profile_id, lat, lon, and AVISO SSH
    temp_data = pd.DataFrame({
        'profile_id': profiles_on_date,
        'lat_position': prof_lat,
        'lon_position': prof_lon,
        'AVISO_SSH': profile_ssh
    })

    for lat, lon, ssh, profile_id in zip(prof_lat, prof_lon, profile_ssh, profiles_on_date):
        additional_data.append({
            'profile_id': profile_id,
            'lat_position': lat,
            'lon_position': lon,
            'AVISO_SSH': ssh
        })
    
    date_of_interest += delta

additional_data_df = pd.DataFrame(additional_data)

# %%
# Merge the additional data with the existing DataFrame using profile_id as the key
ds = ds.merge(additional_data_df, on='profile_id', how='left')


# %% Save processed data
ds.to_pickle("/home/jmiranda/SubsurfaceFields/Data/glider_processed.pkl")
# %%