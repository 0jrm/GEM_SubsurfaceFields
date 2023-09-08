# %%
import numpy as np
import xarray as xr
import os
import sys
import pandas as pd
import datetime
import matplotlib.pyplot as plt
# import seawater as sw
from shapely.geometry import Point
from scipy.interpolate import RegularGridInterpolator
from proj_viz.argo_viz import plot_single_ts_profile, compare_profiles
# # git clone https://github.com/garrettdreyfus/python-holteandtalley.git
# # cd python-holteandtalley & pip install -e .
# from holteandtalley import HolteAndTalley
sys.path.append("/home/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/eoas_pyutils/")
from io_utils.coaps_io_data import get_aviso_by_date, get_sst_by_date, get_sss_by_date
# from io_utils.io_netcdf import read_netcdf_xr, read_multiple_netcdf_xarr # read_multiple not working, using my own
from viz_utils.eoa_viz import select_colormap, EOAImageVisualizer
from viz_utils.eoas_viz_3d import ImageVisualizer3D
from viz_utils.constants import PlotMode

processed =  False

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

def get_prof_ssh_from_aviso(grid_lats, grid_lons, aviso_data, grid_grid_lats, grid_grid_lons):
    """Interpolates AVISO altimetry to profile locations.

    Parameters:
    grid_lats (array-like): Latitudes of the profile locations.
    grid_lons (array-like): Longitudes of the profile locations.
    aviso_data (array-like): AVISO altimetry data on a regular grid.
    grid_grid_lats (array-like): Latitudes of the regular grid.
    grid_grid_lons (array-like): Longitudes of the regular grid.

    Returns:
    numpy.ndarray: Interpolated altimetry values at the profile locations.
    """
    interpolator = RegularGridInterpolator((grid_grid_lats, grid_grid_lons), aviso_data.values, bounds_error=False, fill_value=None)
    coordinates = np.array([grid_lats, grid_lons]).T
    return interpolator(coordinates)

def calculate_MLD(df):
    """
    Calculate Mixed Layer Depth based on density, temperature differences, and maximum curvature of T profiles.
    
    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    
    Returns:
    pandas.DataFrame: A DataFrame with columns: profile_id, MLD_density, MLD_temperature, MLD_curvature.
    """
    
    results = []

    for profile_id in df['profile_id'].unique():
        # Filter the DataFrame for the specific profile ID and sort by pressure
        tdf = df[df['profile_id'] == profile_id]
        tdf = tdf.sort_values(by='pressure')

        # Get lists of temperature, pressure, salinity, and density
        temp = tdf['temperature'].tolist()
        pres = tdf['pressure'].tolist()
        sal = tdf['salinity'].tolist()
        den = tdf['density'].tolist()

        # Find the reference density and temperature (values below 3m)
        reference_density = None
        reference_temperature = None
        for p, d, t in zip(pres, den, temp):
            if p > 3:
                reference_density = d
                reference_temperature = t
                break

        # Calculate MLD for density
        mld_density = None
        for p, d in zip(pres, den):
            if abs(d - reference_density) > 0.125:
                mld_density = p
                break
        
        # Calculate MLD for temperature
        mld_temperature = None
        for p, t in zip(pres, temp):
            if abs(t - reference_temperature) > 0.8:
                mld_temperature = p
                break

        # Calculate MLD based on the maximum curvature of T profiles
        interp_pres = np.linspace(min(pres), max(pres), 1000)
        interp_temp = np.interp(interp_pres, pres, temp)
        curvature = np.gradient(np.gradient(interp_temp, interp_pres), interp_pres)
        mld_curvature = interp_pres[np.argmax(curvature)]
        
        results.append([profile_id, mld_density, mld_temperature, mld_curvature])

    # Convert results to a DataFrame
    result_df = pd.DataFrame(results, columns=['profile_id', 'MLD_T', 'MLD_c', 'MLD_d'])
    
    return result_df

def plot_MLD_profile(df, profileNumber):
    """Plot temperature, salinity, and density profiles for a specific day.

    Parameters:
    df (pandas.DataFrame): Processed data, including.
    day (datetime.date): The day to plot profiles for.
    """
    # Filter the DataFrame for the chosen profile
    tdf = df[df['profile_id'] == profileNumber]
    temp = tdf['temperature'].tolist()
    pres = tdf['pressure'].tolist()
    sal = tdf['salinity'].tolist()
    den = tdf['density'].tolist()

    mld_names = ['MLD_T', 'MLD_c', 'MLD_d']
    if all(mld in df.columns for mld in mld_names):
        mld_tsd = [tdf['MLD_T'].iloc[0], tdf['MLD_c'].iloc[0], tdf['MLD_d'].iloc[0]]
        # print("processed")
    else:
        mld_tsd = calculate_MLD(tdf)
        mld_tsd.drop('profile_id', axis=1, inplace=True)
        mld_tsd = mld_tsd.values.tolist()[0]       
        # print("unprocessed")

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    variables, titles, ylabels, colormaps = ['temperature', 'salinity', 'density'], ['Temperature Profile', 'Salinity Profile', 'Density Profile'], ['Temperature (°C)', 'Salinity (psu)', 'Density (kg/m³)'], ['RdBu_r', 'Blues', 'viridis']
    for i, var in enumerate(variables):
        axs[i].scatter(tdf[var], tdf['depth'], alpha=0.5, s=1)
        # Add the MLD
        axs[i].axhline(y=mld_tsd[i], linestyle='--', label=f'{var} MLD')
        axs[i].set_title(f'{titles[i]} #{profileNumber}')
        axs[i].set_xlabel(ylabels[i])
        axs[i].set_ylabel('Depth (m)')
        axs[i].invert_yaxis()
        axs[i].grid(True)
    plt.tight_layout()
    plt.show()

def plot_profiles(df, day):
    """Plot temperature, salinity, and density profiles for a specific day.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    day (datetime.date): The day to plot profiles for.
    """
    # Filter the DataFrame for the chosen day
    df_day = df[df['date'] == day]
    
    mld_names = ['MLD_T', 'MLD_c', 'MLD_d']
    plot_mld = False
    if all(mld in df_day.columns for mld in mld_names):
        mld_tsd = [df_day['MLD_T'].tolist(), df_day['MLD_c'].tolist(), df_day['MLD_d'].tolist()]
        plot_mld = True
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    variables, titles, ylabels, colormaps = ['temperature', 'salinity', 'density'], ['Temperature Profile', 'Salinity Profile', 'Density Profile'], ['Temperature (°C)', 'Salinity (psu)', 'Density (kg/m³)'], ['RdBu_r', 'Blues', 'viridis']
    for i, var in enumerate(variables):
        axs[i].scatter(df_day[var], df_day['depth'], alpha=0.5, s=1)
        if plot_mld:
            max_mld = max(mld_tsd[i])
            min_mld = min(mld_tsd[i])
            axs[i].axhline(y=max_mld, linestyle='--', label=f'{var} MLD')            
            axs[i].axhline(y=min_mld, linestyle='--', label=f'{var} MLD')            
        axs[i].set_title(f'{titles[i]} on {day}')
        axs[i].set_xlabel(ylabels[i])
        axs[i].set_ylabel('Depth (m)')
        axs[i].invert_yaxis()
        axs[i].grid(True)
    plt.tight_layout()
    plt.show()
    
    
# %%
# Data Loading and Cleaning
# path = "/home/jmiranda/SubsurfaceFields/Data/subset/"
path = "/home/jmiranda/SubsurfaceFields/Data/LCE_Poseidon/"
pickle_folder = "/home/jmiranda/SubsurfaceFields/Data/glider_processed.pkl"
nc_files = [file for file in os.listdir(path) if file.endswith('.nc')]
variables = ['pressure', 'depth', 'time', 'lat', 'lon', 'salinity', 'temperature', 'density']
if processed:
    ds = pd.read_pickle(pickle_folder)
else:
    ds = load_join_clean(path, nc_files, variables, num_files=len(nc_files))
    # Separate profiles, get position for each profile
    pres_diff = ds['pressure'].diff()
    direction = np.where(pres_diff > 0, 'descent', 'ascent')
    profile_id = (direction != np.roll(direction, shift=1)).cumsum()
    cols = ['profile_id'] + [col for col in ds.columns if col not in ['direction', 'pres_diff']]
    ds = pd.DataFrame({'profile_id': profile_id, **ds}, columns=cols)
    ds.loc[ds['profile_id'] == 0, 'profile_id'] = 1 # Fix the first profile_id


# %%
mld_table = calculate_MLD(ds)# %% MLD
plot_MLD_profile(ds,2)


# %%
# AVISO Data
aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
sst_folder = "/unity/f1/ozavala/DATA/GOFFISH/SST/OISST"
bbox = [17.5, 32.5, -110, -80]
start_date = datetime.datetime(2016, 8, 6)
end_date = datetime.datetime(2016, 9, 5)
delta = datetime.timedelta(days=1)
position = ds.groupby('profile_id')[['lat', 'lon']].first()


# %%
# Get AVISO, plot daily profiles
date_of_interest = start_date
# Initialize an empty DataFrame to store additional data
additional_data = []

while date_of_interest <= end_date:
    date_string = date_of_interest.strftime("%Y-%m-%d")
    aviso_data, grid_lats, grid_lons = get_aviso_by_date(aviso_folder, date_of_interest, bbox)
    if date_of_interest == datetime.datetime(2016, 8, 12, 0, 0):
        sst_data, grid_lats_sst, grid_lons_sst = get_sst_by_date(sst_folder, date_of_interest-delta, bbox)
    else:
        sst_data, grid_lats_sst, grid_lons_sst = get_sst_by_date(sst_folder, date_of_interest, bbox)
    
    profiles_on_date = ds[ds['date'] == date_of_interest.date()]['profile_id'].unique()
    position_on_date = position.loc[profiles_on_date]
    viz_obj = EOAImageVisualizer(lats=grid_lats, lons=grid_lons, disp_images=True, output_folder="outputs", show_var_names=True)
    prof_lat = position_on_date['lat'].values
    prof_lon = position_on_date['lon'].values
    profile_ssh = get_prof_ssh_from_aviso(prof_lat, prof_lon, aviso_data.adt, grid_lats, grid_lons)
    profile_sst = get_prof_ssh_from_aviso(prof_lat, prof_lon, sst_data.analysed_sst[0], grid_lats_sst, grid_lons_sst)
    plot_points = [Point(lon, lat) for lat, lon in zip(prof_lat, prof_lon)]
    viz_obj.__setattr__('additional_polygons', plot_points)
    viz_obj.plot_2d_data_np(aviso_data.adt, ['adt'], f'- glider profiles on {date_string}', 'filepref')
   
    print("date\t\tlat\t\t\tlon\t\t\tAVISO SSH")
    for lat, lon, altimetry_value in zip(prof_lat, prof_lon, profile_ssh):
        print(f"{date_string}\t{lat}\t{lon}\t{altimetry_value}")
    plot_profiles(ds, date_of_interest.date())
    
     # TODO THESE VALUES ARE BEING SAVED AS NANs
     # Create a temporary DataFrame with profile_id, lat, lon, and AVISO SSH
    temp_data = pd.DataFrame({
        'profile_id': profiles_on_date,
        'lat_position': prof_lat,
        'lon_position': prof_lon,
        'AVISO_SSH': profile_ssh,
        'SST' : profile_sst
    })

    for lat, lon, ssh, sst, profile_id in zip(prof_lat, prof_lon, profile_ssh, profile_sst, profiles_on_date):
        additional_data.append({
            'profile_id': profile_id,
            'lat_position': lat,
            'lon_position': lon,
            'AVISO_SSH': ssh,
            'SST' : sst
        })
    
    date_of_interest += delta


# %% Merging and saving processed data
if not processed:
    additional_data_df = pd.DataFrame(additional_data)
    # Merge with ds using profile_id as the key
    ds = pd.merge(ds, additional_data_df, on='profile_id', how='left')
    ds = pd.merge(ds, mld_table, on='profile_id', how='left')
    # Save processed data
    ds.to_pickle(pickle_folder)
    # chatGPT can't read this version of pickle, so save as csv
    gpt_path = "/home/jmiranda/SubsurfaceFields/Data/glider_gpt.csv"
    ds.to_csv(gpt_path, index=False)
    

# %%
