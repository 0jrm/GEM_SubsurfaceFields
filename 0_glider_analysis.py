# %%
import numpy as np
import xarray as xr
# import os
import sys
import pandas as pd
from scipy.interpolate import griddata
import datetime
import matplotlib.pyplot as plt
import cmocean.cm as ccm

sys.path.append("/home/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/eoas_pyutils/")
from io_utils.coaps_io_data import get_aviso_by_date, get_sst_by_date, get_sss_by_date

pickle_folder = "/home/jmiranda/SubsurfaceFields/Data/glider_processed.pkl"
ds = pd.read_pickle(pickle_folder)

# %%
# Generate the sections from the glider data

# Function to calculate distance between two lat-lon points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

# Extract unique profiles and their lat-lon positions
unique_profiles = ds[['profile_id', 'lat_position', 'lon_position']].drop_duplicates()

# Calculate distances between profiles
distances = [0]  # Start with 0 for the first profile
for i in range(1, len(unique_profiles)):
    d = haversine(
        unique_profiles.iloc[i-1]['lat_position'], 
        unique_profiles.iloc[i-1]['lon_position'], 
        unique_profiles.iloc[i]['lat_position'], 
        unique_profiles.iloc[i]['lon_position']
    )
    distances.append(distances[-1] + d)  # Cumulative distance

# Map profile_id to its distance
profile_to_distance = dict(zip(unique_profiles['profile_id'], distances))
ds['distance'] = ds['profile_id'].map(profile_to_distance)

# Define the grid
grid_z = np.arange(0, ds['depth'].max() + 1, 1)  # 1m resolution in depth
grid_x = np.array(distances)

# Create meshgrid for interpolation
X, Z = np.meshgrid(grid_x, grid_z)

# Interpolate the properties onto the grid
temperature_grid = griddata(
    (ds['distance'], ds['depth']),
    ds['temperature'],
    (X, Z),
    method='linear',
    fill_value=np.nan
)

salinity_grid = griddata(
    (ds['distance'], ds['depth']),
    ds['salinity'],
    (X, Z),
    method='linear',
    fill_value=np.nan
)

density_grid = griddata(
    (ds['distance'], ds['depth']),
    ds['density'],
    (X, Z),
    method='linear',
    fill_value=np.nan
)

# Define the plotting function
def plot_filled_contour(X, Z, grid, title, colormap, cbar_label, ax=None):
    if ax is None:
        ax = plt.gca()
    cs = ax.contourf(X, Z, grid, cmap=colormap, levels=25, extend='both')
    ax.contour(X, Z, grid, colors='black', levels=25, linewidths=0.2)
    cbar = plt.colorbar(cs, ax=ax, orientation='vertical')
    cbar.set_label(cbar_label)
    ax.set_title(title)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Depth (m)')
    ax.invert_yaxis()
    ax.grid(True, color='black', linestyle=':', linewidth=0.2)

# Plot the properties
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(14, 15))

plot_filled_contour(X, Z, temperature_grid, 'Temperature ($^\circ$C)', ccm.thermal, 'Temperature ($^\circ$C)', axs[0])
plot_filled_contour(X, Z, salinity_grid, 'Salinity (PSU)', ccm.haline, 'Salinity (PSU)', axs[1])
plot_filled_contour(X, Z, density_grid, 'Density (kg/m$^3$)', 'hot_r', 'Density (kg/m$^3$)', axs[2])
plt.suptitle('Original Track\n', fontsize=18)
plt.tight_layout()
plt.show()

# %%
def line_equation(p1, p2):
    """Calculate the slope (m) and y-intercept (b) of the line passing through points p1 and p2."""
    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - m * p1[0]
    return m, b

def orthogonal_projection(point, m, b):
    """Calculate the orthogonal projection of a point onto a line defined by y = mx + b."""
    x0, y0 = point
    x_proj = (x0 + m * y0 - m * b) / (m**2 + 1)
    y_proj = (m * x0 + m**2 * y0 + b) / (m**2 + 1)
    return x_proj, y_proj

# Calculate the equation of the line formed by the initial and final positions
origin = (unique_profiles.iloc[0]['lon_position'], unique_profiles.iloc[0]['lat_position'])
end = (unique_profiles.iloc[-1]['lon_position'], unique_profiles.iloc[-1]['lat_position'])
m, b = line_equation(origin, end)
projections = unique_profiles.apply(
    lambda row: orthogonal_projection((row['lon_position'], row['lat_position']), m, b), axis=1
)
proj_x = [proj[0] for proj in projections]
proj_y = [proj[1] for proj in projections]

# Get aviso data to use as background
date_of_interest = datetime.datetime(2016, 8, 21)
bbox = [23.5, 27.5, -93, -88]
aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
aviso_data, grid_lats, grid_lons = get_aviso_by_date(aviso_folder, date_of_interest, bbox)

# Creating a meshgrid
X, Y = np.meshgrid(grid_lons.values, grid_lats.values)

# Plotting the data again
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, aviso_data.adt.values, cmap='jet', levels=50, extend='both')
plt.scatter(unique_profiles['lon_position'], unique_profiles['lat_position'], color='blue', label='Profiles')
plt.plot([origin[0], end[0]], [origin[1], end[1]], color='red', label='Origin-End Line')
plt.scatter(proj_x, proj_y, color='green', label='Orthogonal Projections')
for profile, proj in zip(unique_profiles[['lon_position', 'lat_position']].values, projections):
    plt.plot([profile[0], proj[0]], [profile[1], proj[1]], 'k-', lw=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.title('Orthogonal Projections onto Origin-End Line')
plt.grid(True)
plt.axis('equal')
plt.show()

# %%
# Calculate the distances from each projected point to the origin
projected_distances = [haversine(origin[1], origin[0], y, x) for x, y in zip(proj_x, proj_y)]

# Create a mapping of profile_id to its projected distance
profile_to_projected_distance = dict(zip(unique_profiles['profile_id'], projected_distances))

# Map profile_id to its projected distance in the main dataframe
ds['projected_distance'] = ds['profile_id'].map(profile_to_projected_distance)

# Sort the dataframe by this new distance
data_sorted = ds.sort_values(by='projected_distance')
# Define the new grid for x-axis using sorted projected distances
grid_x_new = np.array(sorted(projected_distances))

# # Alternatively, we can use as it is (unsorted)
# data_sorted = ds
# grid_x_new = np.array(projected_distances)

# Create the new meshgrid for interpolation
X_new, Z_new = np.meshgrid(grid_x_new, grid_z)

# Interpolate the properties onto the new grid using projected_distance
temperature_grid_new = griddata(
    (data_sorted['projected_distance'], data_sorted['depth']),
    data_sorted['temperature'],
    (X_new, Z_new),  
    method='linear',
    fill_value=np.nan
)
# Repeat for salinity and density
salinity_grid_new = griddata(
    (data_sorted['projected_distance'], data_sorted['depth']),
    data_sorted['salinity'],
    (X_new, Z_new),  
    method='linear',
    fill_value=np.nan
)

density_grid_new = griddata(
    (data_sorted['projected_distance'], data_sorted['depth']),
    data_sorted['density'],
    (X_new, Z_new), 
    method='linear',
    fill_value=np.nan
)

# Plot the properties using the new interpolated grids
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(14, 15))

plot_filled_contour(X_new, Z_new, temperature_grid_new, 'Temperature ($^\circ$C)', ccm.thermal, 'Temperature ($^\circ$C)', axs[0])
plot_filled_contour(X_new, Z_new, salinity_grid_new, 'Salinity (PSU)', ccm.haline, 'Salinity (PSU)', axs[1])
plot_filled_contour(X_new, Z_new, density_grid_new, 'Density (kg/m$^3$)', 'hot_r', 'Density (kg/m$^3$)', axs[2])
# Repeat for salinity and density with their respective new grids
plt.suptitle('Straight projection of the Track\n', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.grid(True)
plt.show()

# %%
# Implement the Geosprophic and cyclogeostrophic velocities and relative vorticities calculations
# and plot them


# %%
# Save processed data
track_folder = "/home/jmiranda/SubsurfaceFields/Data/glider_track_sorted.pkl"

data_sorted.to_pickle(track_folder)
# %%
