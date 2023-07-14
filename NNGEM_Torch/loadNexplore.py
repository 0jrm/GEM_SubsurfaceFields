# %% 
import numpy as np
import pandas as pd
import mat73
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from os.path import join

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Disable GPU

# %%

input_folder = '/home/jmiranda/SubsurfaceFields/latest_naiveGEMS'
# Julian time is weird
# epoch_start = datetime.datetime(0000, 1, 1)

# ADT_1 = mat73.loadmat(          'ADT_20220920.mat'          , use_attrdict=True) 
# ADT_noseason = mat73.loadmat(   'ADT_noseason_20220920.mat' , use_attrdict=True) 
ARGO = mat73.loadmat( join(input_folder,'ARGO_GoM_20220920.mat')     , use_attrdict=True) 
# LoopCur = loadmat(              'LoopCurrentRings_edges.mat'    )

print(ARGO.keys())
# dict_keys(['ADT_loc', 'ADTnoseason_loc', 'LAT', 'LON', 'PRES', 'RHO', 'SAL', 'SH1950', 'SIG', 'SPICE', 'TEMP', 'TIME'])

# %%
# ARGO.ADT_loc[0] 
ARGO.SH1950[0] 
# datetime.datetime.fromordinal(round( ARGO.TIME[0] - 1721425.5))
import matplotlib.pyplot as plt

example = 0
fig, axs = plt.subplots(1, 2)
axs[0].plot(ARGO.TEMP[:,example], range(2001))
# Flip y axis
axs[0].invert_yaxis()

plt.show()


# %%
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

# %%[2]:

# type(ARGO)
# # pd.dataframe(ARGO.LAT)
# argo_dict = {
#     'ADT_loc': ARGO['ADT_loc'],
#     'ADTnoseason_loc': ARGO['ADTnoseason_loc'],
#     'LAT': ARGO['LAT'],
#     'LON': ARGO['LON'],
#     # 'PRES': ARGO['PRES'],
#     # 'RHO': ARGO['RHO'],
#     # 'SAL': ARGO['SAL'],
#     # 'SH1950': ARGO['SH1950'],
#     # 'SIG': ARGO['SIG'],
#     # 'SPICE': ARGO['SPICE'],
#     # 'TEMP': ARGO['TEMP'],
#     'TIME': ARGO['TIME']
# }

# df = pd.DataFrame.from_dict(argo_dict)

# df.describe()

# %%[3]:

## Data reduction: lower resolution (5m) + remove profiles w/ NaNs
# we know the first 5m and the last 200m are the most problematic, so we prune them
# Extract the relevant arrays from the ARGO variable
# print(ARGO.PRES[:,0]) # from 0 to 2000m
pres = np.arange(5, 1800, 5, dtype=int) #halve the resolution, also seves as index
sal = ARGO.SAL[pres,:]
temp = ARGO.TEMP[pres,:]

# Julian time is weird, so we're not using it for now
# DATE1 = epoch_start + datetime.timedelta(days=ARGO.TIME[0])
# print(DATE1)

# print(np.max(pres))

# data contains NaN values, cleaning...
nan_index = np.logical_or(np.isnan(sal).any(axis=0), np.isnan(temp).any(axis=0))

# print(np.sum(nan_index))

sal = sal[:, ~nan_index]
# print(np.shape(sal))

temp = temp[:, ~nan_index]
lat = ARGO.LAT[~nan_index]
lon = ARGO.LON[~nan_index]
ssh = ARGO.SH1950[~nan_index]

sst = np.mean(temp[:1,:],axis=0) # Average the temperature of the first 10m?
print("pres size: ", np.shape(pres))

# Concatenate the input arrays into a single 2D array
X = np.column_stack((ssh, sst))

# Reshape the output arrays into a 2D array
y = np.column_stack((temp.T, sal.T))

# print(np.isnan(X).any())
# print(np.isnan(y).any())

print(np.shape(X))
print(np.shape(y))



# %%[4]:


# Define the log directory for TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Define the TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


# Find the indices of rows without NaN values
valid_idx = np.where(~np.isnan(X).any(axis=1))[0]

# Select a random subset of the valid rows
np.random.seed(42)  # set random seed for reproducibility
subset_idx = np.random.choice(valid_idx, size=2000, replace=False)

# Split the selected subset into training, validation, and testing sets
X_subset = X[subset_idx]
y_subset = y[subset_idx]
X_train, X_val_test, y_train, y_val_test = train_test_split(X_subset, y_subset, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)


print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)


# Normalize the input data using z-score normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(2001*2, activation=None),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(718, activation=None)
])


# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model with progress indicator
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[tensorboard_callback], verbose=2)

# Evaluate the model
mse_test = model.evaluate(X_test, y_test)

# Use the model for predictions
X_new = np.array([[2.5, 20.0]])  # example input
X_new = scaler.transform(X_new)
y_pred = model.predict(X_new)

# Save the model
model.save('model.h5')


# # Split the dataset into training, validation, and testing sets
# X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# # Normalize the input data using z-score normalization
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)

# # Define the log directory for TensorBoard
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# # Define the TensorBoard callback
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# # Define the neural network architecture
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(2001*2, activation=None),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(4002, activation=None)
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='mse')

# # Train the model with progress indicator
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[tensorboard_callback])

# # Evaluate the model
# mse_test = model.evaluate(X_test, y_test)

# # Use the model for predictions
# X_new = np.array([[2.5, 20.0]])  # example input
# X_new = scaler.transform(X_new)
# y_pred = model.predict(X_new)

# # Save the model
# model.save('model.h5')



# %%[5]:


import matplotlib.pyplot as plt
import numpy as np

# Plot the loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Calculate predictions on validation set
y_pred = model.predict(X_val)

print(np.shape(y_val))
print(np.shape(y_pred))
print(np.shape(pres))




# %%[8]:


value = 0

widgets.IntSlider(
    value=0,
    min=0,
    max=200,
    step=1,
    description='Test:',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

widgets.IntSlider()

# Plot comparison between predictions and ground truth
plt.plot(y_val[value,:359], pres, label='Validation')
plt.plot(y_pred[value,:359], pres, label='Prediction')
plt.title('Comparison between Ground Truth and Predicted Temperature')
plt.gca().invert_yaxis()
plt.ylabel('Pressure')
plt.xlabel('Temperature')
plt.legend()
plt.show()

# Plot comparison between predictions and ground truth
plt.plot(y_val[value,359:], pres, label='Validation')
plt.plot(y_pred[value,359:], pres, label='Prediction')
plt.title('Comparison between Ground Truth and Predicted Salinity')
plt.gca().invert_yaxis()
plt.ylabel('Pressure')
plt.xlabel('Salinity')
plt.legend()
plt.show()

# Calculate and print mean squared error on validation set
mse_val = model.evaluate(X_val, y_val)
print('MSE on validation set:', mse_val)

