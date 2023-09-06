# %%
import sys
# sys.path.append("ai_common_torch/")
# sys.path.append("eoas_pyutils/")

# Local libraries
from proj_io.Generators import ProjDataset, ProjDatasetPCA
from proj_ai.Training import train_model
from proj_ai.proj_models import BasicDenseModel
from configs.RunConfig import RunConfig
from eoas_pyutils.io_utils.io_common import create_folder
from proj_viz.argo_viz import compare_profiles
from  proj_io.argo_io import read_normalize_data_test, revert_normalization_test

# External libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from os.path import join
import pickle

#%%
with_pca = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("Using device: ", device)

profile_code = False
val_perc = 0.1
batch_size_training = 400
workers = 20
if with_pca:
    model_name = 'BasicDenseModel_WithPCA'
else:
    model_name = 'BasicDenseModel_NoPCA'
# Only if using pca
temp_components = 100
sal_components = 100

#%%
# ----- Create DataLoaders --------
data_folder = RunConfig.data_folder.value
output_folder = RunConfig.training_folder.value

if with_pca:
    dataset = ProjDatasetPCA(data_folder, temp_components=temp_components, sal_components=sal_components)
else:
    dataset = ProjDataset(data_folder)

%%
import matplotlib.pyplot as plt
import numpy as np

depths = np.arange(0, 2001)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))

# Plot data
ax[0].fill_betweenx(depths, dataset.avg_temp - dataset.std_temp, dataset.avg_temp + dataset.std_temp, color='lightcoral', label='±3 std. dev.')
ax[1].fill_betweenx(depths, dataset.avg_sal - dataset.std_sal, dataset.avg_sal + dataset.std_sal, color='lightblue', label='±3 std. dev.')
for i in range(len(dataset.temp[0])):
    ax[0].plot(dataset.temp.T[i], depths, color='lightgray', linewidth=0.1, alpha = 0.2)
    ax[1].plot(dataset.salt.T[i], depths, color='lightgray', linewidth=0.1, alpha = 0.2)

ax[0].plot(dataset.avg_temp, depths, 'r-', label='Average Temp.')
ax[0].invert_yaxis()
ax[0].set_xlabel('Temperature (°C)')
ax[0].set_ylabel('Depth (m)')
ax[0].set_title('Temperature Profiles')
# ax[0].legend()

# Plot salinity
ax[1].plot(dataset.avg_sal, depths, 'b-', label='Average Salinity')
ax[1].invert_yaxis()
ax[1].set_xlabel('Salinity (PSU)')
ax[1].set_ylabel('Depth (m)')
ax[1].set_title('Salinity Profiles')
# ax[1].legend()

plt.tight_layout()
plt.show()


# %%# Split train and validation 'normal/default way'
train_size = int( (1 - val_perc) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
print("Total number of training samples: ", len(train_dataset))
print("Total number of validation samples: ", len(val_dataset))

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size_training, shuffle=True, num_workers=workers)
val_loader = DataLoader(val_dataset, batch_size= len(val_dataset),  shuffle=False, num_workers=workers)
print("Done loading data!")

#%%
# Visualize some data
# Plot from a batch of training data
dataiter = iter(train_loader)
inputs, labels = next(dataiter)
print("Input Shape:", inputs.shape) # 400, 2
print("Labels Shape:", labels.shape) # 400, 200 for PCA, 400, 4002 no PCA

#%%
# Initialize the model, loss, and optimizer
inout_dims = dataset.get_inout_dims()

input_size = inout_dims[0]  
output_size = inout_dims[1]  
hidden_layers = 2
neurons_per_layer = 100
# activation_hidden = 'sigmoid' #best for NoPCA
activation_hidden = 'relu' # tanh sigmoid # relu
# activation_hidden = 'tanh' # tanh sigmoid # relu
activation_output = 'linear'
batch_norm = True

# save the variables above
cur_time = datetime.now()
save_folder = f'{model_name}_{cur_time.strftime("%Y-%m-%d_%H:%M")}'
save_folder = join(output_folder, save_folder)
create_folder(save_folder)

with open(join(save_folder, "model_params.pkl"), "wb") as f:
    pickle.dump([input_size, output_size, hidden_layers, neurons_per_layer,
                 activation_hidden, activation_output, batch_norm], f)
    
with open(join(save_folder, "train_val.pkl"), "wb") as f:
    pickle.dump([train_dataset, val_dataset], f)

model = BasicDenseModel(input_size, output_size, hidden_layers, neurons_per_layer, 
                      activation_hidden, activation_output, batch_norm)
model.to(device)

loss_func = nn.MSELoss()
# loss_func = nn.HuberLoss() # CTCLoss CosineEmbeddingLoss # SmoothL1Loss

optimizer = optim.Adam(model.parameters(), lr=0.001)

max_num_epochs = 1000  # Maximum number of epochs to train
patience = 10 # How many epochs to wait before stopping training if no improvement

if profile_code:
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()

#%%
# Train the model
model = train_model(model, optimizer, loss_func, train_loader, val_loader, 
                    max_num_epochs, 
                    model_name,
                    device, 
                    patience=patience,
                    output_folder=output_folder)

if profile_code:
    profiler.disable()
    profiler.dump_stats('profile_stats.prof')

print("Done training!")

# %% 
# Use the model to predict a couple of profiles in the validation set
for batch_idx, (data, target) in enumerate(train_loader):
    print(f'{batch_idx}/{len(train_loader.dataset)}', end='\r')
    data, target = data.to(device), target.to(device)
    

#%%
inoutdims = dataset.get_inout_dims()
if not with_pca:
    # revert_normalization_test(dataset, with_pca, dataset.scaler)
    revert_normalization_test(dataset, with_pca)

for i, batch in enumerate(val_loader):
    # Print the size of the batch, size of input and output
    print(f'Batch size: {batch_size_training}, Input size: {batch[0].shape}, Output size: {batch[1].shape}')
    ssh, ts = batch
    # for j in range(50):
    for j in range(batch_size_training):
        temp_profile_original, sal_profile_original  = dataset.get_original_profile(i*batch_size_training+j)
        # temp_profile_original, sal_profile_original = revert_normalization_prof(temp_profile_original, sal_profile_original, data_folder, dataset.scaler)
        if with_pca:
            temp_profile, sal_profile  = dataset.inverse_pca(ts[j][0:temp_components], ts[j][temp_components:])
            temp_profile_original, sal_profile_original  = dataset.inverse_pca(temp_profile_original, sal_profile_original)
            # temp_profile, sal_profile = revert_normalization_prof(temp_profile, sal_profile, data_folder, dataset.scaler)
            name = 'PCA'
        else:
            temp_profile, sal_profile = ts[j][0:int(inoutdims[1]/2)], ts[j][int(inoutdims[1]/2):]
            # temp_profile, sal_profile = revert_normalization_prof(temp_profile, sal_profile, data_folder, dataset.scaler)
            name = 'NN'
                
        compare_profiles(temp_profile, temp_profile_original, title=f'Batch {i} element {j} SSH {ssh[j][0]}', 
                            labelone="Temp " + name, labeltwo="Temp Original", figsize=5, same_parameter=True)

        compare_profiles(sal_profile, sal_profile_original, title=f'Batch {i} element {j} SSH {ssh[j][0]}', 
                            labelone="Salinity " + name, labeltwo="Salinity Original", figsize=5, same_parameter=True)
    break

#%%