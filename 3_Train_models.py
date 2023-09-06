# %%
import sys
sys.path.append("ai_common_torch/")
sys.path.append("eoas_pyutils/")

# Local libraries
from proj_ai.Training import train_model
from proj_io.Generators import ProjDataset, ProjDatasetPCA
from proj_ai.proj_models import BasicDenseModel
from configs.RunConfig import RunConfig

# External libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print("Using device: ", device)

# %%
profile_code = False
val_perc = 0.1
batch_size_training = 400
workers = 20
model_name = 'BasicDenseModel_WithPCA_Sep6'
with_pca = True
seed = 42
# Only if using pca
temp_componets = 10
sal_components = 10

#%% ----- Create DataLoaders --------
data_folder = RunConfig.data_folder.value
output_folder = RunConfig.training_folder.value

if with_pca:
    dataset = ProjDatasetPCA(data_folder, temp_components=temp_componets, 
                             sal_components=sal_components,
                             test=False)
else:
    dataset = ProjDataset(data_folder)

# %%# Split train and validation 'normal/default way'
train_size = int( (1 - val_perc) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, 
                                                           [train_size, val_size], generator=torch.Generator().manual_seed(seed))
print("Total number of training samples: ", len(train_dataset))
print("Total number of validation samples: ", len(val_dataset))

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=batch_size_training, shuffle=True, num_workers=workers)
val_loader = DataLoader(val_dataset, batch_size= len(val_dataset),  shuffle=False, num_workers=workers)
print("Done loading data!")

#%% Visualize some data
# Plot from a batch of training data
# dataiter = iter(train_loader)
# Make aplot 


#%% Initialize the model, loss, and optimizer
inout_dims = dataset.get_inout_dims()
input_size = inout_dims[0]  
output_size = inout_dims[1]  
hidden_layers = 4
neurons_per_layer = 100
activation_hidden = 'relu'
activation_output = 'linear'
batch_norm = True
model = BasicDenseModel(input_size, output_size, hidden_layers, neurons_per_layer, 
                      activation_hidden, activation_output, batch_norm)
model.to(device)


loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

max_num_epochs = 1000  # Maximum number of epochs to train
patience = 10 # How many epochs to wait before stopping training if no improvement

if profile_code:
    import cProfile
    profiler = cProfile.Profile()
    profiler.enable()

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
