# %%
import sys
sys.path.append("ai_common_torch/")
sys.path.append("eoas_pyutils/")

# Local libraries
from proj_ai.Training import train_model
from proj_io.Generators import ProjDataset
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
model_name = 'BasicDenseModel'

#%% ----- Create DataLoaders --------
data_folder = RunConfig.data_folder.value
output_folder = RunConfig.training_folder.value

train_dataset = ProjDataset(data_folder)

# %%# Split train and validation 'normal/default way'
train_size = int( (1 - val_perc) * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
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
input_size = 2  #  SSH and SST
output_size = 2001*2  # 2001 vertical levels, 2 variables
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