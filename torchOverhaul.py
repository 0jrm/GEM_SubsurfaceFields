#%%
"""
https://chat.openai.com/share/45e1a02e-0808-49b8-ac2c-f95f54b05138

https://chat.openai.com/share/f450167a-b09e-4a2b-a567-fb36265e4216

https://chat.openai.com/share/f6717d7b-d070-45cf-8c14-8ec32d7a9413
"""
import numpy as np
import mat73
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

class TemperatureSalinityDataset(torch.utils.data.Dataset):
    """
    Custom dataset for temperature and salinity profiles.
    
    Attributes:
    - TEMP: Temperature profiles matrix.
    - SAL: Salinity profiles matrix.
    - SSH: Sea Surface Height vector.
    - pca_temp: PCA model for temperature profiles.
    - pca_sal: PCA model for salinity profiles.
    - temp_pcs: Transformed temperature profiles using PCA.
    - sal_pcs: Transformed salinity profiles using PCA.
    """
    def __init__(self, path, n_components=15, input_params=None):
        """
        Args:
        - path (str): File path to the dataset.
        - n_components (int): Number of PCA components to retain.
        """
        data = mat73.loadmat(path)
        # Filtering profiles
        valid_mask = self._get_valid_mask(data)
        self.TEMP, self.SAL, self.SSH, self.TIME, self.LAT, self.LON = self._filter_and_fill_data(data, valid_mask)
        # Applying PCA
        self.temp_pcs, self.pca_temp = self._apply_pca(self.TEMP, n_components)
        self.sal_pcs, self.pca_sal = self._apply_pca(self.SAL, n_components)
        
        # Define which parameters to include
        if input_params is None:
            input_params = {
                "time": False,
                "lat": False,
                "lon": False,
                "sst": True,  # First value of temperature
                "ssh": True
            }

        self.input_params = input_params

    def __getitem__(self, idx):
        """
        Args:
        - idx (int): Index of the profile.

        Returns:
        - tuple: input values and concatenated PCA components for temperature and salinity.
        """
        
        inputs = []
        
        if self.input_params["time"]:
            inputs.append(np.cos(self.TIME[idx]%365))  # assuming TIME is available in your dataset
        
        if self.input_params["lat"]:
            inputs.append(self.LAT[idx])  # assuming LAT is available in your dataset

        if self.input_params["lon"]:
            inputs.append(self.LON[idx])  # assuming LON is available in your dataset

        if self.input_params["sst"]:
            inputs.append(self.TEMP[0, idx])  # First value of temperature profile
        
        if self.input_params["ssh"]:
            inputs.append(self.SSH[idx])
            
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        profiles = torch.tensor(np.hstack([self.temp_pcs[:, idx], self.sal_pcs[:, idx]]), dtype=torch.float32)
        
        return inputs_tensor, profiles
    
    def _get_valid_mask(self, data):
        """Internal method to get mask of valid profiles based on missing values."""
        temp_mask = np.sum(np.isnan(data['TEMP']), axis=0) <= 10
        sal_mask = np.sum(np.isnan(data['SAL']), axis=0) <= 10
        return np.logical_and(temp_mask, sal_mask)
    
    def _filter_and_fill_data(self, data, valid_mask):
        """Internal method to filter data using the mask and fill missing values."""
        TEMP = data['TEMP'][:, valid_mask]
        SAL = data['SAL'][:, valid_mask]
        SSH = data['SH1950'][valid_mask]
        TIME = data['TIME'][valid_mask]
        LAT = data['LAT'][valid_mask]
        LON = data['LON'][valid_mask]
        
        # Fill missing values using interpolation
        for i in range(TEMP.shape[1]):
            valid_temp_idx = np.where(~np.isnan(TEMP[:, i]))[0]
            TEMP[:, i] = np.interp(range(TEMP.shape[0]), valid_temp_idx, TEMP[valid_temp_idx, i])
            valid_sal_idx = np.where(~np.isnan(SAL[:, i]))[0]
            SAL[:, i] = np.interp(range(SAL.shape[0]), valid_sal_idx, SAL[valid_sal_idx, i])
            
        return TEMP, SAL, SSH, TIME, LAT, LON

    def _apply_pca(self, data, n_components):
        """Internal method to apply PCA transformation to the data."""
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(data.T).T
        return pcs, pca
    
    def __len__(self):
        """Returns number of profiles in the dataset."""
        return self.TEMP.shape[1]
    
    # def __getitem__(self, idx):
    #     """
    #     Args:
    #     - idx (int): Index of the profile.

    #     Returns:
    #     - tuple: SSH value and concatenated PCA components for temperature and salinity.
    #     """
    #     return torch.tensor(self.SSH[idx], dtype=torch.float32).unsqueeze(-1), torch.tensor(np.hstack([self.temp_pcs[:, idx], self.sal_pcs[:, idx]]), dtype=torch.float32)

    def inverse_transform(self, pcs):
        """
        Inverse the PCA transformation.

        Args:
        - pcs (numpy.ndarray): Concatenated PCA components for temperature and salinity.

        Returns:
        - tuple: Inversed temperature and salinity profiles.
        """
        temp_profiles = self.pca_temp.inverse_transform(pcs[:, :15]).T
        sal_profiles = self.pca_sal.inverse_transform(pcs[:, 15:]).T
        return temp_profiles, sal_profiles

class PredictionModel(nn.Module):
    """
    Neural Network model for predicting temperature and salinity profiles based on sea surface height (SSH).

    Attributes:
    - model (nn.Sequential): Sequential model containing layers defined by `layers_config`.

    Parameters:
    - input_dim (int): Dimension of the input feature(s). Default is 1 (for SSH).
    - layers_config (list of int): List where each element represents the number of neurons in 
                                   a respective layer. Default is [512, 256].
    - output_dim (int): Dimension of the output. Default is 30 (15 components for TEMP and 15 for SAL).

    Methods:
    - forward(x: torch.Tensor) -> torch.Tensor: Forward pass through the model.
    """

    def __init__(self, input_dim=1, layers_config=[512, 256], output_dim=30):
        super(PredictionModel, self).__init__()
        
        # Construct layers based on the given configuration
        layers = []
        prev_dim = input_dim
        for neurons in layers_config:
            layers.append(nn.Linear(prev_dim, neurons))
            layers.append(nn.ReLU())
            prev_dim = neurons
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the model.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
        - torch.Tensor: Model's predictions of shape (batch_size, output_dim).
        """
        return self.model(x)

def train_model(model, dataloader, criterion, optimizer, device, epochs=100, patience=10):
    """
    Train the model with early stopping and CUDA support.

    Parameters:
    - model: the PyTorch model.
    - dataloader: the DataLoader for training data.
    - criterion: the loss function.
    - optimizer: the optimizer.
    - device: device to which data and model should be moved before training.
    - epochs: number of total epochs to train.
    - patience: number of epochs to wait for improvement before stopping.

    Returns:
    - model: trained model.
    """
    model.to(device)
    best_loss = float('inf')
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")

        # Check for early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at Epoch {epoch + 1}")
                break

    return model

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate the model on the provided data with CUDA support.

    Parameters:
    - model: the PyTorch model.
    - dataloader: the DataLoader for evaluation data.
    - criterion: the loss function.
    - device: device to which data and model should be moved before evaluation.

    Returns:
    - avg_loss: average loss over the dataset.
    """
    model.to(device)
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss

def visualize_results(true_values, predicted_values, num_samples=5):
    """
    Visualize the true vs. predicted values for a sample of profiles using only matplotlib.

    Parameters:
    - true_values: ground truth temperature and salinity profiles.
    - predicted_values: model's predicted temperature and salinity profiles.
    - num_samples: number of random profiles to visualize.

    Returns:
    - None (plots the results).
    """
    n_depths = 2001
    depth_levels = np.arange(n_depths)
    indices = np.random.choice(int(true_values.shape[0]/n_depths), num_samples, replace=False)

    for idx in indices:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        idxx = np.arange(idx*n_depths, (idx+1)*n_depths)
        # Temperature profile
        axs[0].plot(true_values[idxx, 0, 0], depth_levels, label="True")
        axs[0].plot(predicted_values[idxx, 0, 0], depth_levels, label="Predicted")
        axs[0].invert_yaxis()  # To have the surface (depth=0) on top
        axs[0].set_title(f"Temperature Profile {idx}")
        axs[0].set_ylabel("Depth Level")
        axs[0].set_xlabel("Temperature")
        axs[0].legend()

        # Salinity profile
        axs[1].plot(true_values[idxx, 1, 0], depth_levels, label="True")
        axs[1].plot(predicted_values[idxx, 1, 0], depth_levels, label="Predicted")
        axs[1].invert_yaxis()  # To have the surface (depth=0) on top
        axs[1].set_title(f"Salinity Profile {idx}")
        axs[1].set_ylabel("Depth Level")
        axs[1].set_xlabel("Salinity")
        axs[1].legend()

        plt.tight_layout()
        plt.show()

#%%
if __name__ == "__main__":
    # Configurable parameters
    data_path = "/home/jmiranda/SubsurfaceFields/Data/ARGO_GoM_20220920.mat"
    epochs = 500
    patience = 15
    n_components = 15
    batch_size = 100
    learning_rate = 0.001
    layers_config = [100, 100]
    input_params = {
        "time": False,
        "lat": False,
        "lon": False,
        "sst": True,
        "ssh": True
    }

    # Load data
    dataset = TemperatureSalinityDataset(path=data_path, n_components=n_components, input_params=input_params)

    # Compute the input dimension dynamically
    input_dim = sum(val for val in input_params.values())

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # dataset = TemperatureSalinityDataset(path=data_path)
    # # Initialize model, criterion, and optimizer
    # model = PredictionModel(input_dim=1, layers_config=layers_config, output_dim=30)
    
    model = PredictionModel(input_dim=input_dim, layers_config=layers_config, output_dim=n_components*2)
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    trained_model = train_model(model, dataloader, criterion, optimizer, device, epochs, patience)
    
    # Evaluation
    loss = evaluate_model(trained_model, dataloader, criterion, device)
    print(f"Evaluation Loss: {loss:.4f}")

    # Get predictions
    true_profiles, predicted_profiles = [], []
    last_true_profile, last_predicted_profile = None, None

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            predicted_pcs = trained_model(inputs).cpu()
            true_temp, true_sal = dataset.inverse_transform(labels)
            pred_temp, pred_sal = dataset.inverse_transform(predicted_pcs)

            # Convert numpy arrays to tensors
            true_temp_tensor = torch.tensor(true_temp)
            true_sal_tensor = torch.tensor(true_sal)
            pred_temp_tensor = torch.tensor(pred_temp)
            pred_sal_tensor = torch.tensor(pred_sal)

            # Stack along dimension 1
            if i < len(dataloader) - 1:  # for all batches except the last one
                true_profiles.append(torch.stack([true_temp_tensor, true_sal_tensor], dim=1))
                predicted_profiles.append(torch.stack([pred_temp_tensor, pred_sal_tensor], dim=1))
            else:  # specifically handle the last batch
                last_true_profile = torch.stack([true_temp_tensor, true_sal_tensor], dim=1)
                last_predicted_profile = torch.stack([pred_temp_tensor, pred_sal_tensor], dim=1)

    # Concatenate tensors outside the loop
    true_profiles = torch.cat(true_profiles, dim=0)
    predicted_profiles = torch.cat(predicted_profiles, dim=0)

    # Now, true_profiles and predicted_profiles don't include the last batch.
    # last_true_profile and last_predicted_profile contain the tensors for the last batch.

    # Visualization
    visualize_results(true_profiles, predicted_profiles)

    #%%