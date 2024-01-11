#%%
import sys
import os
import numpy as np
import mat73
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr 
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
# from scipy.signal import convolve2d
from scipy.interpolate import RegularGridInterpolator
from torch.utils.data import random_split
import pickle
import cartopy.crs as ccrs
from datetime import datetime, timedelta
from sklearn.cluster import MiniBatchKMeans

# Set the seed for reproducibility
seed = 99
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# from matplotlib.dates import date2num, num2date
# from sklearn.cluster import MiniBatchKMeans
sys.path.append("/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/eoas_pyutils/")
from io_utils.coaps_io_data import get_aviso_by_date, get_sst_by_date, get_sss_by_date
sys.path.append("/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/")

def datenum_to_datetime(matlab_datenum):
    # MATLAB's datenum (1) is equivalent to January 1, year 0000, but Python's datetime minimal year is 1
    # There are 366 days for year 0 in MATLAB (it's a leap year in proleptic ISO calendar)
    days_from_year_0_to_year_1 = 366
    python_datetime = datetime.fromordinal(int(matlab_datenum) - days_from_year_0_to_year_1) + timedelta(days=matlab_datenum % 1)
    return python_datetime

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
    def __init__(self, path, n_components=15, input_params=None, max_depth = 2000, min_depth = 20):
        """
        Args:
        - path (str): File path to the dataset.
        - n_components (int): Number of PCA components to retain.
        """
        self.aviso_folder = "/unity/f1/ozavala/DATA/GOFFISH/AVISO/GoM/"
        self.sst_folder = "/unity/f1/ozavala/DATA/GOFFISH/SST/OISST"
        self.sss_folder = "/Net/work/ozavala/DATA/GOFFISH/SSS/SMAP_Global/"
        
        self.max_depth = max_depth
        self.min_depth = min_depth # data quality is poor above 20m
        
        self.data = mat73.loadmat(path)
        self.TIME = [datenum_to_datetime(datenum) for datenum in self.data['TIME']]
        # self.TIME = data['TIME']
        self.LAT = self.data['LAT']
        self.LON = self.data['LON']
        self.ADT = self.data['ADTnoseason_loc']
        self.min_lat = 17.0
        self.max_lat = 29.0
        self.min_lon = -91.0
        self.max_lon = -78.0
        
        # Define which parameters to include
        if input_params is None:
            input_params = {
                "timecos": False,
                "timesin": False,
                "latcos": False,
                "latsin": False,
                "loncos": False,
                "lonsin": False,
                "sst": True,  # First value of temperature
                "sss": True,
                "ssh": True
            }

        self.input_params = input_params
        
        self.SSS, self.SST, self.SSH = self._load_satellite_data()
        self.satSSS, self.satSST, self.satSSH = self.SSS, self.SST, self.SSH #backup
        
        valid_mask = self._get_valid_mask(self.data)
        self.TEMP, self.SAL, self.SSH, self.SST, self.SSS, self.TIME, self.LAT, self.LON, self.ADT = self._filter_and_fill_data(self.data, valid_mask)
        
        # Applying PCA
        self.temp_pcs, self.pca_temp = self._apply_pca(self.TEMP, n_components)
        self.sal_pcs, self.pca_sal = self._apply_pca(self.SAL, n_components)

    def reload(self):
        # in case we want to change parameters...
        self.SSS, self.SST, self.SSH = self.satSSS, self.satSST, self.satSSH
        valid_mask = self._get_valid_mask(self.data)
        self.TEMP, self.SAL, self.SSH, self.SST, self.SSS, self.TIME, self.LAT, self.LON, self.ADT = self._filter_and_fill_data(self.data, valid_mask)
        
        # Applying PCA
        self.temp_pcs, self.pca_temp = self._apply_pca(self.TEMP, n_components)
        self.sal_pcs, self.pca_sal = self._apply_pca(self.SAL, n_components)

    def _load_satellite_data(self):
        """
        New method to load SST and SSH data
        """
        unique_dates = sorted(list(set(self.TIME)))
        sss_data = np.nan * np.ones(len(self.TIME))
        sst_data = np.nan * np.ones(len(self.TIME))
        aviso_data = np.nan * np.ones(len(self.TIME))
        bbox=(self.min_lat, self.max_lat, self.min_lon, self.max_lon)
        
        # # Convert serialized date numbers to date objects
        # base_date = datetime(1, 1, 1)

        # for idx, serialized_date in enumerate(unique_dates):
        for idx, c_date in enumerate(unique_dates):
            # c_date = base_date + timedelta(days=float(serialized_date))
            date_idx = np.array([date_obj == c_date for date_obj in self.TIME])  # Ensure both sides of the comparison are datetime objects
            coordinates = np.array([self.LAT[date_idx], self.LON[date_idx]]).T
            
            try:
                sss_datapoint, lats, lons = get_sss_by_date(self.sss_folder, c_date, bbox)
                interpolator = RegularGridInterpolator((lats, lons), sss_datapoint.sss_smap.values, bounds_error=False, fill_value=None)
                sss_data[date_idx] = interpolator(coordinates)
            except Exception as e:
                print(f"SSS not found for date {c_date}. Error: {e}")
            
            try:
                aviso_date, aviso_lats, aviso_lons = get_aviso_by_date(self.aviso_folder, c_date, bbox)
                interpolator_ssh = RegularGridInterpolator((aviso_lats, aviso_lons), aviso_date.adt.values, bounds_error=False, fill_value=None)
                aviso_data[date_idx] = interpolator_ssh(coordinates)
            except Exception as e:
                print("AVISO not found for date ", c_date, "Error: ", str(e))
                continue

            try:
                sst_date, sst_lats, sst_lons = get_sst_by_date(self.sst_folder, c_date, bbox)
                interpolator_sst = RegularGridInterpolator((sst_lats, sst_lons), sst_date.analysed_sst.values[0], bounds_error=False, fill_value=None)
                sst_data[date_idx] = interpolator_sst(coordinates)
            except Exception as e:
                print("SST not found for date ", c_date, "Error: ", str(e))
                continue

    # Check if data was actually filled
            if np.isnan(aviso_data[date_idx]).all():
                print(f"No AVISO data for date {c_date}")
            if np.isnan(sst_data[date_idx]).all():
                print(f"No SST data for date {c_date}")

        return sss_data, sst_data, aviso_data

    def __getitem__(self, idx):
        """
        Args:
        - idx (int): Index of the profile.

        Returns:
        - tuple: input values and concatenated PCA components for temperature and salinity.
        """
        
        inputs = []
        
        if self.input_params["timecos"]:
            inputs.append(np.cos(2*np.pi*(self.TIME[idx]%365)/365)) 
            
        if self.input_params["timesin"]:
            inputs.append(np.sin(2*np.pi*(self.TIME[idx]%365)/365))  
        
        if self.input_params["latcos"]:
            inputs.append(np.cos(2*np.pi*(self.LAT[idx]/180)))

        if self.input_params["latsin"]:
            inputs.append(np.sin(2*np.pi*(self.LAT[idx]/180)))  

        if self.input_params["loncos"]:
            inputs.append(np.cos(2*np.pi*(self.LON[idx]/360)))  
            
        if self.input_params["loncos"]:
            inputs.append(np.sin(2*np.pi*(self.LON[idx]/360)))
            
        if self.input_params["sat"]:                
            if self.input_params["sss"]:
                # inputs.append(self.SAL[0, idx])
                inputs.append(self.SSS[idx])

            if self.input_params["sst"]:
                # inputs.append(self.TEMP[0, idx])  # First value of temperature profile
                inputs.append(self.SST[idx])
                
            if self.input_params["ssh"]:
                # inputs.append(self.ADT[idx]) #Uses profile SSH
                inputs.append(self.SSH[idx]) #Uses satellite SSH
        else:
            if self.input_params["sss"]:
                inputs.append(self.SAL[0, idx])
                # inputs.append(self.SSS[idx])

            if self.input_params["sst"]:
                inputs.append(self.TEMP[0, idx])  # First value of temperature profile
                # inputs.append(self.SST[idx])
                
            if self.input_params["ssh"]:
                inputs.append(self.ADT[idx]) #Uses profile SSH
                # inputs.append(self.SSH[idx]) #Uses satellite SSH
            
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
        profiles = torch.tensor(np.hstack([self.temp_pcs[:, idx], self.sal_pcs[:, idx]]), dtype=torch.float32)
        return inputs_tensor, profiles
    
    def _get_valid_mask(self, data):
        """Internal method to get mask of valid profiles based on missing values."""
        temp_mask = np.sum(np.isnan(data['TEMP']), axis=0) <= 10
        sal_mask = np.sum(np.isnan(data['SAL']), axis=0) <= 10
        
        ssh_mask = ~np.isnan(self.SSH)
        sst_mask = ~np.isnan(self.SST)
        sss_mask = ~np.isnan(self.SSS)
        # print(len(temp_mask), len(sal_mask), len(ssh_mask), len(sst_mask))
        combined_mask = np.logical_and(temp_mask, sal_mask)
        print(f"Filtered dataset (sal/temp) contains {np.sum(combined_mask)} profiles.")
        combined_mask = np.logical_and(combined_mask, sst_mask)
        print(f"Filtered dataset (sal/temp/sst) contains {np.sum(combined_mask)} profiles.")
        combined_mask = np.logical_and(combined_mask, ssh_mask)
        print(f"Filtered dataset (sal/temp/ssh/sst) contains {np.sum(combined_mask)} profiles.")
        combined_mask = np.logical_and(combined_mask, sss_mask)
        print(f"Filtered dataset (sal/temp/ssh/sst/sss) contains {np.sum(combined_mask)} profiles.")
        
        return combined_mask
    
    def _filter_and_fill_data(self, data, valid_mask):
        """Internal method to filter data using the mask and fill missing values."""
        TEMP = data['TEMP'][self.min_depth:self.max_depth+1, valid_mask]
        SAL = data['SAL'][self.min_depth:self.max_depth+1, valid_mask]
        SSH = self.SSH[valid_mask]
        SST = self.SST[valid_mask]
        SSS = self.SSS[valid_mask]
        TIME = data['TIME'][valid_mask]
        LAT = data['LAT'][valid_mask]
        LON = data['LON'][valid_mask]
        ADT = data['ADTnoseason_loc'][valid_mask]
        
        # Fill missing values using interpolation
        for i in range(TEMP.shape[1]):
            valid_temp_idx = np.where(~np.isnan(TEMP[:, i]))[0]
            TEMP[:, i] = np.interp(range(TEMP.shape[0]), valid_temp_idx, TEMP[valid_temp_idx, i])
            valid_sal_idx = np.where(~np.isnan(SAL[:, i]))[0]
            SAL[:, i] = np.interp(range(SAL.shape[0]), valid_sal_idx, SAL[valid_sal_idx, i])
        
        # # Define the moving average filter (e.g., for a window size of 5)
        # window_size = 5
        # passes = 1
        # filter_weights = np.ones((window_size, 1)) / window_size
        
        # for i in range(passes):
        #     # Apply the moving average filter to the entire matrix
        #     TEMP = convolve2d(TEMP, filter_weights, boundary='symm', mode='same')
        #     SAL = convolve2d(SAL, filter_weights, boundary='symm', mode='same') 
        
        return TEMP, SAL, SSH, SST, SSS, TIME, LAT, LON, ADT

    def _apply_pca(self, data, n_components):
        """Internal method to apply PCA transformation to the data."""
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(data.T).T
        return pcs, pca
    
    def __len__(self):
        """Returns number of profiles in the dataset."""
        return self.TEMP.shape[1]
    
    def inverse_transform(self, pcs):
        """
        Inverse the PCA transformation.

        Args:
        - pcs (numpy.ndarray): Concatenated PCA components for temperature and salinity.

        Returns:
        - tuple: Inversed temperature and salinity profiles.
        """
        temp_profiles = self.pca_temp.inverse_transform(pcs[:, :n_components]).T
        sal_profiles = self.pca_sal.inverse_transform(pcs[:, n_components:]).T
        return temp_profiles, sal_profiles
    
    def get_profiles(self, indices, pca_approx=False):
        """
        Returns temperature and salinity profiles for the given indices.

        Args:
        - indices (list or numpy.ndarray): List of indices for which profiles are needed.
        - pca_approx (bool): Flag to return PCA approximated profiles if True, 
                             or original profiles if False.

        Returns:
        - numpy.ndarray: concatenated temperature and salinity profiles in the required format for visualization.
        """
        if pca_approx:
            # Concatenate temp and sal PCA components for the given indices
            concatenated_pcs = np.hstack([self.temp_pcs[:, indices].T, self.sal_pcs[:, indices].T])
            # Obtain PCA approximations using the concatenated components
            temp_profiles, sal_profiles = self.inverse_transform(concatenated_pcs)
        else:
            temp_profiles = self.TEMP[:, indices]
            sal_profiles = self.SAL[:, indices]

        # Stack along the third dimension
        profiles_array = np.stack([temp_profiles, sal_profiles], axis=1)

        return profiles_array
    
    def get_gem_profiles(self, indices, sat_ssh = False, filename='/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/GEM_polyfits.pkl'):
        # Load GEM polyfits from file
        with open(filename, 'rb') as f:
            gem_polyfits = pickle.load(f)
            
        # Interpolate the temperature and salinity data onto the new grid
        temp_GEM = np.empty((len(indices), self.max_depth+1-self.min_depth))
        sal_GEM = np.empty((len(indices), self.max_depth+1-self.min_depth))

        if sat_ssh:
            # ssh = self.SSH[indices] 
            ssh = 2668.9* self.SSH[indices]  + 4214.1
        else:
            ssh = self.ADT[indices]
        
        # For each pressure level
        for i in range(len(gem_polyfits['TEMP'])):
            # Evaluate the fitted polynomial at the given SSH values
            temp_GEM[:, i] = gem_polyfits['TEMP'][i](ssh)
            sal_GEM[:, i] = gem_polyfits['SAL'][i](ssh)
        
        # Interpolate missing values in temp_GEM and sal_GEM
        for array in [temp_GEM, sal_GEM]:
            for row in range(array.shape[0]):
                valid_mask = ~np.isnan(array[row])
                if not valid_mask.any():  # skip rows with only NaNs
                    continue

                array[row] = np.interp(np.arange(array.shape[1]), np.where(valid_mask)[0], array[row, valid_mask])
        
                # If NaNs at the start, fill with the first non-NaN value
                first_valid_idx = valid_mask.argmax()
                array[row, :first_valid_idx] = array[row, first_valid_idx]
                
                # If NaNs at the end, fill with the last non-NaN value
                last_valid_idx = len(array[row]) - valid_mask[::-1].argmax() - 1
                array[row, last_valid_idx+1:] = array[row, last_valid_idx]
        
        return temp_GEM, sal_GEM
    
    def get_inputs(self, idx):
        sst_inputs = self.TEMP[0, idx]
        ssh_inputs = self.SSH[idx]
        return sst_inputs, ssh_inputs
    
    def get_lat_lon_date(self, idx):
        lat = self.LAT[idx]
        lon = self.LON[idx]
        date = self.TIME[idx]
        return lat, lon, date
    
    def get_pca_variance(self):
        """
        Get the concatenated vector of the variance represented by each PC of the temperature and salinity datasets.

        Returns:
        - numpy.ndarray: Concatenated vector of variances for temperature and salinity PCs.
        """
        temp_variance = self.pca_temp.explained_variance_
        sal_variance = self.pca_sal.explained_variance_
        concatenated_variance = np.concatenate([temp_variance, sal_variance])
        return concatenated_variance

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

    def __init__(self, input_dim=1, layers_config=[512, 256], output_dim=30, dropout_prob = 0.5):
        super(PredictionModel, self).__init__()
        
        # Construct layers based on the given configuration
        layers = []
        prev_dim = input_dim
        for neurons in layers_config:
            layers.append(nn.Linear(prev_dim, neurons))
            layers.append(nn.ReLU())
            # layers.append(nn.Tanh())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob)) # added dropout
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
        # print(f"x shape: {x.shape}")
        return self.model(x)
    
def split_dataset(dataset, train_size, val_size, test_size, batch_size=32, use_batches=True):
    """
    Splits the dataset into training, validation, and test sets.
    
    Parameters:
    - dataset: The entire dataset to be split.
    - train_size, val_size, test_size: Proportions for splitting. They should sum to 1.
    
    Returns:
    - train_dataset, val_dataset, test_dataset: Split datasets.
    """
    total_size = len(dataset)
    train_len = int(total_size * train_size)
    val_len = int(total_size * val_size)
    test_len = total_size - train_len - val_len

    return random_split(dataset, [train_len, val_len, test_len])

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, patience=10):
    """
    Train the model with early stopping and CUDA support.

    Parameters:
    - model: the PyTorch model.
    - train_loader: the DataLoader for training data.
    - val_loader: the DataLoader for validation data.
    - criterion: the loss function.
    - optimizer: the optimizer.
    - device: device to which data and model should be moved before training.
    - epochs: number of total epochs to train.
    - patience: number of epochs to wait for improvement before stopping.

    Returns:
    - model: trained model.
    """
    model.to(device)
    best_val_loss = float('inf')
    best_weights = None  # To store best model weights
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        
        # Validation loss
        avg_val_loss = evaluate_model(model, val_loader, criterion, device)
        
        if epoch == 0 or epoch%10==9:
            print(f"Epoch [{(epoch + 1):4.0f}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Patience left: {(100*(patience - no_improve_count)/patience):3.0f}% | Best: {best_val_loss:.4f}")

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = model.state_dict()  # Save the model weights
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at Epoch {epoch + 1}")
                break

    model.load_state_dict(best_weights)  # Load the best model weights
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

def get_predictions(model, dataloader, device):
    """
    Get model's predictions on the provided data with CUDA support.

    Parameters:
    - model: the PyTorch model.
    - dataloader: the DataLoader for the data.
    - device: device to which data and model should be moved before getting predictions.

    Returns:
    - predictions: model's predictions.
    """
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs[0].to(device)  # Getting only the input features, ignoring labels
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)

def get_inputs(dataloader, device):
    """
    Get inputs from the provided dataloader with CUDA support.

    Parameters:
    - dataloader: the DataLoader for the data.
    - device: device to which data should be moved.

    Returns:
    - all_inputs: list of inputs from the dataloader.
    """
    all_inputs = []

    for inputs in dataloader:
        inputs = inputs[0].to(device)  # Getting only the input features, ignoring labels
        all_inputs.extend(inputs.cpu().numpy())

    return np.array(all_inputs)

def predict_with_numpy(model, numpy_input, device="cuda"):
    # Convert numpy array to tensor
    tensor_input = torch.tensor(numpy_input, dtype=torch.float32)
    
    # Check if CUDA is available and move tensor to the appropriate device
    if device == "cuda" and torch.cuda.is_available():
        tensor_input = tensor_input.cuda()
        model = model.cuda()
    
    # Make sure the model is in evaluation mode
    model.eval()
    
    # Make predictions
    with torch.no_grad():
        predictions = model(tensor_input)
    
    # Convert predictions back to numpy (if on GPU, move to CPU first)
    numpy_predictions = predictions.cpu().numpy()
    
    return numpy_predictions

def inverse_transform(pcs, pca_temp, pca_sal, n_components):
    """
    Inverse the PCA transformation.

    Args:
    - pcs (numpy.ndarray): Concatenated PCA components for temperature and salinity.
    - pca_temp, pca_sal: PCA models for temperature and salinity respectively.
    - n_components (int): Number of PCA components.

    Returns:
    - tuple: Inversed temperature and salinity profiles.
    """
    temp_profiles = pca_temp.inverse_transform(pcs[:, :n_components]).T
    sal_profiles = pca_sal.inverse_transform(pcs[:, n_components:]).T
    return temp_profiles, sal_profiles

## Custom Loss
class WeightedMSELoss(nn.Module):
    """
    The code defines several loss functions for use in a PCA-based model, including a weighted MSE loss
    and a combined PCA loss.
    
    @param n_components The parameter `n_components` represents the number of principal components to
    consider in the PCA loss. It determines the dimensionality of the PCA space for both temperature and
    salinity profiles.
    @param device The "device" parameter in the code refers to the device on which the computations will
    be performed. It can be either "cuda" for GPU acceleration or "cpu" for CPU computation.
    @param weights The "weights" parameter is a list of weights that are used to assign different
    importance to each element in the loss calculation. These weights are used in the WeightedMSELoss
    class to multiply the squared differences between the predicted and true values. The weights are
    normalized so that they sum up to
    
    @return The `forward` method of the `CombinedPCALoss` class returns the combined loss, which is the
    sum of the PCA loss and the weighted MSE loss.
    """
    def __init__(self, weights, device):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)

    def forward(self, input, target):
        squared_diff = (input - target) ** 2
        weighted_squared_diff = self.weights * squared_diff
        loss = weighted_squared_diff.mean()
        return loss

def genWeightedMSELoss(n_components, device, weights):
    # Normalizing weights so they sum up to 1
    normalized_weights = weights / np.sum(weights)
    return WeightedMSELoss(normalized_weights, device)
   
class PCALoss(nn.Module):
    def __init__(self, temp_pca, sal_pca, n_components):
        super(PCALoss, self).__init__()
        self.n_components = n_components
        self.n_samples = len(temp_pca)
        # convert PCS to tensors
        self.temp_pca_components = torch.nn.Parameter(torch.from_numpy(temp_pca.temp_pcs).float().to('cuda'), requires_grad=False)
        self.sal_pca_components = torch.nn.Parameter(torch.from_numpy(sal_pca.sal_pcs).float().to('cuda'), requires_grad=False)

    def inverse_transform(self, pcs, pca_components):
        # Perform the inverse transform using PyTorch operations
        return torch.mm(pcs, pca_components) # + pca_mean

    def forward(self, pcs, targets):
        # Split the predicted and true pcs for temp and sal
        pred_temp_pcs, pred_sal_pcs = pcs[:, :self.n_components], pcs[:, self.n_components:]
        true_temp_pcs, true_sal_pcs = targets[:, :self.n_components], targets[:, self.n_components:]
        
        # Inverse transform the PCA components to get the profiles
        pred_temp_profiles = self.inverse_transform(pred_temp_pcs, self.temp_pca_components)
        pred_sal_profiles = self.inverse_transform(pred_sal_pcs, self.sal_pca_components)
        true_temp_profiles = self.inverse_transform(true_temp_pcs, self.temp_pca_components)
        true_sal_profiles = self.inverse_transform(true_sal_pcs, self.sal_pca_components)
        
        # Calculate the Mean Squared Error between the predicted and true profiles
        mse_temp = nn.functional.mse_loss(pred_temp_profiles, true_temp_profiles)
        mse_sal = nn.functional.mse_loss(pred_sal_profiles, true_sal_profiles)
        
        # Combine the MSE for temperature and salinity
        total_mse = (mse_temp/(8**2) + mse_sal/(35**2))/self.n_samples # divide by the square of the mean values
        return total_mse
    
class CombinedPCALoss(nn.Module):
    def __init__(self, temp_pca, sal_pca, n_components, weights, device):
        super(CombinedPCALoss, self).__init__()
        self.pca_loss = PCALoss(temp_pca, sal_pca, n_components)
        self.weighted_mse_loss = genWeightedMSELoss(n_components, device, weights)

    def forward(self, pcs, targets):
        # Calculate the PCA loss
        pca_loss = self.pca_loss(pcs, targets)

        # Calculate the weighted MSE loss
        weighted_mse_loss = self.weighted_mse_loss(pcs, targets)

        # Combine the losses
        # You may need to adjust the scaling factor to balance the two losses
        combined_loss = 5.5*pca_loss + 3.6*weighted_mse_loss
        return combined_loss
    
class maxPCALoss(nn.Module):
    def __init__(self, temp_pca, sal_pca, n_components):
        super(maxPCALoss, self).__init__()
        self.n_components = n_components
        self.n_samples = len(temp_pca)
        # Convert PCA components to tensors
        self.temp_pca_components = torch.nn.Parameter(torch.from_numpy(temp_pca.temp_pcs).float().to('cuda'), requires_grad=False)
        self.sal_pca_components = torch.nn.Parameter(torch.from_numpy(sal_pca.sal_pcs).float().to('cuda'), requires_grad=False)

    def inverse_transform(self, pcs, pca_components):
        # Perform the inverse transform using PyTorch operations
        return torch.mm(pcs, pca_components)

    def forward(self, pcs, targets):
        # Split the predicted and true pcs for temp and sal
        pred_temp_pcs, pred_sal_pcs = pcs[:, :self.n_components], pcs[:, self.n_components:]
        true_temp_pcs, true_sal_pcs = targets[:, :self.n_components], targets[:, self.n_components:]
        
        # Inverse transform the PCA components to get the profiles
        pred_temp_profiles = self.inverse_transform(pred_temp_pcs, self.temp_pca_components)
        pred_sal_profiles = self.inverse_transform(pred_sal_pcs, self.sal_pca_components)
        true_temp_profiles = self.inverse_transform(true_temp_pcs, self.temp_pca_components)
        true_sal_profiles = self.inverse_transform(true_sal_pcs, self.sal_pca_components)
        
        # Calculate the maximum absolute difference for temperature and salinity
        max_diff_temp = torch.max(torch.abs(pred_temp_profiles - true_temp_profiles))
        max_diff_sal = torch.max(torch.abs(pred_sal_profiles - true_sal_profiles))
        
        # Combine the maximum differences for temperature and salinity
        total_max_diff = max_diff_temp/8 + max_diff_sal/35
        return total_max_diff

def visualize_combined_results(true_values, gem_temp, gem_sal, predicted_values, sst_values, ssh_values, min_depth = 20, max_depth=2000, num_samples=5):
    # TODO: add date to plot
    """
    Visualize the true vs. predicted vs. GEM approximated values for a sample of profiles and their differences.

    Parameters:
    - true_values: ground truth temperature and salinity profiles.
    - gem_temp: GEM approximated temperature profiles.
    - gem_sal: GEM approximated salinity profiles.
    - predicted_values: model's predicted temperature and salinity profiles.
    - sst_values: Sea Surface Temperature values for each profile.
    - ssh_values: Sea Surface Height (adt) values for each profile.
    - num_samples: number of random profiles to visualize.

    Returns:
    - None (plots the results).
    """
    n_depths = max_depth + 1
    depth_levels = np.arange(min_depth, n_depths)
    population_size = true_values.shape[2]

    if num_samples == population_size:
        indices = np.arange(num_samples)
    else:
        indices = np.random.choice(int(population_size), num_samples, replace=False)

    for idx in indices:
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))

        # First row: Actual Profiles
        # Temperature profile
        axs[0][0].plot(gem_temp[idx], depth_levels, 'g', label="GEM Profile", alpha = 0.75)
        axs[0][0].plot(predicted_values[0][:, idx], depth_levels, 'r', label="NN Profile", alpha = 0.75)
        axs[0][0].plot(true_values[:,0, idx], depth_levels, 'k', label="Target", linewidth = 0.7)
        axs[0][0].invert_yaxis()
        axs[0][0].set_title(f"Temperature Profile")
        axs[0][0].set_ylabel("Depth")
        axs[0][0].set_xlabel("Temperature")
        axs[0][0].legend(loc='lower right')
        axs[0][0].grid(color='gray', linestyle='--', linewidth=0.5)

        # Salinity profile
        axs[0][1].plot(gem_sal[idx], depth_levels, 'g', label="GEM Profile", alpha = 0.75)
        axs[0][1].plot(predicted_values[1][:, idx], depth_levels, 'r', label="NN Profile", alpha = 0.75)
        axs[0][1].plot(true_values[:,1, idx], depth_levels, 'k', label="Target", linewidth = 0.7)
        axs[0][1].invert_yaxis()
        axs[0][1].set_title(f"Salinity Profile")
        axs[0][1].set_ylabel("Depth")
        axs[0][1].set_xlabel("Salinity")
        axs[0][1].legend(loc='lower right')
        axs[0][1].grid(color='gray', linestyle='--', linewidth=0.5)

        # Second row: Differences
        gem_temp_dif = gem_temp[idx]-true_values[:,0, idx]
        gem_sal_dif = gem_sal[idx]-true_values[:,1, idx]
        nn_temp_dif = predicted_values[0][:, idx]-true_values[:,0, idx]
        nn_sal_dif = predicted_values[1][:, idx]-true_values[:,1, idx]
        
        axs[1][0].plot(gem_temp_dif, depth_levels, 'g', label="GEM Profile", alpha = 0.75)
        axs[1][0].plot(nn_temp_dif, depth_levels, 'r', label="NN Profile", alpha = 0.75)
        axs[1][0].axvline(0, color='k', linestyle='--', linewidth=0.5)
        axs[1][0].invert_yaxis()
        axs[1][0].set_title(f"Temperature Differences")
        axs[1][0].set_ylabel("Depth")
        axs[1][0].set_xlabel("RMSE (°C)")
        axs[1][0].legend(loc='best')
        axs[1][0].grid(color='gray', linestyle='--', linewidth=0.5)

        # Salinity difference
        axs[1][1].plot(gem_sal_dif, depth_levels, 'g', label="GEM Profile", alpha = 0.75)
        axs[1][1].plot(nn_sal_dif, depth_levels, 'r', label="NN Profile", alpha = 0.75)
        axs[1][1].axvline(0, color='k', linestyle='--', linewidth=0.5)
        axs[1][1].invert_yaxis()
        axs[1][1].set_title(f"Salinity Differences")
        axs[1][1].set_ylabel("Depth")
        axs[1][1].set_xlabel("RMSE (PSU²)")
        axs[1][1].legend(loc='best')
        axs[1][1].grid(color='gray', linestyle='--', linewidth=0.5)

        gem_temp_rmse_individual = np.sqrt(np.mean(gem_temp_dif**2))
        gem_sal_rmse_individual = np.sqrt(np.mean(gem_sal_dif**2))
        nn_temp_rmse_individual = np.sqrt(np.mean(nn_temp_dif**2))
        nn_sal_rmse_individual = np.sqrt(np.mean(nn_sal_dif**2))

        accuracy_gain_temp = 100*(gem_temp_rmse_individual - nn_temp_rmse_individual) / gem_temp_rmse_individual
        accuracy_gain_sal = 100*(gem_sal_rmse_individual - nn_sal_rmse_individual) / gem_sal_rmse_individual

        # Add sst, ssh and accuracy gain information to the suptitle
        plt.suptitle(f"Profile {idx} - SST: {sst_values[idx]:.2f}, SSH (adt): {ssh_values[idx]:.2f}\n"
                     f"T prediction improvement: {accuracy_gain_temp:.2f}%, S prediction improvement: {accuracy_gain_sal:.2f}%", fontsize=16)

        plt.tight_layout()
        plt.show()

    # RMSE Calculations and Accuracy Gain
    gem_temp_errors = (gem_temp.T - true_values[:, 0, :]) ** 2
    gem_sal_errors = (gem_sal.T - true_values[:, 1, :]) ** 2

    nn_temp_errors = (predicted_values[0][:, :] - true_values[:, 0, :]) ** 2
    nn_sal_errors = (predicted_values[1][:, :] - true_values[:, 1, :]) ** 2
        
    gem_temp_rmse = np.sqrt(np.mean(gem_temp_errors))
    gem_sal_rmse = np.sqrt(np.mean(gem_sal_errors))

    nn_temp_rmse = np.sqrt(np.mean(nn_temp_errors))
    nn_sal_rmse = np.sqrt(np.mean(nn_sal_errors))

    accuracy_gain_temp = 100*(gem_temp_rmse-nn_temp_rmse)/gem_temp_rmse
    accuracy_gain_sal = 100*(gem_sal_rmse-nn_sal_rmse)/gem_sal_rmse
    
    print(f"NN Average temperature RMSE: {nn_temp_rmse:.3f}°C")
    print(f"NN Average salinity RMSE: {nn_sal_rmse:.3f} PSU")
    print(f"GEM Average temperature RMSE: {gem_temp_rmse:.3f}°C")
    print(f"GEM Average salinity RMSE: {gem_sal_rmse:.3f} PSU")
    
    # print(f"Average temperature accuracy gain: {accuracy_gain_temp:.3f}% (entire depth range)")
    # print(f"Average salinity accuracy gain: {accuracy_gain_sal:.3f}% (entire depth range)")

    gem_temp_errors = (gem_temp.T[150:,:] - true_values[150:, 0, :]) ** 2
    gem_sal_errors = (gem_sal.T[150:,:] - true_values[150:, 1, :]) ** 2

    nn_temp_errors = (predicted_values[0][150:, :] - true_values[150:, 0, :]) ** 2
    nn_sal_errors = (predicted_values[1][150:, :] - true_values[150:, 1, :]) ** 2

    gem_temp_rmse = np.sqrt(np.mean(gem_temp_errors))
    gem_sal_rmse = np.sqrt(np.mean(gem_sal_errors))

    nn_temp_rmse = np.sqrt(np.mean(nn_temp_errors))
    nn_sal_rmse = np.sqrt(np.mean(nn_sal_errors))

    accuracy_gain_temp = 100*(gem_temp_rmse-nn_temp_rmse)/gem_temp_rmse
    accuracy_gain_sal = 100*(gem_sal_rmse-nn_sal_rmse)/gem_sal_rmse
    # print(f"Average temperature accuracy gain: {accuracy_gain_temp:.3f}% (150m to max depth)")
    # print(f"Average salinity accuracy gain: {accuracy_gain_sal:.3f}% (150m to max depth)")

def matlab2datetime(matlab_datenum):
    python_datetime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days=366)
    return python_datetime

def filter_by_season(data, dates, season):
    SEASONS = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Fall": [9, 10, 11]
    }
    months = SEASONS[season]
    indices = [i for i, date in enumerate(dates) if matlab2datetime(date).month in months]
    return [data[i] for i in indices]

def seasonal_plots(lat_val, lon_val, dates_val, original_profiles, gem_temp, gem_sal, val_predictions, sst_inputs, ssh_inputs, max_depth, num_samples):
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    total_samples = len(lat_val)
    indexes = np.arange(total_samples)
    for season in seasons:
        idx = np.array(filter_by_season(indexes, dates_val, season))
        print(season)
        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 10))
        ax.set_global()
        ax.coastlines()
        # Setting plot limits to the Gulf of Mexico region
        ax.set_extent([-98, -80, 18, 31])
        scatter = ax.scatter(lon_val[idx], lat_val[idx], c=ssh_inputs[idx], cmap='viridis', edgecolors='k', linewidth=0.5, transform=ccrs.PlateCarree())
        cbar = plt.colorbar(scatter, ax=ax, orientation="vertical", pad=0.02, shrink=1)
        cbar.set_label("SSH Value")

        ax.set_title(f"{season} profiles in validation", fontsize=16)
        plt.show()
        
        # Now plot some samples from this season
        sliced_val_pred = [array[:, idx] for array in val_predictions]
        visualize_combined_results(original_profiles[:,:, idx], gem_temp[idx], gem_sal[idx], sliced_val_pred, sst_inputs[idx], ssh_inputs[idx], max_depth = max_depth, num_samples=num_samples)
     
def plot_relative_errors(true_values, gem_temp, gem_sal, predicted_values, min_depth = 20, max_depth=2000):
    depth_levels = np.arange(min_depth, max_depth+1)
    n_depths = len(depth_levels)
    
    # Helper function to compute RMSE, gain, and their respective min/max for both temperature and salinity
    def compute_rmse_gain(gem_errors, nn_errors):
        gem_rmse_depth = np.sqrt(np.mean(gem_errors, axis=1))
        nn_rmse_depth = np.sqrt(np.mean(nn_errors, axis=1))
        
        gem_std_depth = np.std(np.sqrt(gem_errors), axis=1)
        nn_std_depth = np.std(np.sqrt(nn_errors), axis=1)
        
        gain_depth = 100*(gem_rmse_depth - nn_rmse_depth) / gem_rmse_depth
        gain_std_depth = 100*(gem_std_depth - nn_std_depth) / gem_std_depth
        
        return gem_rmse_depth, nn_rmse_depth, gem_std_depth, nn_std_depth, gain_depth, gain_std_depth
    
    # Using helper function for temperature and salinity
    results_temp = compute_rmse_gain(gem_temp_errors, nn_temp_errors)
    results_sal = compute_rmse_gain(gem_sal_errors, nn_sal_errors)
    
    # Unpacking results
    gem_temp_rmse_depth, nn_temp_rmse_depth, gem_temp_std_depth, nn_temp_std_depth, gain_temp_depth, gain_temp_std_depth = results_temp
    gem_sal_rmse_depth, nn_sal_rmse_depth, gem_sal_std_depth, nn_sal_std_depth, gain_sal_depth, gain_sal_std_depth = results_sal

    # Visualization
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
        
    # Temperature RMSE
    axs[0,0].fill_betweenx(depth_levels, nn_temp_rmse_depth - nn_temp_std_depth, nn_temp_rmse_depth + nn_temp_std_depth, color='r', alpha=0.1, label='NN average RMSE ± std')
    axs[0,0].fill_betweenx(depth_levels, gem_temp_rmse_depth - gem_temp_std_depth, gem_temp_rmse_depth + gem_temp_std_depth, color='g', alpha=0.1, label='GEM average RMSE ± std')
    axs[0,0].plot(nn_temp_rmse_depth, depth_levels, 'r-', label='NN average RMSE')
    axs[0,0].plot(gem_temp_rmse_depth, depth_levels, 'g-', label='GEM average RMSE')
    axs[0,0].invert_yaxis()
    axs[0,0].legend(loc='lower right')
    axs[0,0].set_title("Temperature RMSE per Depth")
    axs[0,0].set_ylabel("Depth")
    axs[0,0].set_xlabel("RMSE")
    #grid on
    

    # Gain in Temperature Prediction
    axs[1,0].plot(gain_temp_depth, depth_levels, 'b-', label='Average Gain')
    axs[1,0].fill_betweenx(depth_levels, gain_temp_depth - gain_temp_std_depth, gain_temp_depth + gain_temp_std_depth, color='b', alpha=0.1, label='±1 std')
    axs[1,0].axvline(0, color='k', linestyle='--')
    axs[1,0].invert_yaxis()
    axs[1,0].legend(loc='lower right')
    axs[1,0].set_title("Accuracy gain in Temperature, Prediction by Depth")
    axs[1,0].set_ylabel("Depth")
    axs[1,0].set_xlabel("Gain (%)")
    
    # Salinity RMSE
    axs[0,1].fill_betweenx(depth_levels, gem_sal_rmse_depth - gem_sal_std_depth, gem_sal_rmse_depth + gem_sal_std_depth, color='g', alpha=0.1, label='GEM average RMSE ±1 std')
    axs[0,1].fill_betweenx(depth_levels, nn_sal_rmse_depth - nn_sal_std_depth, nn_sal_rmse_depth + nn_sal_std_depth, color='r', alpha=0.1, label='NN average RMSE ±1 std')
    axs[0,1].plot(nn_sal_rmse_depth, depth_levels, 'r-', label='NN average RMSE')
    axs[0,1].plot(gem_sal_rmse_depth, depth_levels, 'g-', label='GEM average RMSE')
    axs[0,1].invert_yaxis()
    axs[0,1].legend(loc='lower right')
    axs[0,1].set_title("Salinity RMSE per Depth")
    axs[0,1].set_ylabel("Depth")
    axs[0,1].set_xlabel("RMSE")
    
    # Gain in Salinity Prediction
    axs[1,1].plot(gain_sal_depth, depth_levels, 'm-', label='Average Gain')
    axs[1,1].fill_betweenx(depth_levels, gain_sal_depth - gain_sal_std_depth, gain_sal_depth + gain_sal_std_depth, color='m', alpha=0.1, label='±1 std')
    axs[1,1].axvline(0, color='k', linestyle='--')
    axs[1,1].invert_yaxis()
    axs[1,1].legend(loc='lower right')
    axs[1,1].set_title("Accuracy gain in Salinity Prediction, by Depth")
    axs[1,1].set_ylabel("Depth")
    axs[1,1].set_xlabel("Gain (%)")

    # Add grids to all subplots
    for ax in axs.ravel():
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    
    # print(f"Average temperature RMSE: {np.mean(nn_temp_rmse_depth):.3f} ±{np.mean(nn_temp_std_depth):.3f}")
    # print(f"Average temperature RMSE: {np.mean(nn_temp_rmse_depth):.3f} ±{np.mean(nn_sal_std_depth):.3f}")
    
    file_name = "ARGO_X_GEM_RMSE_metrics.pdf"

    for ax in axs.flat:
        for c in ax.collections:
            c.set_rasterized(True)
    fig.savefig(file_name, bbox_inches='tight', dpi=150)

    print(f"Saved plots to {file_name}")
    
def calculate_average_rmse_per_bin(lon_bins, lat_bins, lon_val, lat_val, rmse_values):
    avg_rmse_grid = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
    num_prof_grid = np.zeros((len(lat_bins)-1, len(lon_bins)-1))

    for i in range(len(lon_bins)-1):
        for j in range(len(lat_bins)-1):
            # Find points that fall into the current bin
            in_bin = (lon_val >= lon_bins[i]) & (lon_val < lon_bins[i+1]) & (lat_val >= lat_bins[j]) & (lat_val < lat_bins[j+1])
            # Calculate average RMSE for points in the bin
            avg_rmse_grid[j, i] = np.mean(rmse_values[in_bin])
            num_prof_grid[j, i] = np.sum(in_bin)

    return avg_rmse_grid, num_prof_grid

def plot_rmse_maps(lon_bins, lat_bins, avg_rmse_nn, avg_rmse_gem, num_prof, title_prefix):
    # Calculate the difference
    avg_rmse_gain = avg_rmse_nn - avg_rmse_gem

    # Calculate centers of the bins
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
    
    vmin = 0

    if title_prefix == "Temperature":
        cmap = "YlOrRd"
        units = "(°C)"
        vmax = 3
    else:
        cmap = "PuBuGn"
        units = "(PSU)"
        vmax = 0.35
        
    # Create subplot grid
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 15), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the maps
    plot_rmse_on_ax(ax1, lon_centers, lat_centers, avg_rmse_nn, num_prof, f"NN Average RMSE - {title_prefix}")

    pcm = ax1.pcolormesh(lon_centers, lat_centers, avg_rmse_nn, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax1, orientation="vertical", pad=0.04, fraction=0.465*(1/15), label=f"Average RMSE {units}")
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    # Set x and y ticks, 
    ax1.set_xticks(np.arange(-99, -81, 1))
    ax1.set_yticks(np.arange(18, 30, 1))
    #add grid
    ax1.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.show()
    
def plot_rmse_on_ax(ax, lon_centers, lat_centers, avg_rmse_grid, num_prof, title):
    ax.set_extent([-99, -81, 18, 30])  # Set to your area of interest
    ax.coastlines()

    pcm = ax.pcolormesh(lon_centers, lat_centers, avg_rmse_grid, cmap='coolwarm', vmin=-3, vmax=3)
    ax.set_title(title, fontsize=18)

    # Annotate each cell with the average RMSE value
    for i, lon in enumerate(lon_centers):
        for j, lat in enumerate(lat_centers):
            value = avg_rmse_grid[j, i]
            number = num_prof[j, i]
            if not np.isnan(value):  # Check if the value is not NaN, and if there are more than 2 profiles in the bin
                ax.text(lon, lat+0.2, f'{number:.0f}', color='gray', ha='center', va='center', fontsize=8, transform=ccrs.PlateCarree())
                ax.text(lon, lat-0.2, f'{value:.2f}', color='black', ha='center', va='center', fontsize=8, transform=ccrs.PlateCarree())

def plot_comparison_maps(lon_centers, lat_centers, avg_rmse_nn, avg_rmse_gdem, title_prefix):
    # Calculate the difference
    avg_rmse_diff = avg_rmse_gdem - avg_rmse_nn

    # Set up color maps and limits
    if title_prefix == "temperature":
        cmap = "YlOrRd"
        units = "(°C)"
        vmax = 3
    else:
        cmap = "PuBuGn"
        units = "(PSU)"
        vmax = 0.35

    # Custom colormap for difference plot
    diff_cmap = plt.get_cmap('coolwarm')
    norm_diff = plt.Normalize(-vmax, vmax)

    # Create subplot grid
    fig, axes = plt.subplots(1, 3, figsize=(30, 15), subplot_kw={'projection': ccrs.PlateCarree()})

    # Titles for each subplot
    titles = [f"NN (ours)", f"ISOP", f"Difference (ISOP - NN)"]

    # Function to add values to bins
    def annotate_bins(ax, data):
        for i, lon in enumerate(lon_centers):
            for j, lat in enumerate(lat_centers):
                value = data[j, i]
                if not np.isnan(value):
                    ax.text(lon, lat, f'{value:.2f}', color='black', ha='center', va='center', fontsize=8, transform=ccrs.PlateCarree())

    # Plotting NN RMSE, ISOP RMSE, and Difference
    for i, (data, title) in enumerate(zip([avg_rmse_nn, avg_rmse_gdem, avg_rmse_diff], titles)):
        if i < 2:
            pcm = axes[i].pcolormesh(lon_centers, lat_centers, data, cmap=cmap, vmin=0, vmax=vmax)
        else:  # For the difference plot
            pcm_diff = axes[i].pcolormesh(lon_centers, lat_centers, data, cmap=diff_cmap, norm=norm_diff)
        #bold titles
        axes[i].set_title(title, weight='bold')
        axes[i].coastlines()
        annotate_bins(axes[i], data)
        axes[i].set_xticks(np.arange(-99, -81, 1))
        axes[i].set_yticks(np.arange(18, 30, 1))
        axes[i].grid(color='gray', linestyle='--', linewidth=0.5)

    # Adding colorbar for the first two plots
    fig.colorbar(pcm, ax=axes[:2], orientation="vertical", pad=0.04, fraction=0.0145, label=f"Average RMSE {units}")

    # Adding colorbar for the difference plot
    fig.colorbar(pcm_diff, ax=axes[2], orientation="vertical", pad=0.04, fraction=0.031, label=f"Difference in RMSE {units}")
    fig.suptitle(f"Average RMSE for synthetic {title_prefix} profiles per region", fontsize=24, y=0.69)
    plt.show()
    
#%%
if __name__ == "__main__":
    # Configurable parameters
    data_path = "/unity/g2/jmiranda/SubsurfaceFields/Data/ARGO_GoM_20220920.mat"
    
    bin_size = 1 # bin size in degrees
    n_components = 15
    n_runs = 1
    layers_config = [512, 512]
    batch_size = 300
    min_depth = 20
    max_depth = 1800
    dropout_prob = 0.2
    epochs = 8000
    patience = 500
    learning_rate = 0.001
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15
    input_params = {
        "timecos": True,
        "timesin": True,
        "latcos":  True,
        "latsin":  True,
        "loncos":  True,
        "lonsin":  True,
        "sat": True,  # Use satellite data?
        "sst": True,  # First value of temperature
        "sss": True,
        "ssh": True
    }
    num_samples = 1 #profiles that will be plotted
    # Define the path of the pickle file
    pickle_file = 'config_dataset_full.pkl'

    if os.path.exists(pickle_file):
        # Load data from the pickle file
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)
            full_dataset = data['full_dataset']
            full_dataset.n_components = n_components
            full_dataset.min_depth = min_depth
            full_dataset.max_depth = max_depth
            full_dataset.input_params = input_params
            full_dataset.reload()
    else:
        # Load and split data
        full_dataset = TemperatureSalinityDataset(path=data_path, n_components=n_components, input_params=input_params, min_depth=min_depth, max_depth=max_depth)

        # Save data to the pickle file
        with open(pickle_file, 'wb') as file:
            data = {
                'min_depth' : min_depth,
                'max_depth': max_depth,
                'epochs': epochs,
                'patience': patience,
                'n_components': n_components,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'dropout_prob': dropout_prob,
                'layers_config': layers_config,
                'input_params': input_params,
                'train_size': train_size,
                'val_size': val_size,
                'test_size': test_size,
                'full_dataset': full_dataset
            }
            pickle.dump(data, file)

    train_dataset, val_dataset, test_dataset = split_dataset(full_dataset, train_size, val_size, test_size)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute the input dimension dynamically
    input_dim = sum(val for val in input_params.values()) - 1*input_params['sat']
   
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Loss function using the variance of the PCA components as weights
    weights = full_dataset.get_pca_variance()
    
    # Set the appropriate loss
    # criterion = genWeightedMSELoss(n_components, device, weights)
    # criterion = PCALoss(temp_pca=train_dataset.dataset, sal_pca=train_dataset.dataset, n_components=n_components)
    # criterion = maxPCALoss(temp_pca=train_dataset.dataset, sal_pca=train_dataset.dataset, n_components=n_components)
    criterion = CombinedPCALoss(temp_pca=train_dataset.dataset, 
                            sal_pca=train_dataset.dataset, 
                            n_components=n_components, 
                            weights=weights, 
                            device=device)
    
    # print parameters and dataset size
    true_params = [param for param, value in input_params.items() if value]
    def printParams():   
        print(f"\nNumber of profiles: {len(full_dataset)}")
        print("Parameters used:", ", ".join(true_params))
        print(f"Min depth: {min_depth}, Max depth: {max_depth}")
        print(f"Number of components used: {n_components} x2")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Dropout probability: {dropout_prob}")
        print(f'Train/test/validation split: {train_size}/{test_size}/{val_size}')
        print(f"Layer configuration: {layers_config}\n")
    
    printParams()
    
    for run in enumerate(np.arange(n_runs)):
        print(f"Run {run[0]}/{n_runs}")
        # Model

        model = PredictionModel(input_dim=input_dim, layers_config=layers_config, output_dim=n_components*2, dropout_prob = dropout_prob)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Training
        trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience)
        
        # Test evaluation
        test_loss = evaluate_model(trained_model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}")

        save_model_path = "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/saved_models/model_"
        save_model_path += str(test_loss) + "_"
        suffix = ".pth"
        if input_params['sat']:
            suffix = "_sat.pth"
        
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        save_model_path += now_str + suffix
        
        torch.save(trained_model, save_model_path)
        
    print("Statistics from the last run:")
    
    # Get predictions for the validation dataset
    val_predictions_pcs = get_predictions(trained_model, val_loader, device)
    # Accessing the original dataset for inverse_transform
    val_predictions = val_dataset.dataset.inverse_transform(val_predictions_pcs)
    
    # load ISOP results
    file_path_new = '/unity/g2/jmiranda/SubsurfaceFields/Data/ISOP1_rmse_bias_1deg_maps.nc'
    data_new = xr.open_dataset(file_path_new)

    # Create bins for longitude and latitude
    lon_bins = np.arange(np.min(data_new.lon) - 0.5, np.max(data_new.lon) + 1.5, 1)
    lat_bins = np.arange(np.min(data_new.lat) - 0.5, np.max(data_new.lat) + 1.5, 1)

    # Calculate centers of the bins
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2

    # Initialize a NaN array for the number of profiles
    num_prof = np.full((len(lat_centers), len(lon_centers)), np.nan)

    # Extracting RMSE data and ensuring it matches the dimensions of the bins
    avg_rmse_isop_t = data_new['t_rmse_syn']
    avg_rmse_isop_s = data_new['s_rmse_syn']
    
    avg_rmse_gdem_t = data_new['t_rmse_gdem']
    avg_rmse_gdem_s = data_new['s_rmse_gdem']
    
    def viz():
        subset_indices = val_loader.dataset.indices

        # For original profiles
        original_profiles = val_dataset.dataset.get_profiles(subset_indices, pca_approx=False)

        # For PCA approximated profiles
        pca_approx_profiles = val_dataset.dataset.get_profiles(subset_indices, pca_approx=True)
        
        #Load GEM polyfits to benchmark
        with open("pca_temp.pkl", "wb") as f:
            pickle.dump(full_dataset.pca_temp, f)

        with open("pca_sal.pkl", "wb") as f:
            pickle.dump(full_dataset.pca_sal, f)
            
        sst_inputs, ssh_inputs = val_dataset.dataset.get_inputs(subset_indices)
        
        gem_temp, gem_sal = val_dataset.dataset.get_gem_profiles(subset_indices)
        
        lat_val, lon_val, dates_val = val_dataset.dataset.get_lat_lon_date(subset_indices)

        visualize_combined_results(original_profiles, gem_temp, gem_sal, val_predictions, sst_inputs, ssh_inputs, min_depth=min_depth, max_depth = max_depth, num_samples=num_samples)
        
        printParams()
        
        print("Let's investigate how the method compares against vanilla GEM with in-situ SSH")
            # RMSE Calculations and Accuracy Gain
        gem_temp_errors = (gem_temp.T - original_profiles[:, 0, :]) ** 2
        gem_sal_errors = (gem_sal.T - original_profiles[:, 1, :]) ** 2

        nn_temp_errors = (val_predictions[0][:, :] - original_profiles[:, 0, :]) ** 2
        nn_sal_errors = (val_predictions[1][:, :] - original_profiles[:, 1, :]) ** 2

        gem_temp_rmse = np.sqrt(np.mean(gem_temp_errors, axis = 0))
        gem_sal_rmse = np.sqrt(np.mean(gem_sal_errors, axis = 0))

        nn_temp_rmse = np.sqrt(np.mean(nn_temp_errors, axis = 0))
        nn_sal_rmse = np.sqrt(np.mean(nn_sal_errors, axis = 0))
        
        accuracy_gain_temp = 100*(gem_temp_rmse-nn_temp_rmse)/gem_temp_rmse
        accuracy_gain_sal = 100*(gem_sal_rmse-nn_sal_rmse)/gem_sal_rmse
        
        # plot_relative_errors(original_profiles, gem_temp, gem_sal, val_predictions, min_depth=min_depth, max_depth = max_depth)
        
        # Now for the satGEM
        print("Now let's see how it performs by using satellite SSH with GEM")
        gem_temp, gem_sal = val_dataset.dataset.get_gem_profiles(subset_indices, sat_ssh = True)
        gem_temp_errors = (gem_temp.T - original_profiles[:, 0, :]) ** 2
        gem_sal_errors = (gem_sal.T - original_profiles[:, 1, :]) ** 2

        nn_temp_errors = (val_predictions[0][:, :] - original_profiles[:, 0, :]) ** 2
        nn_sal_errors = (val_predictions[1][:, :] - original_profiles[:, 1, :]) ** 2

        gem_temp_rmse = np.sqrt(np.mean(gem_temp_errors, axis = 0))
        gem_sal_rmse = np.sqrt(np.mean(gem_sal_errors, axis = 0))

        nn_temp_rmse = np.sqrt(np.mean(nn_temp_errors, axis = 0))
        nn_sal_rmse = np.sqrt(np.mean(nn_sal_errors, axis = 0))
        
        accuracy_gain_temp = 100*(gem_temp_rmse-nn_temp_rmse)/gem_temp_rmse
        accuracy_gain_sal = 100*(gem_sal_rmse-nn_sal_rmse)/gem_sal_rmse            
            
        # Heatmaps of the RMSE
        lon_bins = np.arange(np.floor(np.min(lon_val)), np.ceil(np.max(lon_val)), bin_size)
        lat_bins = np.arange(np.floor(np.min(lat_val)), np.ceil(np.max(lat_val)), bin_size)

        # Calculate average temperature RMSE for NN and GEM
        avg_temp_rmse_nn, num_prof_nn = calculate_average_rmse_per_bin(lon_bins, lat_bins, lon_val, lat_val, nn_temp_rmse)  # Replace nn_rmse with your RMSE values for NN
        avg_temp_rmse_gem, num_prof_gem = calculate_average_rmse_per_bin(lon_bins, lat_bins, lon_val, lat_val, gem_temp_rmse)  # Replace gem_rmse with your RMSE values for GEM
        avg_temp_rmse_gain = avg_temp_rmse_nn - avg_temp_rmse_gem

        plot_rmse_maps(lon_bins, lat_bins, avg_temp_rmse_nn, avg_temp_rmse_gem, num_prof_nn, "Temperature")
        
        #now let's do the same for salinity
        # Calculate average temperature RMSE for NN and GEM
        avg_sal_rmse_nn, num_prof_nn = calculate_average_rmse_per_bin(lon_bins, lat_bins, lon_val, lat_val, nn_sal_rmse)  # Replace nn_rmse with your RMSE values for NN
        avg_sal_rmse_gem, num_prof_gem = calculate_average_rmse_per_bin(lon_bins, lat_bins, lon_val, lat_val, gem_sal_rmse)  # Replace gem_rmse with your RMSE values for GEM
        avg_sal_rmse_gain = avg_sal_rmse_nn - avg_sal_rmse_gem
        
        # Create an empty array for the difference with the same shape as the ISOP data
        avg_rmse_nn_t = np.full(avg_rmse_isop_t.shape, np.nan)
        avg_rmse_nn_s = np.full(avg_rmse_isop_s.shape, np.nan)

        tolerance = 1e-6  # A small tolerance value for floating-point comparison

        # Iterate over each cell in the ISOP data
        for i, lat in enumerate(lat_centers - 0.5):
            for j, lon in enumerate(lon_centers - 0.5):
                # Find the corresponding cell in the NN data
                nn_lat_idx = np.argmin(np.abs(lat - lat_bins))
                nn_lon_idx = np.argmin(np.abs(lon - lon_bins))

                # Assign values to corresponding cells using a tolerance for comparison
                if np.abs(lat - lat_bins[nn_lat_idx]) < tolerance and np.abs(lon - lon_bins[nn_lon_idx]) < tolerance:
                    if nn_lat_idx < avg_temp_rmse_nn.shape[0] and nn_lon_idx < avg_temp_rmse_nn.shape[1]:
                        avg_rmse_nn_t[i, j] = avg_temp_rmse_nn[nn_lat_idx, nn_lon_idx]
                    if nn_lat_idx < avg_sal_rmse_nn.shape[0] and nn_lon_idx < avg_sal_rmse_nn.shape[1]:
                        avg_rmse_nn_s[i, j] = avg_sal_rmse_nn[nn_lat_idx, nn_lon_idx]             

        plot_rmse_maps(lon_bins, lat_bins, avg_sal_rmse_nn, avg_sal_rmse_gem, num_prof_nn, "Salinity")
        
        plot_comparison_maps(lon_centers, lat_centers, avg_rmse_nn_t, avg_rmse_isop_t, "temperature")
        # or
        plot_comparison_maps(lon_centers, lat_centers, avg_rmse_nn_s, avg_rmse_isop_s, "salinity")
        
    viz()
    
    # Entire dataset:
    print("Now let's see how it performs on the entire dataset")
    val_predictions_bkp = val_predictions
    val_dataset_bkp = val_dataset
    val_predictions_pcs_bkp = val_predictions_pcs
    val_loader_bkp = val_loader
    
    # Get predictions for the entire dataset
    train_dataset_0, val_dataset, test_dataset_0 = split_dataset(full_dataset, 0, 1, 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val_predictions_pcs = get_predictions(trained_model, val_loader, device)
    val_predictions = val_dataset.dataset.inverse_transform(val_predictions_pcs)
    
    viz()
    
    if n_runs > 1:
        #load and aggregate all models
        model_directory = "/unity/g2/jmiranda/SubsurfaceFields/GEM_SubsurfaceFields/saved_models/"
        if not input_params['sat']:
            suffix = "_sat.pth"
            model_files = [f for f in os.listdir(model_directory) if not f.endswith(suffix)]
        else:
            model_files = [f for f in os.listdir(model_directory) if f.endswith(suffix)]
            
        models = []
        for file in model_files:
            model_path = os.path.join(model_directory, file)
            model = torch.load(model_path)
            models.append(model)

        # Collect predictions from all models
        ensemble_temp_val_predictions = np.zeros_like(val_predictions[0])
        ensemble_sal_val_predictions = np.zeros_like(val_predictions[1])
        ct = 0
        for model in models:
            predictions_pcs = get_predictions(model, val_loader, device)
            predictions = val_dataset.dataset.inverse_transform(predictions_pcs)        
            ensemble_temp_val_predictions += predictions[0]
            ensemble_sal_val_predictions += predictions[1]
            ct += 1

        # Averaging the predictions
        ensemble_temp_val_predictions /= ct
        ensemble_sal_val_predictions /= ct

        val_predictions = (ensemble_temp_val_predictions, ensemble_sal_val_predictions)
        
        viz()

    # %%
#     def calculate_sound_speed_NPL(T, S, Z, Phi=45):
#     """
#     Calculate sound speed (in m/s) using the NPL equation.
#     T: Temperature in degrees Celsius
#     S: Salinity in PSU
#     Z: Depth in meters
#     Phi: Latitude in degrees (default 45)
#     """
#     c = (1402.5 + 5 * T - 5.44e-2 * T**2 + 2.1e-4 * T**3 
#          + 1.33 * S - 1.23e-2 * S * T + 8.7e-5 * S * T**2 
#          + 1.56e-2 * Z + 2.55e-7 * Z**2 - 7.3e-12 * Z**3 
#          + 1.2e-6 * Z * (Phi - 45) - 9.5e-13 * T * Z**3 
#          + 3e-7 * T**2 * Z + 1.43e-5 * S * Z)
#     return c

# # Recalculate sound speed at each depth using the NPL equation
# sound_speed_profile_NPL = np.array([calculate_sound_speed_NPL(T, S, z) for T, S, z in zip(temperature_profile, salinity_profile, depths)])

# # Finding the Sonic Layer Depth (SLD) using the NPL equation
# max_sound_speed_index_NPL = np.argmax(sound_speed_profile_NPL)
# SLD_NPL = depths[max_sound_speed_index_NPL]
# # Conversion factor from meters to feet
# meters_to_feet = 3.28084

# # Conversion factor for the gradient from per feet to per 100 meters
# conversion_factor = meters_to_feet / 100

# # Calculating the Below Layer Gradient (BLG) using the NPL equation
# gradient_NPL = np.gradient(sound_speed_profile_NPL, depths_feet)
# # Average gradient below MLD in m/s per 100 feet using the NPL equation
# BLG_NPL = np.mean(gradient_NPL[MLD_index:]) * conversion_factor