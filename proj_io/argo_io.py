# %%
import mat73
from os.path import join
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from copy import deepcopy

def read_data(input_folder):
    return mat73.loadmat(join(input_folder,'ARGO_GoM_20220920.mat')     , use_attrdict=True) 

def read_normalize_data(input_folder, scaler=StandardScaler()):
    data = read_data(input_folder)
    
    # ------ added part: removal of missing values
    n_nans = 10
    
    # Identify indexes with excessive NaNs in TEMP and SAL
    indexes_temp = np.where(np.isnan(data.TEMP).sum(axis=1) >= n_nans)[0]
    indexes_sal = np.where(np.isnan(data.SAL).sum(axis=1) >= n_nans)[0]
    
    # Combine the two sets of indexes
    drop_indexes = np.union1d(indexes_temp, indexes_sal)
    
    # Drop these indexes from the dataset
    for attribute in ['TEMP', 'SAL', 'SH1950', 'TIME', 'LAT', 'LON']:
        attribute_data = getattr(data, attribute)
        
        # Check the shape of the attribute
        if len(attribute_data.shape) == 2:
            # 2D data with profile information
            setattr(data, attribute, np.delete(attribute_data, drop_indexes, axis=1))
        elif len(attribute_data.shape) == 1:
            # 1D data, remove the relevant indexes
            setattr(data, attribute, np.delete(attribute_data, drop_indexes))
    # ------ end of added part
    
    # ------- Temp
    data.TEMP = pd.DataFrame(data.TEMP).fillna(method='ffill').fillna(method='bfill').values
    scaler.fit(data.TEMP)
    data.TEMP = scaler.transform(data.TEMP)
    # ------- Salinity
    data.SAL = pd.DataFrame(data.SAL).fillna(method='ffill').fillna(method='bfill').values
    scaler.fit(data.SAL)
    data.SAL = scaler.transform(data.SAL)
    # ------- SSH
    data.SH1950 = pd.DataFrame(data.SH1950).fillna(method='ffill').fillna(method='bfill').values
    scaler.fit(data.SH1950)
    data.SH1950 = scaler.transform(data.SH1950)
    
    return data, scaler

def revert_normalization(dataset, input_folder, scalers): # denormalization doesn't work!
    """
    This function reverts the normalization applied to the dataset attributes.
    """

    # Get the scalers for TEMP, SAL, and SH1950
    scaler_temp = scalers['TEMP']
    scaler_sal = scalers['SAL']
    scaler_ssh = scalers['SH1950']

    # Inverse transform for each attribute
    dataset.temp_raw = scaler_temp.inverse_transform(dataset.temp)
    dataset.salt_raw = scaler_sal.inverse_transform(dataset.salt)
    dataset.ssh = scaler_ssh.inverse_transform(dataset.ssh)

    # No need to return the dataset as changes are in-place
    return dataset

def revert_normalization_prof(temp_prof, sal_prof, input_folder, scalers):
    """
    This function reverts the normalization applied to individual profiles.
    """

    # Get the scalers for TEMP, SAL, and SH1950
    scaler_temp = scalers['TEMP']
    scaler_sal = scalers['SAL']
    
    temp_prof = scaler_temp.inverse_transform(temp_prof.reshape(-1, 1))
    sal_prof = scaler_sal.inverse_transform(sal_prof.reshape(-1, 1))
    
    return temp_prof, sal_prof
    

def read_normalize_data_test(input_folder, n_nans=10, scaler=StandardScaler()):
    # a. Read data
    data = read_data(input_folder)
    
    # # Convert numpy arrays to DataFrame if needed
    # if isinstance(data.TEMP, np.ndarray):
    #     data.TEMP = pd.DataFrame(data.TEMP)
    # if isinstance(data.SAL, np.ndarray):
    #     data.SAL = pd.DataFrame(data.SAL)

# Identify indexes with excessive NaNs in TEMP and SAL
    indexes_temp = np.where(np.isnan(data.TEMP).sum(axis=1) >= n_nans)[0]
    indexes_sal = np.where(np.isnan(data.SAL).sum(axis=1) >= n_nans)[0]
    
    # Combine the two sets of indexes
    drop_indexes = np.union1d(indexes_temp, indexes_sal)
    
    # Drop these indexes from attributes
    for attribute in ['TEMP', 'SAL', 'SH1950', 'TIME', 'LAT', 'LON']:
        attribute_data = getattr(data, attribute)
        
        # Check the shape of the attribute
        if len(attribute_data.shape) == 2:
            # 2D data with profile information
            setattr(data, attribute, np.delete(attribute_data, drop_indexes, axis=1))
        elif len(attribute_data.shape) == 1:
            # 1D data, remove the relevant indexes
            setattr(data, attribute, np.delete(attribute_data, drop_indexes))
            
    # Calculate mean and std for TEMP, SAL, and SH1950
    data.avg_temp = np.nanmean(data.TEMP, axis=1)
    data.std_temp = np.nanstd(data.TEMP, axis=1)
    data.avg_sal = np.nanmean(data.SAL, axis=1)
    data.std_sal = np.nanstd(data.SAL, axis=1)
    data.avg_ssh = np.nanmean(data.SH1950)
    data.std_ssh = np.nanstd(data.SH1950)
    
    # Normalize TEMP, SAL, and SH1950
    data.TEMP = (data.TEMP - data.avg_temp[:, np.newaxis]) / data.std_temp[:, np.newaxis]
    data.SAL = (data.SAL - data.avg_sal[:, np.newaxis]) / data.std_sal[:, np.newaxis]
    data.SH1950 = (data.SH1950 - data.avg_ssh) / data.std_ssh
    
    # Fill in missing values
    data.TEMP = pd.DataFrame(data.TEMP).fillna(method='ffill').fillna(method='bfill').values
    data.SAL = pd.DataFrame(data.SAL).fillna(method='ffill').fillna(method='bfill').values
    
    # # ------- Temp
    # data.TEMP = pd.DataFrame(data.TEMP).fillna(method='ffill').fillna(method='bfill').values
    # scaler.fit(data.TEMP)
    # data.TEMP = scaler.transform(data.TEMP)
    # # ------- Salinity
    # data.SAL = pd.DataFrame(data.SAL).fillna(method='ffill').fillna(method='bfill').values
    # scaler.fit(data.SAL)
    # data.SAL = scaler.transform(data.SAL)
    # # ------- SSH
    # data.SH1950 = pd.DataFrame(data.SH1950).fillna(method='ffill').fillna(method='bfill').values
    # scaler.fit(data.SH1950)
    # data.SH1950 = scaler.transform(data.SH1950)
    
    # scalers = {}  # Dictionary to hold separate scalers for each data type

    # for attribute in ['TEMP', 'SAL', 'SH1950']:
    #     values = pd.DataFrame(getattr(data, attribute)).fillna(method='ffill').fillna(method='bfill').values
    #     scaler.fit(values)
    #     setattr(data, attribute, scaler.transform(values))
        
    #     scalers[attribute] = deepcopy(scaler)  # Important to deep copy the scaler to save its state for each attribute
    
    # # # Save all scalers in a single pickle file
    # # with open(f'{input_folder}/all_scalers.pkl', 'wb') as f:
    # #     pickle.dump(scalers, f)
    
    return data
    # return data, scalers

def revert_normalization_test(dataset, with_pca):
    
    # # Inverse transform for each attribute
    # dataset.temp_raw = scalers['TEMP'].inverse_transform(dataset.temp)
    # dataset.salt_raw = scalers['SAL'].inverse_transform(dataset.salt)
    # dataset.ssh = scalers['SH1950'].inverse_transform(dataset.ssh)

# Undo normalization for TEMP and SAL
    if with_pca:
        dataset.temp = dataset.invpca_temp * dataset.std_temp[:, np.newaxis] + dataset.avg_temp[:, np.newaxis]
        dataset.salt = dataset.invpca_sal * dataset.std_sal[:, np.newaxis] + dataset.avg_sal[:, np.newaxis]
    else:
        dataset.temp = dataset.temp * dataset.std_temp[:, np.newaxis] + dataset.avg_temp[:, np.newaxis]
        dataset.salt = dataset.salt * dataset.std_sal[:, np.newaxis] + dataset.avg_sal[:, np.newaxis]
    # No need to return the dataset as changes are in-place

# %%
