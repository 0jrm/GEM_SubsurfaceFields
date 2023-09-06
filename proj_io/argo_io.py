# %%
import mat73
from os.path import join
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


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

    # TODO here we need to save the scaler object to be used in the test set.
    # ******************* Saving Normalization params, scaler object **********************
    # scaler.path_file = file_name
    # with open(scaler.path_file, 'wb') as f: #scaler.path_file must be defined during training.
    #     pickle.dump(scaler, f)
    # print(f'Scaler/normalizer object saved to: {scaler.path_file}')
    # print(F'Done! Current shape: {data_norm_df.shape} ')
    # return data_norm_df

    return data
