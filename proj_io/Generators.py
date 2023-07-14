# %%
import torch
import os
import mat73
from os.path import join
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

## ------- Custom dataset ------
class ProjDataset(Dataset):
    def __init__(self, input_folder, transform=None):

        data = mat73.loadmat( join(input_folder,'ARGO_GoM_20220920.mat')     , use_attrdict=True) 
        self.input_folder = input_folder
        self.total_examples = data.ADT_loc.shape[0]
        self.transform = transform

        # Normalizing data
        scaler = StandardScaler()
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

        self.temp = data.TEMP.astype(np.float32)
        self.salt = data.SAL.astype(np.float32)
        self.ssh = data.SH1950.astype(np.float32)
        self.time = data.TIME.astype(np.float32)
        self.lats = data.LAT.astype(np.float32)
        self.lons = data.LON.astype(np.float32)

            # TODO here we need to save the scaler object to be used in the test set.
            # ******************* Saving Normalization params, scaler object **********************
            # scaler.path_file = file_name
            # with open(scaler.path_file, 'wb') as f: #scaler.path_file must be defined during training.
            #     pickle.dump(scaler, f)
            # print(f'Scaler/normalizer object saved to: {scaler.path_file}')
            # print(F'Done! Current shape: {data_norm_df.shape} ')
            # return data_norm_df

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):
        # Example of returning just SSH as input and T and S as output
        input =  np.array([self.ssh[idx].squeeze(), self.temp[0,idx]])
        target = np.concatenate((self.temp[:,idx], self.salt[:,idx]))
        # print(f'Getting item {idx} from dataset, input shape: {input.shape}, target shape: {target.shape}')
        return input, target
# %% ----- Test DataLoader --------
if __name__ == "__main__":
    # ----------- Skynet ------------
    import sys
    sys.path.append("./")
    sys.path.append("../")
    from configs.RunConfig import RunConfig
    from proj_viz.argo_viz import plot_single_ts_profile
    input_folder = RunConfig.data_folder.value
    
    dataset = ProjDataset(input_folder=input_folder)
    myloader = DataLoader(dataset, batch_size=20, shuffle=False)
    # --------- Just reading some lats lons ----------
    for i, batch in enumerate(myloader):
        # Print the size of the batch, size of input and output
        print(f'Batch size: {batch[0].shape}, Input size: {batch[0].shape}, Output size: {batch[1].shape}')
        ssh, ts = batch
        plot_single_ts_profile(ts[0][0:2001],ts[0][2001:4002], title=f'Batch {i}')
        break