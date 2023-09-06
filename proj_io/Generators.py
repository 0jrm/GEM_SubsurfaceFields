# %%
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
import numpy as np

from proj_io.argo_io import read_normalize_data_test, read_normalize_data # Needs to be commented for testing 
# from proj_io.argo_io import read_normalize_data  # Needs to be commented for testing 

## ------- Custom dataset ------

class ProjDataset(Dataset):
    def __init__(self, input_folder, transform=None):

        data = read_normalize_data_test(input_folder)
        # data, self.scaler = read_normalize_data_test(input_folder)
        # data, scaler = read_normalize_data(input_folder)
        self.input_folder = input_folder
        self.total_examples = data.SH1950.shape[0]
        self.transform = transform

        self.temp = data.TEMP.astype(np.float32)
        self.salt = data.SAL.astype(np.float32)
        self.avg_temp = data.avg_temp.astype(np.float32)
        self.std_temp = data.std_temp.astype(np.float32)
        self.avg_sal = data.avg_sal.astype(np.float32)
        self.std_sal = data.std_sal.astype(np.float32)
        self.ssh = data.SH1950.astype(np.float32)
        self.time = np.cos(data.TIME.astype(np.float32)%365)
        # self.time = data.TIME.astype(np.float32)
        self.lats = data.LAT.astype(np.float32)
        self.lons = data.LON.astype(np.float32)
        self.inoutdims = [3, len(np.concatenate((self.temp[:,0], self.salt[:,0])))] # Two for SSH and SST, one for T and S
        # self.inoutdims = [2, len(np.concatenate((self.temp[:,0], self.salt[:,0])))] # Two for SSH and SST, one for T and S
        
    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):
        # Example of returning just SSH as input and T and S as output
        # input =  np.array([self.ssh[idx].squeeze(), self.temp[0,idx]])
        input =  np.array([self.ssh[idx].squeeze(), self.temp[0,idx], self.time[idx].squeeze()])
        target = np.concatenate((self.temp[:,idx], self.salt[:,idx]))
        # print(f'Getting item {idx} from dataset, input shape: {input.shape}, target shape: {target.shape}')
        return input, target

    def get_inout_dims(self):
        return self.inoutdims
    
    def get_original_profile(self, idx):
        '''
        Returns the original profile for a given index
        '''
        return self.temp[:,idx], self.salt[:,idx]


class ProjDatasetPCA(Dataset):
    def __init__(self, input_folder, temp_components, sal_components, transform=None):

        self.scaler = {}
        data = read_normalize_data(input_folder)
        # data = read_normalize_data_test(input_folder)
        self.input_folder = input_folder
        self.total_examples = data.SH1950.shape[0]
        self.transform = transform

        self.temp = data.TEMP.astype(np.float32)
        self.salt = data.SAL.astype(np.float32)
        self.avg_temp = data.avg_temp.astype(np.float32)
        self.std_temp = data.std_temp.astype(np.float32)
        self.avg_sal = data.avg_sal.astype(np.float32)
        self.std_sal = data.std_sal.astype(np.float32)
        self.ssh =  data.SH1950.astype(np.float32)
        self.time = data.TIME.astype(np.float32)
        self.lats = data.LAT.astype(np.float32)
        self.lons = data.LON.astype(np.float32)

        # Perform PCA on the temperature
        pca_temp = PCA(n_components=temp_components)
        pca_temp.fit(self.temp.T)
        self.temp = pca_temp.transform(self.temp.T).T
        print(f'Temp PCA shape: {self.temp.shape}, original shape: {self.temp.shape}')
        print(f"Explained Variance Ratio: {pca_temp.explained_variance_ratio_.sum():0.6f}")
        self.pca_temp = pca_temp

        # Perform PCA on the salinity
        pca_sal = PCA(n_components=sal_components)
        pca_sal.fit(self.salt.T)
        self.salt = pca_sal.transform(self.salt.T).T
        print(f'Sal PCA shape: {self.salt.shape}, original shape: {self.salt.shape}')
        print(f"Explained Variance Ratio: {pca_sal.explained_variance_ratio_.sum():0.6f}")
        self.pca_sal = pca_sal

        self.inoutdims = [2, len(np.concatenate((self.temp[:,0], self.salt[:,0])))] # Two for SSH and SST, one for T and S
        
    def __len__(self):
        return self.total_examples

    def get_inout_dims(self):
        return self.inoutdims

    def __getitem__(self, idx):
        # Example of returning just SSH as input and T and S as output
        input =  np.array([self.ssh[idx].squeeze(), self.temp[0,idx]])
        target = np.concatenate((self.temp[:,idx], self.salt[:,idx]))
        # print(f'Getting item {idx} from dataset, input shape: {input.shape}, target shape: {target.shape}')
        return input, target

    def get_original_profile(self, idx):
        '''
        Returns the original profile for a given index
        '''
        return self.temp[:,idx], self.salt[:,idx]

    def inverse_pca(self, temp, sal):
        # Inverse the PCA transformation
        return self.pca_temp.inverse_transform(temp), self.pca_sal.inverse_transform(sal) 


# %% ----- Test DataLoader --------
if __name__ == "__main__":
    # ----------- Skynet ------------
    import sys
    # sys.path.append("./")
    sys.path.append("../")
    from configs.RunConfig import RunConfig
    from proj_viz.argo_viz import compare_profiles
    from argo_io import read_normalize_data
    input_folder = RunConfig.data_folder.value
    
    use_pca = True
    if use_pca:
        temp_components = 100
        sal_components = 100
        dataset = ProjDatasetPCA(input_folder=input_folder, temp_components=temp_components, sal_components=sal_components)
    else:
        dataset = ProjDataset(input_folder=input_folder)

    # Print the inout dimensions for each example
    inoutdims = dataset.get_inout_dims()
    print(f"In size: {inoutdims[0]}, Out size: {inoutdims[1]}")
    batch_size = 10
    myloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # --------- Just reading some lats lons ----------
    for i, batch in enumerate(myloader):
        # Print the size of the batch, size of input and output
        print(f'Batch size: {batch_size}, Input size: {batch[0].shape}, Output size: {batch[1].shape}')
        ssh, ts = batch
        for j in range(batch_size):
            if use_pca:
                temp_profile_original, sal_profile_original  = dataset.get_original_profile(i*batch_size+j)
                temp_profile, sal_profile  = dataset.inverse_pca(ts[j][0:temp_components], ts[j][temp_components:])
                compare_profiles(temp_profile, temp_profile_original, title=f'Batch {i} element {j}', 
                                    labelone="Temp PCA", labeltwo="Temp Original", figsize=5)

                compare_profiles(sal_profile, sal_profile_original, title=f'Batch {i} element {j}', 
                                    labelone="Salinity PCA", labeltwo="Salinity Original", figsize=5)
            else:
                temp_profile, sal_profile = ts[j][0:int(inoutdims[1]/2)], ts[j][int(inoutdims[1]/2):]
                compare_profiles(temp_profile,sal_profile, title=f'Batch {i} element {j}', 
                                 labelone="Temperature", labeltwo="Salinity", figsize=5)
        break
# %%
