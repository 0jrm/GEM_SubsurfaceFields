# %%
import sys
# sys.path.append("./")
sys.path.append("../")

from proj_io.argo_io import read_normalize_data
from configs.RunConfig import RunConfig
from sklearn.decomposition import PCA
from proj_viz.argo_viz import plot_single_ts_profile

import matplotlib.pyplot as plt

# %% Testint the use of PCA

data = read_normalize_data(RunConfig.data_folder.value)
# Testing the use of PCA to reduce the dimensionality of the input data
temp = data.SAL
print(f'Original shape: {temp.shape}')
# Use sklearn.decomposition.PCA to reduce the dimensionality temp data
# %%
index = 320
components = 100
pca = PCA(n_components=components)
pca.fit(temp)
temp_pca = pca.transform(temp)
print(f'PCA shape: {temp_pca.shape}')
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_.sum():0.6f}")

recovered = pca.inverse_transform(temp_pca) 
plot_single_ts_profile(temp[:,index], recovered[:,index])
# %%
