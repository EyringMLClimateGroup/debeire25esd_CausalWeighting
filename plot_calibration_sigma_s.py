# AUTHOR: Kevin Debeire
# License: GNU General Public License v3.0
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','nature'])

def extract_model_name(full_name):
    """
    Extract model name from model_ensemble.

    Parameters:
    - full_name (str): Full model name including ensemble information.

    Returns:
    - str: Model name extracted from the full name.
    """
    return full_name.split('_')[0]

def calculate_cross_model_distances(distance_df):
    """
    Calculate cross-model distances for all unique models in the DataFrame.

    Parameters:
    - distance_df (pd.DataFrame): DataFrame containing distances between model members.

    Returns:
    - list: List of arrays containing cross-model distances for each unique model.
    - list: List of unique model names.
    """
    unique_models = distance_df['model_ensemble'].apply(extract_model_name).unique()
    cross_model_distances = []
    
    for model in unique_models:
        # Calculate distances between current model and all other models
        for ref_model in unique_models:
            if ref_model != model:
                # Calculate distances between model and reference model
                distances = distance_df.loc[(distance_df['model_ensemble'].apply(extract_model_name) == model) & 
                                            (distance_df['model_ensemble_reference'].apply(extract_model_name) == ref_model), 'dcme'].values

                distances = distances[(distances != 0)]
                cross_model_distances.extend(distances)
    
    return cross_model_distances, unique_models

def calculate_inter_model_distances(distance_df):
    """
    Calculate inter-model distances for all unique models in the DataFrame.

    Parameters:
    - distance_df (pd.DataFrame): DataFrame containing distances between model members.

    Returns:
    - list: List of arrays containing inter-model distances for each unique model.
    - list: List of unique model names.
    """
    unique_models = distance_df['model_ensemble'].apply(extract_model_name).unique()
    inter_model_distances = []
    
    for model in unique_models:
        # Calculate distances between current model and all other models
        model_distances = []
        for ref_model in unique_models:
            if ref_model == model:
                # Calculate distances between model and reference model
                distances = distance_df.loc[(distance_df['model_ensemble'].apply(extract_model_name) == model) & 
                                            (distance_df['model_ensemble_reference'].apply(extract_model_name) == ref_model), 'dcme'].values

                distances = distances[(distances != 0)]
                model_distances.extend(distances)
        
        inter_model_distances.append(model_distances)
    
    return inter_model_distances, unique_models

# Calculate inter-model distances
xr_data = xr.open_dataset("./perf_ind_scores/xr_ind_era5_60comps_pcmci_parcorr_timebin1_ssp585.nc")
df = (xr_data/ xr_data.median()).dcme.to_dataframe()
df = df.reset_index()
inter_model_distances, model_names = calculate_inter_model_distances(df)
cross_distances, _ = calculate_cross_model_distances(df)
inter_model_distances.append(cross_distances)
model_names = list(model_names)
model_names.append("Intermodel only")

# Plotting boxplot
import pylab
params = {'legend.fontsize': 'medium',
        #'figure.figsize': (15, 5),
        'axes.labelsize': 'x-large',
        'axes.titlesize':'xx-large',
        'xtick.labelsize':'x-large',
        'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
plt.figure(figsize=(10, 7))

plt.boxplot(inter_model_distances, labels=model_names, vert=False)
plt.xlabel(r'Distance ($1-F_1$ score)')
plt.title('Distances between ensembles of the same model and intermodel distances (ERA5 reference)')
plt.grid(True, which='major', color='darkgrey', linestyle='--', linewidth=0.7, axis="x")
plt.axvline(x=0.9, color='orange', linestyle='--')
plt.tight_layout()
plt.savefig("calibration_sigma_s_ERA5.pdf",dpi=300)
# plt.show()
plt.close()

# Calculate inter-model distances
xr_data = xr.open_dataset("./perf_ind_scores/xr_ind_60comps_pcmci_parcorr_timebin1_ssp585.nc")
df = (xr_data/ xr_data.median()).dcme.to_dataframe()
df = df.reset_index()
inter_model_distances, model_names = calculate_inter_model_distances(df)
cross_distances, _ = calculate_cross_model_distances(df)
inter_model_distances.append(cross_distances)
model_names = list(model_names)
model_names.append("Intermodel only")
# print(cross_distances)
# Plotting boxplot
pylab.rcParams.update(params)
plt.figure(figsize=(10, 7))

plt.boxplot(inter_model_distances, labels=model_names, vert=False)
plt.xlabel(r'Distance ($1-F_1$ score)')
plt.title('Distances between ensembles of the same model and intermodel distances (NCEP/NCAR reference)')
plt.grid(True, which='major', color='darkgrey', linestyle='--', linewidth=0.7, axis="x")
plt.axvline(x=0.9, color='orange', linestyle='--')
plt.tight_layout()
plt.savefig("calibration_sigma_s.pdf",dpi=300)
# plt.show()
plt.close()
#pdfjam calibration_sigma_s.pdf calibration_sigma_s_ERA5.pdf --nup 2x1 --outfile calibration_sigma_combined.pdf --papersize '{13cm,4.5cm}' --scale 1.0 --noautoscale false
