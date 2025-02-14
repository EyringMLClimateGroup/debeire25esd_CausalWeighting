# Causal Weighting of Precipitation Projections over Land of CMIP6 Models

This repository contains the scripts, data, and instructions necessary to perform causal weighting of precipitation projections over land using CMIP6 models. The approach leverages causal inference and performance/independence-based metrics to assess and weight climate model projections:

> Debeire, K., Bock, L., Nowack, P., Runge, J. & Eyring, V. (2025). Constraining uncertainty in projected precipitation over land with causal discovery  (see [article](https://egusphere.copernicus.org/preprints/2024/egusphere-2024-2656/)). Earth System Dynamics.

Author of the repository: Kevin Debeire, DLR, kevin.debeire(at)dlr.de

The current release on Zenodo can be found here: [![DOI](https://zenodo.org/badge/932154073.svg)](https://zenodo.org/badge/latestdoi/932154073)

## Overview

This project is separated in three steps:
1. To implement causal network analysis using SLP data of CMIP6 models, PCA-Varimax and the PCMCI algorithm.
2. To compute **performance** and **independence scores** of CMIP6 models (stored in the `perf_ind_scores` folder).
3. To use these scores for producing weighted projections across different SSP scenarios.

To reproduce the figures for the precipitation over land, **we recommend to directly use the pre-computed performance and indepedence metrics** and run the provided ESMValTool recipes.

## Repository Contents

- `perf_ind_scores/`: Contains the performance and independence scores as normalized complements of F1-scores.
- `.py scripts`: For calculating PCA-Varimax components and estimating causal networks using PCMCI.
- Jupyter Notebooks:
  - For calculating F1-scores.
  - For saving the values in `.nc` files for further use.
- ESMValTool recipe: Produces boxplots, maps, and time series of weighted projections for three different SSP scenarios.

## Requirements

### Environment Setup

1. **Conda Environment**  
   Use the provided `environment_pca.yml` file to set up the Python environment for running the `.py` scripts and Jupyter notebooks:
   ```bash
   mamba env create -n <your_env_name> -f environment_pca.yml
   conda activate <your_env_name>
   ```

2. **ESMValTool Installation**  
   Follow the [ESMValTool Quickstart Installation Guide](https://docs.esmvaltool.org/en/latest/quickstart/installation.html) to install ESMValTool.  
   Ensure that you use the `causal_weighting` branch of ESMValTool:

   ```bash
   git clone -b causal_weighting https://github.com/ESMValGroup/ESMValTool.git
   cd ESMValTool
   mamba env update --file environment.yml
   pip install --editable '.[develop]'
   ```

## Running ESMValTool Recipes

If you want to produce plots for the weighted precipitation projections: you can use the pre-computed performance and independence metrics directly. To run the ESMValTool recipes included in this repository, follow these steps:

1. Ensure that your environment with ESMValTool is activated and configured with access to the necessary CMIP6 model data (more information [here](https://docs.esmvaltool.org/en/latest/quickstart/configuration.html)).
    ```bash
    conda activate <your_env_name>
    ```
2. Navigate to the directory containing the recipe file (e.g., `recipe_weighting_pcmci_ssp245_60comps_final.yml`) and run the following:

   ```bash
   esmvaltool run recipe_weighting_pcmci_ssp245_60comps_final.yml
   ```

### Notes

If you want to produce PCA-Varimax weights, run `generate_varimax_era5_1979-2014_allseason_timebin1.py`. You can then apply those weights to desired CMIP6 models `generate_varimax_for_ALL_CMIP6_MODELS_from_era5_1979-2014_weights_timebin1.py` to create the dimension-reduced data. Then, run `run_pcmci_for_CMIP6_MODELS_timebin1x2_taumax20.py` to estimate the causal networks (install tigramite for this). The Jupyter notebook can be run to produce some plots and calculate the metrics from the calculated causal networks.

## License
GNU General Public License v3.0

See [license.txt](./license.txt) for full text.
