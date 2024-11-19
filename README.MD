# Causal Weighting of Precipitation Projections over Land of CMIP6 Models

Author: Kevin Debeire, DLR, kevin.debeire@dlr.de 

This repository contains the scripts, data, and instructions necessary to perform causal weighting of precipitation projections over land using CMIP6 models. The approach leverages causal inference and performance/independence-based metrics to assess and weight climate model projections.

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

If there is a mistake in this README or you want to add more details, feel free to update this README.