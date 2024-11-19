#use ARGPARSE....py script to send srun PCMCI on DKRZ on different nodes.
import glob
import subprocess

model_path = '/work/bd1083/b309165/CMIP6_CME/output_pca/CMIP6_era5_1979-2014_weights_timebin1/*.bin'
model_name = sorted(glob.glob(model_path))
N_model = len(model_name)

N_model_per_job = 8
idx_start_model = 0

counter = idx_start_model

while counter < N_model:
    upper_bound = min(counter + N_model_per_job,N_model)
    print("SENDING JOB FOR MODELS # %d up to # %d" %(counter,upper_bound))
    CMD = "sbatch --job-name={}_{} --account=bd1083 --time=08:00:00 --partition=compute ARGPARSE_run_pcmci_for_CMIP6_MODELS_from_era5_1979-2014_weights_60selectedcomps_timebin1x2_taumax20.py {} {} > ./output/pcmci_models_{}to{}.txt ".format(counter,upper_bound,counter,upper_bound,counter,upper_bound)
    subprocess.call([CMD],shell=True,stdin=None, stdout=None, stderr=None, close_fds=True)
    counter = upper_bound

