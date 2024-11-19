import numpy as np
import glob
from matplotlib import pyplot as plt
## use `%matplotlib notebook` for interactive figures
import pickle
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
import pandas as pd
import scienceplots
plt.style.use(['science','nature'])
import pylab
params = {'legend.fontsize': 'x-large',
    #'figure.figsize': (15, 5),
    'axes.labelsize': 'xx-large',
    'axes.titlesize':'xx-large',
    'xtick.labelsize':'xx-large',
    'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)
run_preprocessing = False
run_PCMCI = False
run_PCA = False
save_res= True
make_dic= False
use_CMIP6_data=True
plot_figure = True
time_bin=1

if use_CMIP6_data:
    #define all paths to CMIP6 data folders
    

    #all psl paths
    psl_1979_2014_path="/work/bd1083/b309165/CMIP6_CME/data/CMIP6_data/detrend_data/"
    pca_res_path="/work/bd1083/b309165/CMIP6_CME/output_pca/CMIP6_ncar_1979-2014_weights_timebin1/"
    #pcmci_res_path="/work/bd1083/b309165/CMIP6_CME/output_pcmci/CMIP6_ncar_1979-2014_weights_and_pr_mean_timebin1x2_alllinks/"
    pcmci_res_path="/work/bd1083/b309165/CMIP6_CME/output_pcmci/CMIP6_ncar_1979-2014_weights_timebin1x2_selectedcomps/"
    #pr paths
    pr_histo_path = "/work/bd1083/b309165/CMIP6_CME/data/CMIP6_data/pr_data/histo_1979-2014_pr_study_timebinned_masked/"
    pr_scenario_path = "/work/bd1083/b309165/CMIP6_CME/data/CMIP6_data/pr_data/%s_data/"
    #piControl_path= "/work/bd1083/b309165/CMIP6_CME/data/CMIP6_data/pr_data/piControl_data/"
    pr_1979_2014_path= "/work/bd1083/b309165/CMIP6_CME/data/CMIP6_data/pr_data/histo_1979-2014_data/"
    #csv/bin output files
    global_res_path = "/work/bd1083/b309165/CMIP6_CME/results/results_CMIP6/global_link_and_parents_1979-2014_weights_psl_timebin1x2.bin"
    delta_precip_file="/work/bd1083/b309165/CMIP6_CME/results/results_CMIP6/delta_precip_histo_1860-1910_vs_%s_2050-2100_final_version.csv"
    sscore_file="/work/bd1083/b309165/CMIP6_CME/results/results_CMIP6/seasonnal_S_score_model_versus_%s.csv"

make_dic= False

save_global_res= True #only used if make_dic=True
selected_comps_indices=[i for i in range(0,50)]
var_names=["X_"+str(i) for i in range(0,50)]

alpha_list = [0.001,0.0001,0.00001] #[0.01,0.001,0.0001,0.00001]
if make_dic:
    link_mat_alpha_dic = {}
    link_mat_dic= {}
    val_mat_dic={}
    val_mat_alpha_dic={}
    p_mat_dic={}
    p_mat_alpha_dic={}
    q_mat_dic={}
    q_mat_alpha_dic={}
    ak_dic={}
    ak_alpha_dic={}
    for alpha_level in alpha_list:
        for res_file in glob.glob(pcmci_res_path+"/results_*.bin"):

            res = pickle.load(open(res_file,"rb"))
            results= res["results"]
            file_name = res['file_name']
            info_model= file_name.split("_")
            dataset_name = info_model[2]
            ensemble=""
            if dataset_name != "ncar":
                dataset_name= info_model[2]
                if use_CMIP6_data:
                    ensemble= info_model[5]
                else : ensemble= info_model[7]
            if dataset_name == "GISS-E2-R":
                ensemble= info_model[5]
            season= info_model[-1][7:-4]

            print("Current model, ensemble, season : "+dataset_name+" "+ensemble+" "+season )
            file_path = pca_res_path+"/"+ file_name
            datadict = pickle.load(open(file_path, 'rb'))
            d = datadict['results']
            time_mask = d['time_mask']
            fulldata = d['ts_unmasked']
            N = 50
            fulldata_mask = np.repeat(time_mask.reshape(len(d['time']), 1), N, axis=1)
            fulldata = fulldata[:, 0:N]
            fulldata_mask = fulldata_mask[:, 0:N]
            dataframe = pp.DataFrame(fulldata, mask=fulldata_mask)
            
            CI_params = {       'significance':'analytic', 
                                'mask_type':['y'],
                                'recycle_residuals':False,
                                }
            cond_ind_test = ParCorr(**CI_params)
            pcmci=PCMCI(cond_ind_test=cond_ind_test,dataframe=dataframe)
            tau_max=10

            val_mat_dic.setdefault(season,{})
            val_mat_dic[season].setdefault(dataset_name,{})
            val_mat_dic[season][dataset_name].setdefault(ensemble,None)
            val_mat_dic[season][dataset_name][ensemble]= results['val_matrix']
            p_mat_dic.setdefault(season,{})
            p_mat_dic[season].setdefault(dataset_name,{})
            p_mat_dic[season][dataset_name].setdefault(ensemble,)
            p_mat_dic[season][dataset_name][ensemble]= results['p_matrix']


        p_mat_alpha_dic[alpha_level]= p_mat_dic
        val_mat_alpha_dic[alpha_level]= val_mat_dic

    
    global_res = {"p_val":p_mat_alpha_dic,"val_mat":val_mat_alpha_dic}

    if save_global_res :
        print("Writing global results file: "+global_res_path)
        with open(global_res_path, 'wb') as file:
            pickle.dump(global_res, file)

else:
    with open(global_res_path, 'rb') as file:
        global_res = pickle.load(file)

#Function to compute WDM from p_matrices and val_matrices
def get_metric_f1(ref_p_matrix, p_matrix, ref_val_matrix, val_matrix, alpha, 
            tau_min=0, tau_diff=1, same_sign=True):

    N, N, taumaxp1 = val_matrix.shape
    TP = 0
    FP = 0
    FN = 0
    auto = 0
    count = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                for tau in range(tau_min, taumaxp1):
#                     print(np.sum(ref_p_matrix[i,j,tau] < alpha),np.sum(p_matrix[i,j,tau] < alpha))
                    if ref_p_matrix[i,j,tau] > alpha and p_matrix[i,j,tau] < alpha:
                        FP += 1
                    elif ref_p_matrix[i,j,tau] < alpha and np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha):
                        count +=1
                        if same_sign==True and np.sign(ref_val_matrix[i,j,tau]) == np.sign(val_matrix[i,j,tau]):
                            TP += 1
                            # if tau > 10:
                            #     print("TP for lag %d found" %tau)
                        elif same_sign==True and np.sign(ref_val_matrix[i,j,tau]) != np.sign(val_matrix[i,j,tau]):
                            FN += 1
                        elif same_sign==False:
                            TP += 1
                    elif ref_p_matrix[i,j,tau] < alpha and not(np.any(p_matrix[i,j,max(0,tau-tau_diff):tau+tau_diff+1] < alpha)):
                        FN += 1
            else:
                auto +=1
    precision =  float(TP+1e-10) / float(TP + FP +1e-10)
    recall = float(TP+1e-10) / float(TP + FN +1e-10)
    f1 = 2.0*precision*recall/float(precision + recall)
    return precision, recall, TP, FP, FN, f1, auto, count

####This cell creates df_f1score pandasframe which stores WDM for all season,model,ensemble triplets
alpha = 0.0001
ref_ds="ncar"
p_mat_dic=global_res["p_val"][alpha]
val_mat_dic= global_res["val_mat"][alpha]
#ak_mat_dic = global_res["AK_matrix"][alpha]
score_list =[]
for season in p_mat_dic:
    if season !="global":#drop global
        for dataset in p_mat_dic[season]:
            for ensemble in p_mat_dic[season][dataset]:
                if dataset!=ref_ds:
                    ref_p_matrix= p_mat_dic[season][ref_ds][""]
                    p_matrix= p_mat_dic[season][dataset][ensemble]
                    ref_val_matrix= val_mat_dic[season][ref_ds][""]
                    val_matrix= val_mat_dic[season][dataset][ensemble]
                    #print(kept_nodes)
                    precision, recall, TP, FP, FN, score, auto, count = get_metric_f1(ref_p_matrix[:50,:50,:], p_matrix[:50,:50,:], ref_val_matrix[:50,:50,:], val_matrix[:50,:50,:],
                    alpha, tau_min=1, tau_diff=2, same_sign=True) #tau_diff=3
                    score_list.append([season,dataset,ensemble,score])

season,dataset,ensemble,score= [list(a) for a in zip(*score_list)]
df_f1score_global = pd.DataFrame({"season":season,"model":dataset,"ensemble":ensemble,"F1-score":score})
#if not use_CMIP6_data : df_f1score.loc[df_f1score['model'] == "GISS-E2-R","ensemble"]="r1i1p1" #fix for GISS ensemble being wrong
#get average F1-score over seasons
df_f1score_seasonaveraged_global = df_f1score_global.groupby(["model","ensemble"])["F1-score"].mean().rename("F1-score",inplace=True).to_frame()
df_f1score_seasonaveraged_global

to_keep = ["FGOALS-g3_r1i1p1f1","FGOALS-g3_r3i1p1f1","KACE-1-0-G_r3i1p1f1","MIROC-ES2L_r10i1p1f2","MIROC-ES2L_r1i1p1f2",
"MIROC-ES2L_r2i1p1f2","MIROC-ES2L_r3i1p1f2","MIROC-ES2L_r4i1p1f2","MIROC-ES2L_r5i1p1f2","MIROC-ES2L_r6i1p1f2",
"MIROC-ES2L_r7i1p1f2","MIROC-ES2L_r8i1p1f2","MIROC-ES2L_r9i1p1f2","MIROC6_r10i1p1f1","MIROC6_r1i1p1f1","MIROC6_r2i1p1f1",
"MIROC6_r3i1p1f1","MIROC6_r4i1p1f1","MIROC6_r5i1p1f1","MIROC6_r6i1p1f1","MIROC6_r7i1p1f1","MIROC6_r8i1p1f1","MIROC6_r9i1p1f1",
"MPI-ESM1-2-HR_r1i1p1f1","MPI-ESM1-2-HR_r2i1p1f1","MPI-ESM1-2-LR_r10i1p1f1","MPI-ESM1-2-LR_r1i1p1f1","MPI-ESM1-2-LR_r2i1p1f1",
"MPI-ESM1-2-LR_r3i1p1f1","MPI-ESM1-2-LR_r4i1p1f1","MPI-ESM1-2-LR_r5i1p1f1","MPI-ESM1-2-LR_r6i1p1f1","MPI-ESM1-2-LR_r7i1p1f1",
"MPI-ESM1-2-LR_r8i1p1f1","MPI-ESM1-2-LR_r9i1p1f1","MRI-ESM2-0_r1i1p1f1","MRI-ESM2-0_r1i2p1f1","MRI-ESM2-0_r2i1p1f1",
"MRI-ESM2-0_r3i1p1f1","MRI-ESM2-0_r4i1p1f1","MRI-ESM2-0_r5i1p1f1","NorESM2-LM_r1i1p1f1","UKESM1-0-LL_r1i1p1f2","UKESM1-0-LL_r2i1p1f2",
"UKESM1-0-LL_r3i1p1f2","UKESM1-0-LL_r4i1p1f2","KACE-1-0-G_r2i1p1f1","KACE-1-0-G_r1i1p1f1","ACCESS-CM2_r1i1p1f1","ACCESS-CM2_r2i1p1f1",
"ACCESS-CM2_r3i1p1f1","ACCESS-ESM1-5_r1i1p1f1","ACCESS-ESM1-5_r2i1p1f1","ACCESS-ESM1-5_r3i1p1f1","BCC-CSM2-MR_r1i1p1f1",
"CESM2_r1i1p1f1","CESM2-WACCM_r1i1p1f1","CESM2-WACCM_r2i1p1f1","CESM2-WACCM_r3i1p1f1","CNRM-CM6-1_r1i1p1f2","CNRM-CM6-1_r2i1p1f2",
"CNRM-CM6-1_r3i1p1f2","CNRM-CM6-1_r4i1p1f2","CNRM-CM6-1_r5i1p1f2","CNRM-CM6-1_r6i1p1f2","CNRM-ESM2-1_r1i1p1f2",
"CNRM-ESM2-1_r2i1p1f2","CNRM-ESM2-1_r3i1p1f2","CNRM-ESM2-1_r4i1p1f2","CNRM-ESM2-1_r5i1p1f2","CanESM5_r10i1p1f1",
"CanESM5_r1i1p1f1","CanESM5_r1i1p2f1","CanESM5_r2i1p1f1","CanESM5_r2i1p2f1","CanESM5_r3i1p1f1",
"CanESM5_r3i1p2f1","CanESM5_r4i1p1f1","CanESM5_r4i1p2f1","CanESM5_r5i1p1f1","CanESM5_r6i1p1f1",
"CanESM5_r7i1p1f1","CanESM5_r8i1p1f1","CanESM5_r9i1p1f1","EC-Earth3_r11i1p1f1",
"EC-Earth3_r1i1p1f1","EC-Earth3_r4i1p1f1","EC-Earth3-Veg_r1i1p1f1","EC-Earth3-Veg_r2i1p1f1",
"EC-Earth3-Veg_r3i1p1f1","EC-Earth3-Veg_r4i1p1f1","HadGEM3-GC31-LL_r1i1p1f3","HadGEM3-GC31-LL_r2i1p1f3",
"HadGEM3-GC31-LL_r3i1p1f3","HadGEM3-GC31-LL_r4i1p1f3","HadGEM3-GC31-MM_r1i1p1f3",
"HadGEM3-GC31-MM_r2i1p1f3","HadGEM3-GC31-MM_r3i1p1f3","HadGEM3-GC31-MM_r4i1p1f3",
"INM-CM5-0_r1i1p1f1","IPSL-CM6A-LR_r1i1p1f1","IPSL-CM6A-LR_r2i1p1f1","IPSL-CM6A-LR_r3i1p1f1","IPSL-CM6A-LR_r4i1p1f1","IPSL-CM6A-LR_r6i1p1f1"]


##plot
import seaborn as sn
import pandas as pd
import xarray as xr

def barplot(metric: 'xr.DataArray', df_rank: pd.DataFrame, filename: str, title_modif: str, rank_ref):
    """Visualize metric as barplot."""
    name = metric.name
    variable_group = metric.variable_group
    units = metric.units

    metric_df = metric.to_dataframe().reset_index()
    metric_df = metric_df.merge(df_rank, on='model', how='left')  # Merge with rank DataFrame
    ylabel = f' {variable_group}'

    figure, axes = plt.subplots(figsize=(10, 7))
    chart = sn.barplot(x='model',
                       y=name,
                       data=metric_df,
                       ax=axes,
                       color="blue")
    chart.set_xticklabels(chart.get_xticklabels(),
                          rotation=45,
                          horizontalalignment='right')
    if variable_group == 'weight':
        chart.set_title('Performance weights')
    else:
        chart.set_title(f'{variable_group} against NCEP/NCAR networks'+title_modif)
    chart.set_ylabel(ylabel)
    chart.set_xlabel('')
    chart.grid(True, which='major', color='darkgrey', linestyle='--', linewidth=0.7, axis="y")
    ymin, ymax = chart.get_ylim()
    chart.set_ylim([0.5, 0.72])

    # Add rank annotations on top of each bar
    for index, row in metric_df.iterrows():
        chart.text(index, row[name] + 0.003, f'{int(row["Rank"])}', color='black', ha="center",fontsize=15)
    from scipy.stats import spearmanr
    # Calculate the Spearman rank correlation coefficient
    corr, p_val = spearmanr(ranks_df, rank_ref)
    plt.figtext(0.15,0.825,r'$\rho =$' + str(round(corr,3))+', p-value = '+ "{:.1e}".format(p_val),color='red',fontsize=15)
    print(f'{corr}, {p_val}')
    figure.savefig(filename, dpi=300, bbox_inches='tight')
    print("Saved figure in %s" %filename)
    plt.show()
    plt.close(figure)
#####load REFERENCE RANK for 60 comps and 10-5 alpha_mci
ranks_df_ref = pd.read_csv("performance_60comps_ranks.csv")

# Rank the models for the single metric in each dataframe
to_keep_models = set(model.split('_')[0] for model in to_keep)
# Convert to a list for easier use
to_keep_models = list(to_keep_models)
df_plot_perf = df_f1score_global.groupby(["model"],as_index=False)["F1-score"].mean()
df_plot_perf = df_plot_perf[df_plot_perf.isin(to_keep_models).any(axis=1)]

xr_plot_perf = xr.DataArray(data=df_plot_perf["F1-score"].values,dims=["model"],
                            coords=dict(model=("model",df_plot_perf.model.values)),
                            attrs= {"variable_group":"F1-score","units":""},name="F1-score")

df_plot_perf.set_index('model', inplace=True)
ranks_df = df_plot_perf.rank(axis=0, ascending=False, method='min')
ranks_df.rename(columns = {'F1-score':'Rank'}, inplace = True)

barplot(xr_plot_perf , ranks_df, "performance_50comps_alpha0_0001.pdf",r' (50 components, $\alpha_{MCI}=10^{-4}$)',ranks_df_ref)

alpha = 0.00001
ref_ds="ncar"
p_mat_dic=global_res["p_val"][alpha]
val_mat_dic= global_res["val_mat"][alpha]
#ak_mat_dic = global_res["AK_matrix"][alpha]
score_list =[]
for season in p_mat_dic:
    if season !="global":#drop global
        for dataset in p_mat_dic[season]:
            for ensemble in p_mat_dic[season][dataset]:
                if dataset!=ref_ds:
                    ref_p_matrix= p_mat_dic[season][ref_ds][""]
                    p_matrix= p_mat_dic[season][dataset][ensemble]
                    ref_val_matrix= val_mat_dic[season][ref_ds][""]
                    val_matrix= val_mat_dic[season][dataset][ensemble]
                    #print(kept_nodes)
                    precision, recall, TP, FP, FN, score, auto, count = get_metric_f1(ref_p_matrix[:50,:50,:], p_matrix[:50,:50,:], ref_val_matrix[:50,:50,:], val_matrix[:50,:50,:],
                    alpha, tau_min=1, tau_diff=2, same_sign=True) #tau_diff=3
                    score_list.append([season,dataset,ensemble,score])

season,dataset,ensemble,score= [list(a) for a in zip(*score_list)]
df_f1score_global = pd.DataFrame({"season":season,"model":dataset,"ensemble":ensemble,"F1-score":score})
#if not use_CMIP6_data : df_f1score.loc[df_f1score['model'] == "GISS-E2-R","ensemble"]="r1i1p1" #fix for GISS ensemble being wrong
#get average F1-score over seasons
df_f1score_seasonaveraged_global = df_f1score_global.groupby(["model","ensemble"])["F1-score"].mean().rename("F1-score",inplace=True).to_frame()

df_plot_perf = df_f1score_global.groupby(["model"],as_index=False)["F1-score"].mean()
df_plot_perf = df_plot_perf[df_plot_perf.isin(to_keep_models).any(axis=1)]

xr_plot_perf = xr.DataArray(data=df_plot_perf["F1-score"].values,dims=["model"],
                            coords=dict(model=("model",df_plot_perf.model.values)),
                            attrs= {"variable_group":"F1-score","units":""},name="F1-score")

df_plot_perf.set_index('model', inplace=True)
ranks_df = df_plot_perf.rank(axis=0, ascending=False, method='min')
ranks_df.rename(columns = {'F1-score':'Rank'}, inplace = True)
barplot(xr_plot_perf , ranks_df, "performance_50comps_alpha0_00001.pdf",r' (50 components, $\alpha_{MCI}=10^{-5}$)',ranks_df_ref)
# pdfjam performance_50comps_alpha0_00001.pdf performance_60comps_alpha0_0001.pdf --nup 2x1 --outfile performance_60comps_combined.pdf --papersize '{20cm,7cm}' --scale 1.0 --noautoscale false