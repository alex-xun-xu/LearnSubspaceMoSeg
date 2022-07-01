import numpy as np
import Tool

def Kmeans_SOD(X,Nc_range):
## Infer number of clusters with second order difference
#
#   X - N*D feature embedding
#   Nc_range - the range of number of clusters

    sumdist_all = []

    for Nc_i in range(0,len(Nc_range)):

        Nc = Nc_range[Nc_i]
        pred, sumdist = Tool.Kmeans_Inference_mdlsel(X,Nc)
        sumdist_all.append(sumdist)

    log_res = np.log(sumdist_all)

    sod = log_res[2::] + log_res[0:-2] - 2*log_res[1:-1]

    best_Nc_idx = np.argmax(sod)
    best_Nc = Nc_range[best_Nc_idx]+1
    pred, _ = Tool.Kmeans_Inference_mdlsel(X,best_Nc)

    return pred, best_Nc


def Kmeans_SOD_scale(X,Nc_range):
## Infer number of clusters with second order difference
#
#   X - N*D feature embedding
#   Nc_range - the range of number of clusters

    sumdist_all = []

    for Nc_i in range(0,len(Nc_range)):

        Nc = Nc_range[Nc_i]
        Npts = X.shape[0]
        pred, sumdist = Tool.Kmeans_Inference_mdlsel(X,Nc)
        sumdist_all.append(sumdist/Npts)

    sumdist_all = np.array(sumdist_all)
    log_res = np.log(sumdist_all**5)

    sod = log_res[2::] + log_res[0:-2] - 2*log_res[1:-1]

    best_Nc_idx = np.argmax(sod)
    best_Nc = Nc_range[best_Nc_idx]+1
    pred, _ = Tool.Kmeans_Inference_mdlsel(X,best_Nc)

    return pred, best_Nc


def Kmeans_RelDrop(X,Nc_range,tau=0.2):
## Infer number of clusters with second order difference
#
#   X - N*D feature embedding
#   Nc_range - the range of number of clusters

    sumdist_all = []

    for Nc_i in range(0,len(Nc_range)):

        Nc = Nc_range[Nc_i]
        Npts = X.shape[0]

        pred, sumdist = Tool.Kmeans_Inference_mdlsel(X,Nc)
        sumdist_all.append(sumdist/Npts)

    log_res = np.log(sumdist_all)

    # sod = log_res[2::] + log_res[0:-2] - 2*log_res[1:-1]

    rel_drop = (log_res[0:-1] - log_res[1::])/log_res[0:-1]

    best_Nc_idx = np.sum(rel_drop>tau)
    best_Nc = Nc_range[best_Nc_idx+1]
    pred, _ = Tool.Kmeans_Inference_mdlsel(X,best_Nc)

    return pred, best_Nc