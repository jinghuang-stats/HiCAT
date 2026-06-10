import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler


def filter_low_exp_genes(input_adata, low_exp_thres):
    gene_exp = pd.DataFrame(
        input_adata.X.copy(),
        columns=input_adata.var.index.tolist(),
        index=input_adata.obs.index.tolist(),
    )
    gene_exp = gene_exp.astype(bool)
    nonzero_exp_frac = gene_exp.sum() / gene_exp.shape[0]
    filtered_genes = nonzero_exp_frac[nonzero_exp_frac >= low_exp_thres].index.tolist()
    return filtered_genes


def normalize_adata(input_adata, method="min_max"):
    scaler_adata = MinMaxScaler()
    scaler_adata.fit(input_adata.X)
    X_scaled = scaler_adata.transform(input_adata.X)
    input_adata_sca = sc.AnnData(X_scaled)
    input_adata_sca.obs = input_adata.obs.copy()
    input_adata_sca.var = input_adata.var.copy()
    return input_adata_sca


def make_nonnegative_adata(input_adata):
    X_nonneg = input_adata.X - input_adata.X.min(axis=0)
    output_adata = sc.AnnData(X_nonneg)
    output_adata.obs = input_adata.obs.copy()
    output_adata.var = input_adata.var.copy()
    return output_adata




