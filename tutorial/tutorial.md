<h1><center>HiCAT Tutorial</center></h1>

<center>Author: Jing Huang, Xueqi Shen, Yoland Smith, Lara Harik, Linghua Wang, Jindan Yu, Michael P. Epstein* and Jian Hu*

### Outline
1. [Installation](#1-installation)
2. [Import modules](#2-import-python-modules)
3. [Read in data](#3-read-in-data)
4. [Data preprocessing](#4-data-preprocessing)
5. [Tree inference](#5-tree-inference)
6. [Heterogeneity inference](#6-heterogeneity-inference)
7. [Reference selection](#7-reference-selection)
8. [Label Transfer](#8-label-transfer)

### 1. Installation
To install HiCAT package you must make sure that your python version is over 3.5. If you don’t know the version of python you can check it by:


```python
import platform
platform.python_version()
```

<br>
Now you can install the current release of HiCAT by the following three ways:

#### 1.1 PyPI: Directly install the package from PyPI


```python
pip3 install HiCAT
# Note: you need to make sure that the pip is for python3

# or we could install HiCAT by
python3 -m pip install HiCAT

# If you do not have permission (when you get a permission denied error), you should install HiCAT by
pip3 install --user HiCAT
```

#### 1.2 Github
Download the package from Github and install it locally:


```python
git clone https://github.com/jinghuang-stats/HiCAT
cd HiCAT/HiCAT_package/
python3 setup.py install --user
```

#### 1.3 Anaconda
If you do not have Python3.5 or Python3.6 installed, consider installing Anaconda (see Installing Anaconda). After installing Anaconda, you can create a new environment, for example, HiCAT_env (or any name that you like).


```python
# create an environment called HiCAT_env
conda create -n HiCAT_env python=3.11.5 -y

# activate your environment 
conda activate HiCAT_env

git clone https://github.com/jinghuang-stats/HiCAT
cd HiCAT/HiCAT_package/
python3 setup.py build
python3 setup.py install
conda deactivate
```

### 2. Import python modules


```python
import os, csv, time, pickle
import warnings
warnings.filterwarnings('ignore')
import statistics
import hnswlib
import math
import cv2
import pandas as pd
import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import anndata as ad
import TESLA as tesla
import HiCAT as hicat
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from anndata import AnnData
from scipy.sparse import issparse
from scipy.stats import rankdata
from scipy.spatial.distance import cdist, pdist, squareform

```


```python
hicat.__version__
```

    '1.0.0'


### 3. Read in data
Data notes:
- Toydata are made available at the [shared folder](https://drive.google.com/drive/folders/1BaqScSe3mxz7JGlixYd-4SSmzHZBOoVb?usp=share_link).
- We also provided precomputed reference information available in the [results folder](https://drive.google.com/drive/folders/1dvCbkgciSRbCc7SAMad0dBVa2tEQtMI-?usp=sharing). Users can directly use these files to perform label transfer for breast cancer tissue type, without having to construct their own reference datasets or generate reference information from scratch. 
  The precomputed reference information includes:
  - node_dic: A dictionary mapping each tree node to its corresponding tissue regions;
  - adj_dic: A dictionary describing parent-child relationships in the hierarchical tree;
  - hier_genes: Reference-specific hierarchical marker genes to guide each split.
- Intermediate results are saved in the [figures folder](https://github.com/jinghuang-stats/HiCAT/blob/main/tutorial/figures).
<br>


The current version of HiCAT requires two main input data: 
1. Reference gene expression matrix: ref_expression_matrix.h5ad
This file should be an AnnData object with dimensions (N $\times$ G), where (N) is the nuber of spots/cells and (G) is the number of genes. The expression matrix is stored in .X, while .obs contains pathologist-generated annotations. Gene-level information is stored in .var, and additional unstructured information can be stored in .uns. 
2. Query gene expression matrix: qry_expression_matrix.h5ad
This file should also be an AnnData object with dimensions (N $\times$ G), containing the gene expression matrix for the query dataset to be annotated. 

Optional input:
3. Query image feature matrix: qry_image_matrix.h5ad
This optional file contains morphology-derived image features for the query data, with dimensions (N $\times$ I), where I is the number of image features.

```python
# Set the working directory
plot_dir="." # set a data directory (need to make up folders of results and figures)
if not os.path.exists(plot_dir+"/figures"):
	os.mkdir(plot_dir+"/figures")


if not os.path.exists(plot_dir+"/results"):
	os.mkdir(plot_dir+"/results")


```


```python
data_path = "./toy_data"
dataset_name = "HER2+BC"
label_key = "label"
sample_key = "sample"
low_exp_thres = 0.02

# Read in reference adata
ref_section_list = ["H1", "G2", "E1"]

ref_results = construct_ref_adata_dic(
    ref_section_list=ref_section_list,
    data_path=data_path,
    dataset_name=dataset_name,
    file_template="sudo_{dataset_name}_{section}_log_s=1_res=50_nbr=10_k=2.h5ad",
    label_key=label_key,
    sample_key=sample_key,
    low_exp_thres=low_exp_thres,
)

ref_adata_dic = ref_results["ref_adata_dic_raw"]
ref_adata_sca_dic = ref_results["ref_adata_dic_filtered"]
all_adata_sca = ref_results["all_adata"]
ref_common_genes = ref_results["common_genes"]

```

```python
# Read in qry adata
qry_section = "H2"

qry_file = os.path.join(
    data_path,
    f"sudo_{dataset_name}_{qry_section}_log_s=1_res=50_nbr=10_k=2+hipt.h5ad"
)

qry_adata = sc.read_h5ad(qry_file)

```

### 4. Data preprocessing
```python
label_color_dict = {
    "cancer_in_situ": [254, 128, 22],
    "connective_tissue": [63, 72, 203],
    "adipose_tissue": [113, 227, 223],
    "invasive_cancer": [236, 28, 36],
    "breast_glands": [15, 209, 69],
    "immune_infiltrate": [255, 242, 4],
}

output_dir = "./figures"
os.makedirs(output_dir, exist_ok=True)

# Extract scribble masks for reference sections
for ref_section in ref_section_list:
    image_path = os.path.join(data_path, f"{ref_section}.jpg")
    annotated_image_path = os.path.join(data_path, f"{ref_section}_annotated.jpg")

    ref_mask, label_id_mask, d_mask = extract_scribble_masks(
        image_path=image_path,
        annotated_image_path=annotated_image_path,
        label_color_dict=label_color_dict,
        output_dir=output_dir,
    )

# Split query data into gene and image features
qry_features = qry_adata.var.index.tolist()

qry_gene_features = [f for f in qry_features if "hipt" not in f]
qry_img_features = [f for f in qry_features if "hipt" in f]

qry_gene_adata = qry_adata[:, qry_adata.var.index.isin(qry_gene_features)].copy()
qry_img_adata = qry_adata[:, qry_adata.var.index.isin(qry_img_features)].copy()

print("Query gene features:", qry_gene_adata.n_vars)
print("Query image features:", qry_img_adata.n_vars)

```

### 5. Tree inference
#### 5.1 Integrate multi-modal distances
```python
gene_filtering_paras = {
    "min_fold_change": 1.1,
    "min_in_out_group_ratio": 1,
    "min_in_group_fraction": 0,
    "pvals_adj": 0.05,
    "gene_num": 10,
}

image_filtering_paras = {
    "min_fold_change": 1.1,
    "min_in_out_group_ratio": 1,
    "min_in_group_fraction": 0,
    "pvals_adj": 0.05,
    "gene_num": 5,
}

features_dic = {}

for sample in ref_section_list:
    sample_adata = ref_adata_dic[sample]

    all_features = sample_adata.var.index.tolist()

    gene_features = [f for f in all_features if "hipt" not in f]
    image_features = [f for f in all_features if "hipt" in f]

    gene_adata = sample_adata[:, sample_adata.var.index.isin(gene_features)].copy()
    image_adata = sample_adata[:, sample_adata.var.index.isin(image_features)].copy()

    region_genes_dic, gene_list = tree_str_gene_selection(
        input_adata=gene_adata,
        gene_num=gene_filtering_paras["gene_num"],
        min_fold_change=gene_filtering_paras["min_fold_change"],
        min_in_out_group_ratio=gene_filtering_paras["min_in_out_group_ratio"],
        min_in_group_fraction=gene_filtering_paras["min_in_group_fraction"],
        pvals_adj=gene_filtering_paras["pvals_adj"],
        label_key=label_key,
    )

    if image_adata.n_vars > 0:
        region_image_dic, image_list = tree_str_gene_selection(
            input_adata=image_adata,
            gene_num=image_filtering_paras["gene_num"],
            min_fold_change=image_filtering_paras["min_fold_change"],
            min_in_out_group_ratio=image_filtering_paras["min_in_out_group_ratio"],
            min_in_group_fraction=image_filtering_paras["min_in_group_fraction"],
            pvals_adj=image_filtering_paras["pvals_adj"],
            label_key=label_key,
        )
    else:
        image_list = []

    features_dic[sample] = {
        "gene": gene_list,
        "image": image_list,
    }

print(features_dic)

```

#### 5.2 Infer hierarchical tree structure
```python
weights = {
    "w_G": 1,
    "w_I": 1,
    "w_S": 1,
}

integrated_dists, integrated_ranks, sample_dists_dic = multi_sample_distance(
    adata_dic=ref_adata_dic,
    features_dic=features_dic,
    w_G=weights["w_G"],
    w_I=weights["w_I"],
    w_S=weights["w_S"],
    neighbors=None,
    shape="hexagon",
    x_key="x",
    y_key="y",
    label_key=label_key,
    scale=True,
    return_sample_dists=True,
)

tree = build_hier_tree(rank_matrix=integrated_ranks, show=True)
split_df = make_split_table(tree)

save_tree_inference_results(
    config_dir=output_dir,
    tree=tree,
    integrated_dists=integrated_dists,
    integrated_ranks=integrated_ranks,
    sample_dists_dic=sample_dists_dic,
    split_df=split_df,
)

```

### 6. Heterogeneity inference
```python
tissue_region_list = [
    "adipose_tissue",
    "connective_tissue",
    "breast_glands",
    "immune_infiltrate",
    "cancer_in_situ",
    "invasive_cancer",
]

results_bc = infer_heterogeneity_scores(
    ref_adata_dic=ref_adata_dic,
    all_adata=all_adata_sca,
    tissue_region_list=tissue_region_list,
    label_key=label_key,
    sample_key=sample_key,
    low_exp_thres=low_exp_thres,
    gene_num=gene_filtering_paras["gene_num"],
    min_fold_change=gene_filtering_paras["min_fold_change"],
    min_in_out_group_ratio=gene_filtering_paras["min_in_out_group_ratio"],
    min_in_group_fraction=gene_filtering_paras["min_in_group_fraction"],
    pvals_adj=gene_filtering_paras["pvals_adj"],
)

bc_hetero_summary = results_bc["hetero_summary"]

```

### 7. Reference selection
```python
qry_adata_dic = {
    qry_section: qry_gene_adata,
}

sort_by = "similarity"

results_dic = select_references_pipeline(
    ref_adata_dic=ref_adata_sca_dic,
    qry_adata_dic=qry_adata_dic,
    sort_by=sort_by,
)

selected_refs_dic = results_dic["selected_refs_dic"]
print(selected_refs_dic)
print(selected_refs_dic)

```

### 8. Label transfer




