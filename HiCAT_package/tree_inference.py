import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata


try:
    from bigtree import list_to_tree
except ImportError:  # pragma: no cover - optional visualization dependency

    class _FallbackTree:
        def __init__(self, tree_list):
            self.tree_list = tree_list

        def hshow(self):
            for item in self.tree_list:
                print(item)

    def list_to_tree(tree_list):
        return _FallbackTree(tree_list)


def identify_region_pairs(rank_matrix):
    region_pairs_dic = {}
    regions_num = rank_matrix.shape[0]
    region_names = rank_matrix.columns.tolist()
    for i in range(regions_num - 1):
        for j in range(i + 1, regions_num):
            region_pairs_dic[rank_matrix.iloc[i, j]] = [region_names[i], region_names[j]]
    region_pairs_dic = dict(sorted(region_pairs_dic.items()))
    return region_pairs_dic


def pairs_to_nodes(region_pairs_dic, region_names):
    node_dic = {}
    hier_dic = {}
    adj_dic = {}
    for i in range(len(region_names)):
        node_dic["node" + str(i)] = [region_names[i]]
    hier_dic[0] = list(node_dic.keys())
    node_num = len(region_names) - 1
    for rank, pairs in region_pairs_dic.items():
        node0 = [key for key, value in node_dic.items() if pairs[0] in value][-1]
        node0_hier = [hier for hier, node_list in hier_dic.items() if node0 in node_list][0]
        node1 = [key for key, value in node_dic.items() if pairs[1] in value][-1]
        node1_hier = [hier for hier, node_list in hier_dic.items() if node1 in node_list][0]
        if node0 != node1:
            node_num = node_num + 1
            new_node = "node" + str(node_num)
            new_node_hier = max([node0_hier, node1_hier]) + 1
            new_node_regions = node_dic[node0] + node_dic[node1]
            node_dic[new_node] = new_node_regions
            if new_node_hier not in list(hier_dic.keys()):
                hier_dic[new_node_hier] = []
            hier_dic[new_node_hier].append(new_node)
            adj_dic[new_node] = [node0, node1]
        if len(new_node_regions) == len(region_names):
            break
    return node_dic, hier_dic, adj_dic


def traverse_node_path(adj_dic, start_node, end_node, path=None):
    if path is None:
        path = []
    path.append(start_node)
    if start_node == end_node:
        return path
    for root_node, leaf_nodes in adj_dic.items():
        if start_node in leaf_nodes:
            if root_node not in path:
                path = traverse_node_path(adj_dic, root_node, end_node, path)
                if path is not None:
                    return path
    path.pop()
    return None


def node_to_tree(node_dic, adj_dic, region_names, show=True):
    path_dic = {}
    tree_list = []
    end_node = [key for key, value in node_dic.items() if len(value) == len(region_names)][0]
    for region in region_names:
        start_node = [key for key, value in node_dic.items() if value == [region]][0]
        path = traverse_node_path(adj_dic, start_node, end_node)
        path_dic[start_node] = path
        node_tree_path = region
        for i in range(1, len(path)):
            node_tree_path = path[i] + "/" + node_tree_path
        tree_list.append(node_tree_path)
    if show is True:
        tree_str = list_to_tree(tree_list)
        tree_str.hshow()
    return path_dic, tree_list


def features_distance(input_adata, features_set, label_col="label", scale=True):
    adata_sub = input_adata[:, input_adata.var.index.isin(features_set)].copy()
    tissue_regions = adata_sub.obs[label_col].value_counts().index.tolist()
    tissue_regions = [i for i in tissue_regions if i not in ["nan", "unknown"]]
    features_mean_df = pd.DataFrame(
        np.zeros((len(tissue_regions), len(features_set))),
        index=tissue_regions,
        columns=features_set,
    )
    for region in tissue_regions:
        for feature in features_set:
            tmp = adata_sub[adata_sub.obs[label_col] == region, adata_sub.var.index == feature].copy()
            features_mean_df.loc[region, feature] = np.mean(tmp.X, axis=0)
    print("========== Features Mean within each Tissue Region ==========")
    print(features_mean_df.round(2))
    dists = squareform(pdist(features_mean_df, metric="euclidean"))
    dists_df = pd.DataFrame(dists, index=tissue_regions, columns=tissue_regions)
    print("========== Pairwise Distances between Tissue Regions ==========")
    print(dists_df.round(2))
    if scale is False:
        return dists_df
    scaled_dists_df = pd.DataFrame(np.zeros(dists.shape), index=tissue_regions, columns=tissue_regions)
    upper_diag_elements = dists[np.triu_indices(len(dists), k=1)]
    dists_min = upper_diag_elements.min()
    dists_max = upper_diag_elements.max()
    for i in range(len(tissue_regions)):
        for j in range(len(tissue_regions)):
            if i != j:
                scaled_dists_df.iloc[i, j] = (dists_df.iloc[i, j] - dists_min) / (dists_max - dists_min)
    print("========== Scaled Pairwise Distances between Tissue_regions ==========")
    print(scaled_dists_df.round(2))
    return dists_df, scaled_dists_df


def spatial_distance(spatial_df, label_col, x_col="x", y_col="y", neighbors=10, scale=True):
    spa_loc = spatial_df[[x_col, y_col]].to_numpy()
    spa_adata = sc.AnnData(spa_loc, dtype=spa_loc.dtype)
    spa_adata.obs[label_col] = spatial_df[label_col].to_numpy()
    spa_dists = squareform(pdist(spa_adata.X, metric="euclidean"))
    upper_dists = spa_dists[np.triu_indices(len(spa_dists), k=1)]
    spots_num = spatial_df.shape[0]
    qtl = (neighbors * spots_num) / (spots_num**2 - spots_num)
    radius = np.quantile(upper_dists, q=qtl)
    spa_mask = (spa_dists <= radius) * 1
    ave_nbr = np.sum(spa_mask) / len(spa_mask)
    print("The average number of neighboring spots: " + str(ave_nbr.round(1)) + ".")
    tissue_regions = spa_adata.obs[label_col].value_counts().index.tolist()
    tissue_regions = [i for i in tissue_regions if i not in ["nan", "unknown"]]
    regions_num = len(tissue_regions)
    nbr_df = pd.DataFrame(np.zeros((regions_num, regions_num)), index=tissue_regions, columns=tissue_regions)
    for i in range(regions_num):
        name = tissue_regions[i]
        region_index = spa_adata[spa_adata.obs[label_col] == name].obs.index.tolist()
        region_index = [int(k) for k in region_index]
        tmp = spa_mask[region_index, :]
        col_indices = np.nonzero(tmp)[1]
        uni_indices = list(set(col_indices))
        nbr_counts = spa_adata[uni_indices].obs[label_col].value_counts()
        nbr_regions = nbr_counts.index.tolist()
        nbr_regions = [n for n in nbr_regions if n not in ["nan", "unknown"]]
        common_regions = [n for n in tissue_regions if n in nbr_regions]
        print("==========" + name + "==========")
        print(nbr_counts)
        for region in common_regions:
            nbr_df.loc[name, region] = nbr_counts[region]
    print("==========Spatial Neighborhood Components==========")
    print(nbr_df.astype(int))
    if (nbr_df == 0).any().any():
        nbr_upd = nbr_df + 1
    else:
        nbr_upd = nbr_df
    nbr_pct = pd.DataFrame(np.zeros((regions_num, regions_num)), index=tissue_regions, columns=tissue_regions)
    row_sums = nbr_upd.sum(axis=1)
    for i in range(regions_num):
        rowsum = row_sums.iloc[i] - nbr_upd.iloc[i, i]
        for j in range(regions_num):
            if i == j:
                nbr_pct.iloc[i, j] = 1
            else:
                nbr_pct.iloc[i, j] = nbr_upd.iloc[i, j] / rowsum
    print("==========The percentage of Neighborhood Components==========")
    print(nbr_pct.round(2))
    dists = pd.DataFrame(np.zeros((regions_num, regions_num)), index=tissue_regions, columns=tissue_regions)
    for i in range(regions_num):
        for j in range(regions_num):
            dists.iloc[i, j] = np.log(1 / (nbr_pct.iloc[i, j] * nbr_pct.iloc[j, i]))
    print("==========Transformed Pairwise Spatial Distances==========")
    print(dists.round(2))
    if scale is False:
        return dists
    scaled_dists = pd.DataFrame(np.zeros((regions_num, regions_num)), index=tissue_regions, columns=tissue_regions)
    dists_np = dists.to_numpy()
    upper_elements = dists_np[np.triu_indices(len(dists_np), k=1)]
    dists_min = upper_elements.min()
    dists_max = upper_elements.max()
    for i in range(regions_num):
        for j in range(regions_num):
            if i != j:
                scaled_dists.iloc[i, j] = (dists.iloc[i, j] - dists_min) / (dists_max - dists_min)
    print("==========Scaled Pairwise Spatial Distances==========")
    print(scaled_dists.round(2))
    return dists, scaled_dists


def multi_modal_distance(
    input_adata,
    features_dic,
    w_G=1,
    w_I=0,
    w_S=0.5,
    neighbor_radius=None,
    shape="hexagon",
    x_col="x",
    y_col="y",
    label_col="label",
    scale=True,
):
    scaled_dists_list = []
    weights = [w_G, w_I, w_S]
    if w_G > 0:
        gene_list = features_dic["gene"]
        print("---------------------------------------------------------------------------------")
        print("================================ Gene Expression ================================")
        print("---------------------------------------------------------------------------------")
        dists_gene, scaled_dists_gene = features_distance(input_adata, gene_list, label_col, scale)
        scaled_dists_list.append(scaled_dists_gene)
    if w_I > 0:
        image_list = features_dic["image"]
        print("---------------------------------------------------------------------------------")
        print("================================ Image Features ================================")
        print("---------------------------------------------------------------------------------")
        dists_img, scaled_dists_img = features_distance(input_adata, image_list, label_col, scale)
        scaled_dists_list.append(scaled_dists_img)
    if w_S > 0:
        if neighbor_radius is None:
            if shape == "hexagon":
                neighbor_radius = 6
            elif shape == "square":
                neighbor_radius = 4
            else:
                print("Shape not recognized, shape='hexagon' for Visium data and 'square' for ST data.")
        spatial_df = input_adata.obs[[x_col, y_col, label_col]].copy()
        print("---------------------------------------------------------------------------------------")
        print("================================ Spatial Neighborhoods ================================")
        print("---------------------------------------------------------------------------------------")
        dists_spa, scaled_dists_spa = spatial_distance(spatial_df, label_col, x_col, y_col, neighbor_radius, scale)
        scaled_dists_list.append(scaled_dists_spa)
    weights = [w for w in weights if w != 0]
    scaled_dists_all = sum(w * dists for w, dists in zip(weights, scaled_dists_list))
    print("================================ Rank based on overall distance ================================")
    scaled_dists_ranks = rank_dists(scaled_dists_all)
    return scaled_dists_all, scaled_dists_ranks


def rank_dists(scaled_dists):
    tissue_regions = scaled_dists.columns.tolist()
    scaled_dists_array = np.array(scaled_dists)
    upper_diag_elements = scaled_dists_array[np.triu_indices(len(scaled_dists_array), k=1)]
    ranks = rankdata(upper_diag_elements, method="average").astype(int)
    scaled_dists_ranks = pd.DataFrame(np.zeros(scaled_dists.shape), index=tissue_regions, columns=tissue_regions, dtype=int)
    c = 0
    for i in range(len(tissue_regions) - 1):
        for j in range(i + 1, len(tissue_regions)):
            scaled_dists_ranks.iloc[i, j] = ranks[c]
            scaled_dists_ranks.iloc[j, i] = ranks[c]
            c = c + 1
    print("========== The Ranking of Pairwise Distances ==========")
    print(scaled_dists_ranks)
    return scaled_dists_ranks


def tree_str_gene_selection(
    input_adata,
    gene_num,
    min_fold_change,
    min_in_out_group_ratio,
    min_in_group_fraction,
    pvals_adj=0.05,
    label_col="label",
):
    region_genes_dic = {}
    gene_list = []
    tissue_regions = input_adata.obs[label_col].value_counts().index.tolist()
    tissue_regions = [i for i in tissue_regions if i not in ["nan", "unknown"]]
    print(f"Included tissue regions: {tissue_regions}")
    for region in tissue_regions:
        df1 = rank_genes_groups(
            input_adata=input_adata,
            target=region,
            label_col=label_col,
            non_target="rest",
            two_sides=False,
            logged=True,
        )
        df1_filtered = df1[
            (df1["pvals_adj"] <= pvals_adj)
            & (df1["in_out_group_ratio"] >= min_in_out_group_ratio)
            & (df1["in_group_fraction"] >= min_in_group_fraction)
            & (df1["fold_change"] >= min_fold_change)
        ]
        df1_filtered = df1_filtered.sort_values(by="fold_change", ascending=False)
        filtered_ngene = df1_filtered.shape[0]
        print("After applying filtering parameters, " + region + " -> " + str(filtered_ngene) + " genes left")
        region_genes = df1_filtered["genes"].tolist()[0 : np.min([gene_num, filtered_ngene])]
        region_genes_dic[region] = region_genes
    for _, region_genes in region_genes_dic.items():
        gene_list = list(set(gene_list) | set(region_genes))
    return region_genes_dic, gene_list


def ensure_output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_adata(path):
    return sc.read(str(path))


def write_adata(adata, path):
    path = Path(path)
    ensure_output_dir(path.parent)
    adata.write_h5ad(str(path))
    return path


def save_tree_results(
    output_dir,
    region_pairs_dic=None,
    node_dic=None,
    hier_dic=None,
    adj_dic=None,
    path_dic=None,
    scaled_dists_all=None,
    scaled_dists_ranks=None,
    prefix="tree_inference",
):
    output_dir = ensure_output_dir(output_dir)
    payload = {
        "region_pairs_dic": region_pairs_dic,
        "node_dic": node_dic,
        "hier_dic": hier_dic,
        "adj_dic": adj_dic,
        "path_dic": path_dic,
    }
    if scaled_dists_all is not None:
        payload["scaled_dists_all"] = scaled_dists_all.to_dict()
    if scaled_dists_ranks is not None:
        payload["scaled_dists_ranks"] = scaled_dists_ranks.to_dict()
    with open(output_dir / f"{prefix}.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return output_dir / f"{prefix}.json"

