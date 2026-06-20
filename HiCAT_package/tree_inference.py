from __future__ import annotations

import json
import pickle
import pandas as pd
import numpy as np
import scanpy as sc
from scipy import sparse
from bigtree import list_to_tree
from scipy.stats import rankdata
from scipy.spatial.distance import cdist, pdist, squareform

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Local package imports
from .util import get_region_genes, rank_genes_groups

# I. some supplementary utility functions
# import rank_genes_group function
# import get_region_genes function

# II. feature select and distance measurements
# logic: select region-specific features
# feature distance, spatial distance
# multi-modal distance
# multi-sample distance: shared regions vs. unique regions

# III. infer tree structure
# hierarchical tree -> present tree as .png or .txt structure

# IV. 
# use the structure to guide gene selection for both range-based and nn-based framework

# V.
# demo: testing data and expected results | can be further imported into tutorial note



@dataclass
class HierTree:
    node_dic: Dict[str, List[str]]
    hier_dic: Dict[int, List[str]]
    adj_dic: Dict[str, List[str]]
    path_dic: Dict[str, List[str]]
    tree_list: List[str]
    region_names: List[str]
    root_node: str

    def show(self) -> None:
        """Display the hierarchical tree structure using bigtree"""
        try:
            from bigtree import list_to_tree
            tree_str = list_to_tree(self.tree_list)
            tree_str.hshow()
        except ImportError:
            print("bigtree is not installed. Showing tree paths instead:")
            for path in self.tree_list:
                print(path)

    def get_children(self, node: str) -> List[str]:
        """Return the two child nodes of an internal node."""
        if node not in self.adj_dic:
            raise ValueError(f"{node} is a leaf node and has no children.")
        return self.adj_dic[node]

    def get_regions(self, node: str) -> List[str]:
        """Return tissue regions contained in a node."""
        if node not in self.node_dic:
            raise ValueError(f"{node} is not found in the tree.")
        return self.node_dic[node]

    def is_leaf(self, node: str) -> bool:
        """Check whether a node is a leaf node."""
        return node not in self.adj_dic

    def get_leaf_nodes(self) -> List[str]:
        """Return all leaf nodes."""
        return [node for node in self.node_dic if node not in self.adj_dic]

    def get_internal_nodes(self) -> List[str]:
        """Return all internal nodes."""
        return list(self.adj_dic.keys())

    def get_split_pairs(self, order: str = "root_to_leaf") -> List[Tuple[str, str, str]]:
        """
        Return binary split information.

        Returns
        -------
        split_pairs:
            List of tuples:
            (parent_node, child_node_1, child_node_2)

        This is useful for hierarchical gene selection and hierarchical label transfer.
        """
        split_pairs = [
            (parent, children[0], children[1])
            for parent, children in self.adj_dic.items()
        ]

        if order == "root_to_leaf":
            split_pairs = sorted(
                split_pairs,
                key=lambda x: len(self.node_dic[x[0]]),
                reverse=True
            )
        elif order == "leaf_to_root":
            split_pairs = sorted(
                split_pairs,
                key=lambda x: len(self.node_dic[x[0]])
            )
        else:
            raise ValueError("order must be either 'root_to_leaf' or 'leaf_to_root'.")

        return split_pairs

    def to_text(self) -> str:
        """Create a readable text representation of the tree."""
        lines = []
        lines.append(f"Root node: {self.root_node}")
        lines.append("")
        lines.append("Node dictionary:")
        for node, regions in self.node_dic.items():
            lines.append(f"{node}: {regions}")

        lines.append("")
        lines.append("Adjacency dictionary:")
        for parent, children in self.adj_dic.items():
            lines.append(f"{parent}: {children}")

        lines.append("")
        lines.append("Tree paths:")
        for path in self.tree_list:
            lines.append(path)

        return "\n".join(lines)

    def save_txt(self, output_path: str | Path) -> None:
        """Save tree structure as a .txt file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(self.to_text())

    def save_png(self, output_path: str | Path) -> None:
        """
        Save tree structure as a .png file.

        This version uses networkx + matplotlib.
        """
        import matplotlib.pyplot as plt
        import networkx as nx

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        graph = nx.DiGraph()

        for parent, children in self.adj_dic.items():
            for child in children:
                graph.add_edge(parent, child)

        labels = {}
        for node, regions in self.node_dic.items():
            if len(regions) == 1:
                labels[node] = f"{node}\n{regions[0]}"
            else:
                labels[node] = node

        pos = _hierarchy_pos(graph, self.root_node)

        plt.figure(figsize=(10, 6))
        nx.draw(
            graph,
            pos,
            labels=labels,
            with_labels=True,
            node_size=2500,
            font_size=8,
            arrows=False
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()


def _hierarchy_pos(
    graph,
    root,
    width: float = 1.0,
    vert_gap: float = 0.2,
    vert_loc: float = 0.0,
    xcenter: float = 0.5,
    pos: Optional[dict] = None,
):
    """
    Helper function to place tree nodes hierarchically.

    Used by HierTree.save_png().
    """
    if pos is None:
        pos = {}

    pos[root] = (xcenter, vert_loc)
    children = list(graph.successors(root))

    if len(children) != 0:
        dx = width / len(children)
        next_x = xcenter - width / 2 - dx / 2

        for child in children:
            next_x += dx
            pos = _hierarchy_pos(
                graph,
                child,
                width=dx,
                vert_gap=vert_gap,
                vert_loc=vert_loc - vert_gap,
                xcenter=next_x,
                pos=pos
            )

    return pos


def tree_str_gene_selection(input_adata, gene_num=10, min_fold_change=1.15, min_in_out_group_ratio=1, min_in_group_fraction=0.5, pvals_adj=0.05, label_key="label", exclude_regions=("nan", "unknown"), print_results=True):
    """
    Select region-specific genes for all tissue regions in input_adata.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.
    gene_num : int
        Maximum number of genes selected for each region.
    min_fold_change : float
        Minimum fold-change threshold.
    min_in_out_group_ratio : float
        Minimum in/out group ratio threshold.
    min_in_group_fraction : float
        Minimum fraction of cells/spots expressing the gene in the target group.
    pvals_adj : float
        Maximum adjusted p-value threshold.
    label_key : str
        Column name in input_adata.obs containing tissue region labels.
    exclude_regions : tuple
        Region labels to exclude.
    print_results : bool
        Whether to print intermediate results.

    Returns
    -------
    region_genes_dic : dict
        Dictionary where keys are tissue regions and values are selected marker genes.
    gene_list : list
        Union of selected genes across all regions.
    """

    region_genes_dic={}

    region_list=input_adata.obs[label_key].value_counts().index.tolist()
    region_list=[region for region in region_list if str(region) not in exclude_regions]

    print(f"Included tissue regions: {region_list}")

    # identify region-specific genes
    for region in region_list:
        region_genes, df1_filtered=get_region_genes(input_adata=input_adata, 
            region=region, 
            label_key=label_key, 
            gene_num=gene_num, 
            min_fold_change=min_fold_change, 
            min_in_out_group_ratio=min_in_out_group_ratio, 
            min_in_group_fraction=min_in_group_fraction, 
            pvals_adj=pvals_adj, 
            print_results=print_results)
        
        region_genes_dic[region]=region_genes

    # summarize the identified genes into a gene list
    gene_list = sorted(set(gene for genes in region_genes_dic.values() for gene in genes))

    return region_genes_dic, gene_list


def spatial_distance(spatial_df, label_key="label", x_key="x", y_key="y", neighbors=10, scale=True):
    #========== Step 0. Formatting ==========#
    # Turn spatial df into spatial AnnData format
    # spa_adata.X: spatial location
    # spa_adata.obs: labels
    spa_loc = spatial_df[[x_key, y_key]].to_numpy()
    spa_adata = sc.AnnData(spa_loc, dtype=spa_loc.dtype)

    # Use numpy array to avoid index alignment issues
    spa_adata.obs[label_key] = spatial_df[label_key].to_numpy() # take only the values in the exact row order

    #========== Step 1. Pairwise Spatial Distances ==========#
    spa_dists = squareform(pdist(spa_adata.X, metric="euclidean"))

    # Get upper triangular distances, excluding the main diagonal
    upper_dists = spa_dists[np.triu_indices(len(spa_dists), k=1)] # k = 1, excluding the main diagnonal

    #========== Step 2. Infer the Radius ==========#
    spots_num = spatial_df.shape[0]

    # qtl is approximately neighbors / (spots_num - 1)
    qtl = (neighbors * spots_num) / (spots_num ** 2 - spots_num)

    # Prevent invalid quantile values
    qtl = min(qtl, 1.0)

    radius = np.quantile(upper_dists, q=qtl)

    #========== Step 3. Calculate the Spatial Neighborhood Components ==========#
    spa_mask = (spa_dists <= radius).astype(int)

    ave_nbr = np.sum(spa_mask) / len(spa_mask)
    print(f"The average number of neighboring spots: {ave_nbr.round(1)}.")

    labels = spa_adata.obs[label_key]
    tissue_regions = labels.dropna().value_counts().index.tolist()
    tissue_regions = [i for i in tissue_regions if str(i).lower() not in ["nan", "unknown"]]

    regions_num = len(tissue_regions)

    if regions_num == 0:
        raise ValueError("No valid tissue regions found after removing 'nan' and 'unknown'.")

    nbr_df = pd.DataFrame(
        np.zeros((regions_num, regions_num)),
        index=tissue_regions,
        columns=tissue_regions
    )

    for i in range(regions_num):
        name = tissue_regions[i]

        # Safer than converting AnnData obs index from string to int
        region_index = np.where(spa_adata.obs[label_key].to_numpy() == name)[0]

        tmp = spa_mask[region_index, :]
        col_indices = np.nonzero(tmp)[1]

        # Unique neighboring spot indices
        uni_indices = list(set(col_indices))

        nbr_labels = spa_adata[uni_indices].obs[label_key]
        nbr_counts = nbr_labels.dropna().value_counts()

        nbr_regions = nbr_counts.index.tolist()
        nbr_regions = [n for n in nbr_regions if str(n).lower() not in ["nan", "unknown"]]

        common_regions = [n for n in tissue_regions if n in nbr_regions]

        print(f"========== {name} ==========")
        print(nbr_counts)

        for region in common_regions:
            nbr_df.loc[name, region] = nbr_counts[region]

    print("========== Spatial Neighborhood Components ==========")
    print(nbr_df.astype(int))

    #========== Step 4. Transform Neighborhood Components into Spatial Distances ==========#
    # Update zero values
    if (nbr_df == 0).any().any():
        nbr_upd = nbr_df + 1
    else:
        nbr_upd = nbr_df.copy()

    # Calculate percentage of neighborhood components
    nbr_pct = pd.DataFrame(
        np.zeros((regions_num, regions_num)),
        index=tissue_regions,
        columns=tissue_regions
    )

    row_sums = nbr_upd.sum(axis=1)

    for i in range(regions_num):
        rowsum = row_sums.iloc[i] - nbr_upd.iloc[i, i]

        for j in range(regions_num):
            if i == j:
                nbr_pct.iloc[i, j] = 1
            else:
                nbr_pct.iloc[i, j] = nbr_upd.iloc[i, j] / rowsum

    print("========== The percentage of Neighborhood Components ==========")
    print(nbr_pct.round(2))

    # Transform into pairwise spatial distances
    dists = pd.DataFrame(
        np.zeros((regions_num, regions_num)),
        index=tissue_regions,
        columns=tissue_regions
    )

    for i in range(regions_num):
        for j in range(regions_num):
            dists.iloc[i, j] = np.log(1 / (nbr_pct.iloc[i, j] * nbr_pct.iloc[j, i]))

    print("========== Transformed Pairwise Spatial Distances ==========")
    print(dists.round(2))

    # Return unscaled distances
    if scale is False:
        return dists

    # Scale pairwise spatial distance by (x - min) / (max - min)
    scaled_dists = pd.DataFrame(
        np.zeros((regions_num, regions_num)),
        index=tissue_regions,
        columns=tissue_regions
    )

    dists_np = dists.to_numpy()
    upper_elements = dists_np[np.triu_indices(len(dists_np), k=1)]

    dists_min = upper_elements.min()
    dists_max = upper_elements.max()

    if dists_max == dists_min:
        print("All pairwise distances are identical. Scaled distance matrix is all zeros.")
        return dists, scaled_dists

    for i in range(regions_num):
        for j in range(regions_num):
            if i != j:
                scaled_dists.iloc[i, j] = (dists.iloc[i, j] - dists_min) / (dists_max - dists_min)

    print("========== Scaled Pairwise Spatial Distances ==========")
    print(scaled_dists.round(2))

    return dists, scaled_dists


def features_distance(input_adata, features_set, label_key="label", scale=True):
    # Keep feature order, remove duplicates, and keep only features present in input_adata
    features_set = list(dict.fromkeys(features_set))
    available_features = [f for f in features_set if f in input_adata.var.index]

    missing_features = [f for f in features_set if f not in input_adata.var.index]
    if len(missing_features) > 0:
        print(f"Warning: {len(missing_features)} features are not in input_adata.var.index and will be skipped:")
        print(missing_features)

    if len(available_features) == 0:
        raise ValueError("None of the features in features_set are found in input_adata.var.index.")

    adata_sub = input_adata[:, input_adata.var.index.isin(available_features)].copy()

    # Get valid tissue regions
    tissue_regions = adata_sub.obs[label_key].value_counts().index.tolist()
    tissue_regions = [
        i for i in tissue_regions
        if str(i).lower() not in ["nan", "unknown"]
    ]

    if len(tissue_regions) == 0:
        raise ValueError("No valid tissue regions found after removing 'nan' and 'unknown'.")

    # Calculate the mean of features within each tissue region
    features_mean_df = pd.DataFrame(
        np.zeros((len(tissue_regions), len(available_features))),
        index=tissue_regions,
        columns=available_features
    )

    for region in tissue_regions:
        region_mask = (adata_sub.obs[label_key].to_numpy() == region)

        for feature in available_features:
            feature_mask = (adata_sub.var.index.to_numpy() == feature)

            tmp = adata_sub[region_mask, feature_mask]

            if sparse.issparse(tmp.X):
                mean_value = tmp.X.mean(axis=0)
                mean_value = np.asarray(mean_value).ravel()[0]
            else:
                mean_value = np.asarray(tmp.X).mean(axis=0).ravel()[0]

            features_mean_df.loc[region, feature] = mean_value

    print("========== Features Mean within each Tissue Region ==========")
    print(features_mean_df.round(2))

    # Evaluate pairwise distances between tissue regions
    dists = squareform(pdist(features_mean_df, metric="euclidean"))
    dists_df = pd.DataFrame(dists, index=tissue_regions, columns=tissue_regions)

    print("========== Pairwise Distances between Tissue Regions ==========")
    print(dists_df.round(2))

    if scale is False:
        return dists_df

    # Scale pairwise distances by min-max scaling
    scaled_dists_df = pd.DataFrame(
        np.zeros(dists.shape),
        index=tissue_regions,
        columns=tissue_regions
    )

    upper_diag_elements = dists[np.triu_indices(len(dists), k=1)]

    dists_min = upper_diag_elements.min()
    dists_max = upper_diag_elements.max()

    # Handle edge case: all pairwise distances are identical
    if dists_max == dists_min:
        print("Warning: All pairwise distances are identical. Scaled non-diagonal distances are set to 0.")
    else:
        for i in range(len(tissue_regions)):
            for j in range(len(tissue_regions)):
                if i != j:
                    scaled_dists_df.iloc[i, j] = (
                        (dists_df.iloc[i, j] - dists_min) / (dists_max - dists_min)
                    )

    print("========== Scaled Pairwise Distances between Tissue Regions ==========")
    print(scaled_dists_df.round(2))

    return dists_df, scaled_dists_df


def rank_dists(scaled_dists, method="average"):
    """
    Rank pairwise distances in a square symmetric distance matrix.

    Parameters
    ----------
    scaled_dists : pandas.DataFrame
        Square distance matrix with identical row and column labels.
    method : str
        Ranking method passed to scipy.stats.rankdata.
        Common choices: "average", "min", "dense".

    Returns
    -------
    scaled_dists_ranks : pandas.DataFrame
        Symmetric matrix of pairwise distance ranks.
        Diagonal values are 0.
    """

    # Check input shape
    if scaled_dists.shape[0] != scaled_dists.shape[1]:
        raise ValueError("scaled_dists must be a square distance matrix.")

    # Check row/column labels
    if not scaled_dists.index.equals(scaled_dists.columns):
        raise ValueError("scaled_dists must have identical row and column labels.")

    # Check symmetry
    if not np.allclose(scaled_dists.values, scaled_dists.values.T, equal_nan=True):
        raise ValueError("scaled_dists must be symmetric.")

    tissue_regions = scaled_dists.index.tolist()
    n_regions = len(tissue_regions)

    scaled_dists_array = scaled_dists.to_numpy()

    # Extract upper triangle, excluding diagonal
    upper_diag_indices = np.triu_indices(n_regions, k=1)
    upper_diag_elements = scaled_dists_array[upper_diag_indices]

    # Rank distances
    ranks = rankdata(upper_diag_elements, method=method)

    # Use float if method="average"; otherwise int is usually okay
    dtype = float if method == "average" else int

    scaled_dists_ranks = pd.DataFrame(
        np.zeros(scaled_dists.shape),
        index=tissue_regions,
        columns=tissue_regions,
        dtype=dtype
    )

    # Assign ranks symmetrically
    for rank_value, i, j in zip(ranks, upper_diag_indices[0], upper_diag_indices[1]):
        scaled_dists_ranks.iloc[i, j] = rank_value
        scaled_dists_ranks.iloc[j, i] = rank_value

    print("========== The Ranking of Pairwise Distances ==========")
    print(scaled_dists_ranks)

    return scaled_dists_ranks


def multi_modal_distance(
    input_adata, 
    features_dic, 
    w_G=1, 
    w_I=0, 
    w_S=0.5, 
    neighbors=None, 
    shape="hexagon", 
    x_key="x", 
    y_key="y", 
    label_key="label", 
    scale=True):
    weighted_dists_list = []

    # -----------------------------
    # Gene expression distances
    # -----------------------------
    if w_G > 0:
        gene_list = features_dic.get("gene", [])

        if len(gene_list) == 0:
            raise ValueError("w_G > 0, but features_dic['gene'] is missing or empty.")

        print("---------------------------------------------------------------------------------")
        print("================================ Gene Expression ================================")
        print("---------------------------------------------------------------------------------")

        dists_gene, scaled_dists_gene = features_distance(
            input_adata=input_adata,
            features_set=gene_list,
            label_key=label_key,
            scale=scale
        )

        weighted_dists_list.append((w_G, scaled_dists_gene))

    # -----------------------------
    # Image feature distances
    # -----------------------------
    if w_I > 0:
        image_list = features_dic.get("image", [])

        if len(image_list) == 0:
            raise ValueError("w_I > 0, but features_dic['image'] is missing or empty.")

        print("---------------------------------------------------------------------------------")
        print("================================ Image Features =================================")
        print("---------------------------------------------------------------------------------")

        dists_img, scaled_dists_img = features_distance(
            input_adata=input_adata,
            features_set=image_list,
            label_key=label_key,
            scale=scale
        )

        weighted_dists_list.append((w_I, scaled_dists_img))

    # -----------------------------
    # Spatial neighborhood composition distances
    # -----------------------------
    if w_S > 0:
        if neighbors is None:
            if shape == "hexagon":
                neighbors = 6
            elif shape == "square":
                neighbors = 4
            else:
                raise ValueError(
                    "shape must be either 'hexagon' for Visium data or 'square' for ST data."
                )

        required_cols = [x_key, y_key, label_key]
        missing_cols = [col for col in required_cols if col not in input_adata.obs.columns]

        if len(missing_cols) > 0:
            raise ValueError(f"Missing columns in input_adata.obs: {missing_cols}")

        spatial_df = input_adata.obs[[x_key, y_key, label_key]].copy()

        print("---------------------------------------------------------------------------------------")
        print("================================ Spatial Neighborhoods ================================")
        print("---------------------------------------------------------------------------------------")

        dists_spa, scaled_dists_spa = spatial_distance(
            spatial_df=spatial_df,
            label_key=label_key,
            x_key=x_key,
            y_key=y_key,
            neighbors=neighbors,
            scale=scale
        )

        weighted_dists_list.append((w_S, scaled_dists_spa))

    # -----------------------------
    # Check at least one modality is used
    # -----------------------------
    if len(weighted_dists_list) == 0:
        raise ValueError("At least one of w_G, w_I, or w_S must be greater than 0.")

    # -----------------------------
    # Check distance matrix consistency
    # -----------------------------
    first_dists = weighted_dists_list[0][1]
    base_index = first_dists.index
    base_columns = first_dists.columns

    for _, dists in weighted_dists_list[1:]:
        if not dists.index.equals(base_index) or not dists.columns.equals(base_columns):
            raise ValueError(
                "Distance matrices from different modalities have inconsistent row or column labels."
            )

    # -----------------------------
    # Integrate distances
    # -----------------------------
    scaled_dists_all = sum(
        weight * dists for weight, dists in weighted_dists_list
    )

    # -----------------------------
    # Rank the integrated distances
    # -----------------------------
    print("================================ Rank based on overall distance ================================")

    scaled_dists_ranks = rank_dists(scaled_dists_all)

    return scaled_dists_all, scaled_dists_ranks


def integrate_distance_matrices(dists_dic, spot_counts, fill_diagonal=True, return_pair_weights=False):
    """
    Integrate region-level distance matrices across multiple samples.

    For each region pair, only samples containing both regions are used.
    The sample weights are proportional to sample spot counts and are
    re-normalized among the available samples for that region pair.

    Parameters
    ----------
    dists_dic : dict
        Dictionary of sample-specific distance matrices.

        Example:
            {
                "H1": dists_H1,
                "G2": dists_G2,
                "B1": dists_B1,
            }

        Each value should be a square pandas DataFrame with region names
        as both index and columns.

    spot_counts : dict
        Dictionary mapping sample names to total spot numbers.

        Example:
            {
                "H1": adata1.shape[0],
                "G2": adata2.shape[0],
                "B1": adata3.shape[0],
            }

    fill_diagonal : bool, default=True
        Whether to force the diagonal values to 0.

    return_pair_weights : bool, default=False
        Whether to also return the pairwise sample weights used for each
        region pair.

    Returns
    -------
    integrated_dists : pandas.DataFrame
        Integrated distance matrix across samples.

    pair_weight_dic : dict, optional
        Returned only when return_pair_weights=True.
        Keys are region pairs and values are dictionaries of normalized
        sample weights used for that pair.
    """

    if len(dists_dic) == 0:
        raise ValueError("dists_dic is empty.")

    missing_counts = [s for s in dists_dic if s not in spot_counts]
    if len(missing_counts) > 0:
        raise ValueError(f"Missing spot counts for samples: {missing_counts}")

        if spot_counts[sample_name] <= 0:
            raise ValueError(f"{sample_name}: spot count must be positive.")

    # Union of all regions across all samples
    all_regions = sorted(
        set(region for dists in dists_dic.values() for region in dists.index)
    )

    integrated_dists = pd.DataFrame(
        np.nan,
        index=all_regions,
        columns=all_regions,
        dtype=float,
    )

    pair_weight_dic = {}

    for region_i in all_regions:
        for region_j in all_regions:

            available_samples = []
            raw_weights = []
            dist_values = []

            for sample_name, dists in dists_dic.items():
                has_i = region_i in dists.index
                has_j = region_j in dists.columns

                if has_i and has_j:
                    dist_value = dists.loc[region_i, region_j]

                    if pd.notna(dist_value):
                        available_samples.append(sample_name)
                        raw_weights.append(spot_counts[sample_name])
                        dist_values.append(dist_value)

            if len(available_samples) == 0:
                continue

            raw_weights = np.asarray(raw_weights, dtype=float)
            norm_weights = raw_weights / raw_weights.sum()
            dist_values = np.asarray(dist_values, dtype=float)

            integrated_dists.loc[region_i, region_j] = np.sum(
                norm_weights * dist_values
            )

            if return_pair_weights:
                pair_weight_dic[(region_i, region_j)] = dict(
                    zip(available_samples, norm_weights)
                )

    # Keep symmetry stable, in case of tiny numerical differences
    integrated_dists = (integrated_dists + integrated_dists.T) / 2

    if fill_diagonal:
        np.fill_diagonal(integrated_dists.values, 0.0)

    if return_pair_weights:
        return integrated_dists, pair_weight_dic

    return integrated_dists


def multi_sample_distance(
    adata_dic,
    features_dic,
    w_G=1,
    w_I=0,
    w_S=0.5,
    neighbors=None,
    shape="hexagon",
    x_key="x",
    y_key="y",
    label_key="label",
    scale=True,
    return_sample_dists=True,
):
    """
    Compute and integrate region-level distances across multiple samples.

    This function first calls multi_modal_distance() for each sample, then
    integrates the resulting distance matrices using sample spot-number weights.

    Parameters
    ----------
    adata_dic : dict
        Dictionary of AnnData objects.

        Example:
            {
                "H1": adata1,
                "G2": adata2,
                "B1": adata3,
            }

    features_dic : dict
        Either a single feature dictionary shared by all samples:

            {
                "gene": genes_features,
                "image": image_features,
            }

        or a nested dictionary with sample-specific feature dictionaries:

            {
                "H1": {"gene": genes_H1, "image": image_H1},
                "G2": {"gene": genes_G2, "image": image_G2},
            }

    w_G, w_I, w_S : float
        Modality weights for gene, image, and spatial distances.

    neighbors : int or None
        Spatial neighbor radius. If None, decided by shape.

    shape : {"hexagon", "square"}
        Spatial platform layout.

    x_key, y_key, label_key : str
        Column names in adata.obs.

    scale : bool
        Whether to scale modality-specific distance matrices.

    return_sample_dists : bool
        Whether to return sample-level distance matrices.

    Returns
    -------
    integrated_dists : pandas.DataFrame
        Integrated distance matrix across samples.

    integrated_ranks : pandas.DataFrame
        Rank matrix of the integrated distance matrix.

    sample_dists_dic : dict, optional
        Sample-specific integrated multi-modal distance matrices.
    """

    if len(adata_dic) == 0:
        raise ValueError("adata_dic is empty.")

    sample_dists_dic = {}
    sample_rank_dic = {}
    spot_counts = {}

    for sample_name, adata in adata_dic.items():
        print(f"\n==================== Sample: {sample_name} ====================")

        # Support either shared features_dic or sample-specific features_dic
        # if features_dic contains sample-specific features,
        # uses the features for the current sample.
        if sample_name in features_dic and isinstance(features_dic[sample_name], dict):
            current_features_dic = features_dic[sample_name]
        else:
            current_features_dic = features_dic

        sample_dists, sample_ranks = multi_modal_distance(
            input_adata=adata,
            features_dic=current_features_dic,
            w_G=w_G,
            w_I=w_I,
            w_S=w_S,
            neighbors=neighbors,
            shape=shape,
            x_key=x_key,
            y_key=y_key,
            label_key=label_key,
            scale=scale,
        )

        sample_dists_dic[sample_name] = sample_dists
        sample_rank_dic[sample_name] = sample_ranks
        spot_counts[sample_name] = adata.shape[0]

    print("\n==================== Integrating distances across samples ====================")

    integrated_dists = integrate_distance_matrices(
        dists_dic=sample_dists_dic,
        spot_counts=spot_counts,
        fill_diagonal=True,
    )

    integrated_ranks = rank_dists(integrated_dists)

    if return_sample_dists:
        return integrated_dists, integrated_ranks, sample_dists_dic

    return integrated_dists, integrated_ranks


def identify_region_pairs(rank_matrix):
    """
    Convert a rank matrix into an ordered list of region pairs.

    Parameters
    ----------
    rank_matrix : pandas.DataFrame
        Square rank matrix.

    Returns
    -------
    region_pairs : list of tuple
        Each tuple is:
        (rank_value, region_1, region_2)
    """

    if rank_matrix.shape[0] != rank_matrix.shape[1]:
        raise ValueError("rank_matrix must be square.")

    if not rank_matrix.index.equals(rank_matrix.columns):
        raise ValueError("rank_matrix must have identical row and column labels.")

    region_names = rank_matrix.columns.tolist()
    regions_num = len(region_names)

    region_pairs = []

    for i in range(regions_num - 1):
        for j in range(i + 1, regions_num):
            rank_value = rank_matrix.iloc[i, j]
            region_pairs.append((rank_value, region_names[i], region_names[j]))

    region_pairs = sorted(region_pairs, key=lambda x: x[0])

    return region_pairs


def pairs_to_nodes(region_pairs, region_names):
    """
    Build hierarchical nodes from ordered region pairs.

    Parameters
    ----------
    region_pairs : list of tuple
        Output from identify_region_pairs().
        Each tuple is:
        (rank_value, region_1, region_2)

    region_names : list
        Original tissue region names.

    Returns
    -------
    node_dic : dict
        node -> list of regions

    hier_dic : dict
        hierarchy level -> list of nodes

    adj_dic : dict
        parent node -> two child nodes
    """

    node_dic = {}
    hier_dic = {}
    adj_dic = {}

    # Initialize leaf nodes
    for i, region in enumerate(region_names):
        node_dic[f"node{i}"] = [region]

    hier_dic[0] = list(node_dic.keys())
    node_num = len(region_names) - 1

    for rank, region0, region1 in region_pairs:
        # Find the current newest node containing region0
        node0_candidates = [
            key for key, value in node_dic.items()
            if region0 in value
        ]
        node0 = node0_candidates[-1]

        # Find the current newest node containing region1
        node1_candidates = [
            key for key, value in node_dic.items()
            if region1 in value
        ]
        node1 = node1_candidates[-1]

        # If they are already in the same node, skip
        if node0 == node1:
            continue

        node0_hier = [
            hier for hier, node_list in hier_dic.items()
            if node0 in node_list
        ][0]

        node1_hier = [
            hier for hier, node_list in hier_dic.items()
            if node1 in node_list
        ][0]

        # Merge two nodes
        node_num += 1
        new_node = f"node{node_num}"
        new_node_hier = max(node0_hier, node1_hier) + 1
        new_node_regions = node_dic[node0] + node_dic[node1]

        node_dic[new_node] = new_node_regions

        if new_node_hier not in hier_dic:
            hier_dic[new_node_hier] = []

        hier_dic[new_node_hier].append(new_node)
        adj_dic[new_node] = [node0, node1]

        # Stop once all regions are merged into one root node
        if len(new_node_regions) == len(region_names):
            break

    return node_dic, hier_dic, adj_dic


def traverse_node_path(adj_dic, start_node, end_node, path=None):
    """
    Traverse from a leaf node to the root node.

    Returns
    -------
    path : list
        Node path from start_node to end_node.
    """

    if path is None:
        path = []

    path = path + [start_node]

    if start_node == end_node:
        return path

    for root_node, leaf_nodes in adj_dic.items():
        if start_node in leaf_nodes:
            if root_node not in path:
                new_path = traverse_node_path(
                    adj_dic=adj_dic,
                    start_node=root_node,
                    end_node=end_node,
                    path=path
                )

                if new_path is not None:
                    return new_path

    return None


def node_to_tree(node_dic, adj_dic, region_names, show=True):
    """
    Convert node/adjacency dictionaries into tree paths.

    Returns
    -------
    path_dic : dict
        leaf node -> path from leaf to root

    tree_list : list
        list of tree paths compatible with bigtree.list_to_tree
    """

    path_dic = {}
    tree_list = []

    root_candidates = [
        key for key, value in node_dic.items()
        if len(value) == len(region_names)
    ]

    if len(root_candidates) != 1:
        raise ValueError(
            "Expected exactly one root node, but found "
            f"{len(root_candidates)} root candidates: {root_candidates}"
        )

    end_node = root_candidates[0]

    for region in region_names:
        start_candidates = [
            key for key, value in node_dic.items()
            if value == [region]
        ]

        if len(start_candidates) != 1:
            raise ValueError(f"Expected one leaf node for region {region}.")

        start_node = start_candidates[0]

        path = traverse_node_path(
            adj_dic=adj_dic,
            start_node=start_node,
            end_node=end_node
        )

        if path is None:
            raise ValueError(f"No path found from {start_node} to {end_node}.")

        path_dic[start_node] = path

        node_tree_path = region
        for i in range(1, len(path)):
            node_tree_path = path[i] + "/" + node_tree_path

        tree_list.append(node_tree_path)

    if show is True:
        try:
            from bigtree import list_to_tree
            tree_str = list_to_tree(tree_list)
            tree_str.hshow()
        except ImportError:
            print("bigtree is not installed. Showing tree paths instead:")
            for path in tree_list:
                print(path)

    return path_dic, tree_list


def build_hier_tree(rank_matrix, show=True):
    """
    Build a hierarchical tree object from a pairwise rank matrix.

    Parameters
    ----------
    rank_matrix : pandas.DataFrame
        Square symmetric rank matrix.
        Rows and columns should be tissue region names.

    show : bool
        Whether to print/show the tree.

    Returns
    -------
    tree : HierTree
        Tree object containing node_dic, hier_dic, adj_dic, path_dic,
        tree_list, region_names, and root_node.
    """

    region_names = rank_matrix.columns.tolist()

    region_pairs = identify_region_pairs(rank_matrix)

    node_dic, hier_dic, adj_dic = pairs_to_nodes(
        region_pairs=region_pairs,
        region_names=region_names
    )

    path_dic, tree_list = node_to_tree(
        node_dic=node_dic,
        adj_dic=adj_dic,
        region_names=region_names,
        show=show
    )

    root_candidates = [
        key for key, value in node_dic.items()
        if len(value) == len(region_names)
    ]

    if len(root_candidates) != 1:
        raise ValueError("Could not identify a unique root node.")

    root_node = root_candidates[0]

    tree = HierTree(
        node_dic=node_dic,
        hier_dic=hier_dic,
        adj_dic=adj_dic,
        path_dic=path_dic,
        tree_list=tree_list,
        region_names=region_names,
        root_node=root_node
    )

    return tree


def make_split_table(tree: HierTree) -> pd.DataFrame:
    """
    Prepare tree split information for downstream hierarchical gene selection
    and hierarchical label transfer.

    Each row represents one binary split:
        parent_node -> child_1 vs child_2

    Parameters
    ----------
    tree : HierTree
        Hierarchical tree object returned by build_hier_tree().

    Returns
    -------
    split_df : pandas.DataFrame
        DataFrame containing parent node, child nodes, and their corresponding regions.
    """

    rows = []

    for parent_node, child_1, child_2 in tree.get_split_pairs(order="root_to_leaf"):
        rows.append({
            "parent_node": parent_node,
            "child_1": child_1,
            "child_2": child_2,
            "child_1_regions": ";".join(tree.get_regions(child_1)),
            "child_2_regions": ";".join(tree.get_regions(child_2)),
            "parent_regions": ";".join(tree.get_regions(parent_node)),
        })

    split_df = pd.DataFrame(rows)

    return split_df

