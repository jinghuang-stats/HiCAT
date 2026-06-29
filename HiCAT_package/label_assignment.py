import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Label assignment
# ============================================================
def assign_hierarchical_labels(
    input_adata,
    hier_index=("a", "b"),
    hier_anchor_key=("a_anchors", "b_anchors"),
    infer_key="a_vs_b",
    cluster_key="leiden_clusters",
    x_key="x",
    y_key="y",
    min_cluster_spots=10,
    min_anchor_pct=5,
    unassigned_label="novel_cluster",
    allow_novel_clusters=False,
    prop_diff_cutoff=None,
    anchor_normalizer=1.0,
    reassign_novel=True,
    num_nbs=25,
    copy=False,
    print_results=True,
):
    """
    Assign hierarchical labels to spatial spots based on cluster-level anchor enrichment.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.
    hier_index : tuple or list
        Hierarchical labels to assign, for example ("a", "b").
    hier_anchor_key : tuple or list
        Columns in input_adata.obs storing anchor indicators or anchor scores
        for each hierarchy label.
    infer_key : str
        Column name in input_adata.obs for storing inferred labels.
    cluster_key : str
        Column in input_adata.obs storing cluster assignments.
    x_key, y_key : str
        Spatial coordinate columns in input_adata.obs.
    min_cluster_spots : int
        Clusters with fewer than this number of spots are not assigned by
        anchor enrichment and remain as unassigned_label.
    min_anchor_pct : float
        Minimum anchor percentage required for assigning a cluster to a
        hierarchy label.
    unassigned_label : str
        Label used for clusters that cannot be confidently assigned.
    allow_novel_clusters : bool
        If True, small novel clusters are allowed to remain novel. Larger
        novel clusters may still be reassigned by neighborhood smoothing.
    prop_diff_cutoff : float or None
        If not None, clusters whose top two hierarchy anchor percentages are
        within this cutoff are adjusted using anchor distribution weights.
        This is mainly useful for binary splits.
    anchor_normalizer : float or dict
        Normalizer for anchor values. Use 1.0 for binary anchors. For
        multi-modality summed anchors, use the total possible anchor score,
        for example 2.0 if Gene + Protein anchors are summed. 
        anchor_normalizer = (anchor_weight_G + anchor_weight_P) if having different weights
        A dict can also be used, e.g. {"a": 2.0, "b": 2.0}.
    reassign_novel : bool
        Whether to reassign novel clusters based on neighboring assigned spots.
    num_nbs : int
        Number of neighbors used when reassigning novel clusters.
    copy : bool
        If True, return a copied AnnData object. If False, modify input_adata
        in place unless it is an AnnData view.
    print_results : bool
        Whether to print summary tables.

    Returns
    -------
    adata : AnnData
        AnnData object with inferred labels added to adata.obs[infer_key].
    cross_table : pandas.DataFrame
        Cluster-level anchor percentage table.
    cross_table_upd : pandas.DataFrame
        Updated cluster-level anchor table after optional weighting adjustment.
    """

    if len(hier_index) != len(hier_anchor_key):
        raise ValueError("hier_index and hier_anchor_key must have the same length.")

    required_cols = [cluster_key, *hier_anchor_key]
    if reassign_novel:
        required_cols += [x_key, y_key]

    missing_cols = [col for col in required_cols if col not in input_adata.obs.columns]
    if len(missing_cols) > 0:
        raise KeyError(f"Missing columns in input_adata.obs: {missing_cols}")

    adata = input_adata.copy() if copy or input_adata.is_view else input_adata

    obs = adata.obs
    clusters = obs[cluster_key].dropna().unique().tolist()
    cluster_sizes = obs[cluster_key].value_counts()

    keep_clusters = cluster_sizes[cluster_sizes >= min_cluster_spots].index.tolist()
    drop_clusters = cluster_sizes[cluster_sizes < min_cluster_spots].index.tolist()

    if print_results and len(drop_clusters) > 0:
        print("Dropped small clusters:", ", ".join(map(str, drop_clusters)))

    # Initialize all spots as unassigned.
    adata.obs[infer_key] = unassigned_label

    # Build cluster-level anchor percentage table.
    cross_table = pd.DataFrame(
        0.0,
        index=list(hier_index),
        columns=clusters,
    )

    for label, anchor_key in zip(hier_index, hier_anchor_key):
        if isinstance(anchor_normalizer, dict):
            normalizer = anchor_normalizer.get(label, 1.0)
        else:
            normalizer = anchor_normalizer

        if normalizer <= 0:
            raise ValueError("anchor_normalizer must be positive.")

        anchor_values = pd.to_numeric(obs[anchor_key], errors="coerce").fillna(0)

        anchor_sum_by_cluster = anchor_values.groupby(obs[cluster_key]).sum()
        prop = anchor_sum_by_cluster / cluster_sizes / normalizer * 100

        cross_table.loc[label, prop.index] = prop

    if print_results:
        print("========== Cross Table of Anchors ==========")
        print(cross_table.round(2))

    # Use only sufficiently large clusters for initial assignment.
    assign_table = cross_table.loc[:, keep_clusters].copy()

    if assign_table.shape[1] == 0:
        adata.obs[infer_key] = adata.obs[infer_key].astype("category")
        return adata, cross_table, assign_table

    max_label = assign_table.idxmax(axis=0)
    max_prop = assign_table.max(axis=0)

    cross_table_upd = assign_table.copy()

    # Optional adjustment for ambiguous clusters.
    if prop_diff_cutoff is not None and len(hier_index) == 2:
        prop_diff = (assign_table.iloc[0, :] - assign_table.iloc[1, :]).abs()
        similar_clusters = prop_diff[prop_diff <= prop_diff_cutoff].index.tolist()

        if print_results:
            print("========== Absolute Anchor Proportion Difference ==========")
            print(prop_diff.sort_values())

        if len(similar_clusters) > 0:
            if print_results:
                print(
                    "Clusters with similar anchor proportions:",
                    ", ".join(map(str, similar_clusters)),
                )

            for label, anchor_key in zip(hier_index, hier_anchor_key):
                anchor_mask = pd.to_numeric(obs[anchor_key], errors="coerce").fillna(0) > 0
                anchor_obs = obs.loc[anchor_mask]

                if anchor_obs.shape[0] == 0:
                    continue

                anchor_cluster_prop = anchor_obs[cluster_key].value_counts(normalize=True)

                for cluster in similar_clusters:
                    if cluster in anchor_cluster_prop.index:
                        cross_table_upd.loc[label, cluster] = (
                            cross_table.loc[label, cluster]
                            * anchor_cluster_prop.loc[cluster]
                        )

            max_label = cross_table_upd.idxmax(axis=0)
            max_prop = cross_table_upd.max(axis=0)

            if print_results:
                print("========== Updated Cross Table ==========")
                print(cross_table_upd.round(2))

    # Assign clusters by maximum anchor proportion.
    assigned_clusters = []

    for cluster in assign_table.columns:
        if max_prop.loc[cluster] > min_anchor_pct:
            label = max_label.loc[cluster]
            adata.obs.loc[adata.obs[cluster_key] == cluster, infer_key] = label
            assigned_clusters.append(cluster)

    novel_clusters = [
        cluster for cluster in keep_clusters
        if cluster not in assigned_clusters
    ]

    if print_results:
        print("Novel / unassigned clusters:", novel_clusters)
        print("========== Before Novel Cluster Reassignment ==========")
        print(adata.obs[infer_key].value_counts())

    # Optionally reassign large novel clusters by spatial neighborhood.
    if reassign_novel and len(novel_clusters) > 0:
        if allow_novel_clusters:
            cluster_props = cluster_sizes / cluster_sizes.sum()
            novel_clusters = [
                cluster for cluster in novel_clusters
                if cluster_props.loc[cluster] > 0.1
            ]

        for novel_cluster in novel_clusters:
            reassign_novel_cluster_by_knn(
                adata,
                novel_cluster=novel_cluster,
                cluster_key=cluster_key,
                infer_key=infer_key,
                x_key=x_key,
                y_key=y_key,
                unassigned_label=unassigned_label,
                num_nbs=num_nbs,
                copy=False,
                print_results=print_results,
            )

    adata.obs[infer_key] = adata.obs[infer_key].astype("category")

    if print_results:
        print("========== Inferred Labels ==========")
        print(adata.obs[infer_key].value_counts())

    return adata, cross_table, cross_table_upd


def reassign_novel_cluster_by_knn(
    input_adata,
    novel_cluster,
    cluster_key,
    infer_key,
    x_key="x",
    y_key="y",
    unassigned_label="novel_cluster",
    num_nbs=25,
    metric="euclidean",
    copy=False,
    print_results=True,
):
    """
    Reassign one novel cluster based on the labels of nearby assigned spots.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.
    novel_cluster : str or int
        Cluster ID to reassign.
    cluster_key : str
        Column in input_adata.obs storing cluster assignments.
    infer_key : str
        Column in input_adata.obs storing current inferred labels.
    x_key, y_key : str
        Spatial coordinate columns.
    unassigned_label : str
        Label used for unassigned or novel spots.
    num_nbs : int
        Number of nearest assigned spots used for reassignment.
    metric : str
        Distance metric used by sklearn.neighbors.NearestNeighbors.
    copy : bool
        If True, return a copied AnnData object. If False, modify in place.
    print_results : bool
        Whether to print reassignment summary.

    Returns
    -------
    adata : AnnData
        AnnData object with the novel cluster reassigned if possible.
    """

    adata = input_adata.copy() if copy or input_adata.is_view else input_adata
    obs = adata.obs

    novel_mask = obs[cluster_key] == novel_cluster
    assigned_mask = obs[infer_key] != unassigned_label

    if novel_mask.sum() == 0:
        return adata

    if assigned_mask.sum() == 0:
        if print_results:
            print(f"No assigned spots available to reassign cluster {novel_cluster}.")
        return adata

    novel_coords = obs.loc[novel_mask, [x_key, y_key]].to_numpy()
    assigned_coords = obs.loc[assigned_mask, [x_key, y_key]].to_numpy()
    assigned_labels = obs.loc[assigned_mask, infer_key].to_numpy()

    k = min(num_nbs, assigned_coords.shape[0])

    nbrs = NearestNeighbors(n_neighbors=k, metric=metric)
    nbrs.fit(assigned_coords)

    _, indices = nbrs.kneighbors(novel_coords)

    neighbor_labels = assigned_labels[indices].ravel()
    label_counts = pd.Series(neighbor_labels).value_counts()

    if label_counts.empty:
        return adata

    reassigned_label = label_counts.idxmax()

    adata.obs.loc[novel_mask, infer_key] = reassigned_label

    if print_results:
        print(f"---------------- novel cluster {novel_cluster} ----------------")
        print(label_counts)
        print(
            f"Based on neighborhood composition, novel cluster "
            f"{novel_cluster} reassignment: {reassigned_label}"
        )

    return adata


def refine_labels(
    input_adata,
    pred_key,
    refined_key,
    num_nbs=25,
    x_key="x",
    y_key="y",
    dists_metric="euclidean",
    copy=False,
):
    """
    Refine spot-level labels using spatial nearest-neighbor majority voting.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.
    pred_key : str
        Column in input_adata.obs containing original predicted labels.
    refined_key : str
        Column name for storing refined labels.
    num_nbs : int
        Number of nearest neighbors used for majority voting.
        The spot itself is also included internally.
    x_key, y_key : str
        Spatial coordinate columns in input_adata.obs.
    dists_metric : str
        Distance metric used by sklearn.neighbors.NearestNeighbors.
    copy : bool
        If True, return a copied AnnData object. If False, modify input_adata
        in place unless it is an AnnData view.

    Returns
    -------
    adata : AnnData
        AnnData object with refined labels in adata.obs[refined_key].
    refined_pred : list
        Refined label for each spot, in the same order as adata.obs.
    """

    required_cols = [pred_key, x_key, y_key]
    missing_cols = [col for col in required_cols if col not in input_adata.obs.columns]
    if len(missing_cols) > 0:
        raise KeyError(f"Missing columns in input_adata.obs: {missing_cols}")

    adata = input_adata.copy() if copy or input_adata.is_view else input_adata

    obs = adata.obs
    spot_ids = obs.index.to_numpy()
    pred = obs[pred_key].astype(str).to_numpy()
    coords = obs[[x_key, y_key]].to_numpy()

    if coords.shape[0] == 0:
        adata.obs[refined_key] = pd.Categorical([])
        return adata, []

    k = min(num_nbs + 1, coords.shape[0])

    nbrs = NearestNeighbors(n_neighbors=k, metric=dists_metric)
    nbrs.fit(coords)

    _, indices = nbrs.kneighbors(coords)

    refined_pred = []

    for i, nb_idx in enumerate(indices):
        nb_labels = pred[nb_idx]
        self_label = pred[i]

        label_counts = pd.Series(nb_labels).value_counts()

        self_count = label_counts.get(self_label, 0)
        top_label = label_counts.idxmax()
        top_count = label_counts.max()

        if (self_count < num_nbs / 2) and (top_count > num_nbs / 2):
            refined_pred.append(top_label)
        else:
            refined_pred.append(self_label)

    adata.obs[refined_key] = pd.Categorical(refined_pred, categories=pd.unique(refined_pred))

    return adata, refined_pred


