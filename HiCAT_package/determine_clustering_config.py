import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# Local import
from .preprocessing import *
from .util import *
from .multi_modal_integration import *

#=======================================================================
# Part 1. Select the informative modalities
#=======================================================================
def evaluate_modality_ari(
    modality_ref_dic,
    n_pcs_dic=None,
    default_pcs_num=30,
    label_key="label",
    min_spots=10,
    exclude_regions=("nan", "unknown"),
    exclude_mode="exact",
    random_state=0,
    print_results=True,
):
    """
    Evaluate each modality by computing clustering ARI across reference sections.

    For each modality and reference section, this function:
        1. filters spots/cells with invalid labels,
        2. computes PCA embeddings,
        3. performs K-means clustering using the number of annotated labels,
        4. computes adjusted Rand index between true labels and clusters,
        5. summarizes ARI scores across reference sections.

    Parameters
    ----------
    modality_ref_dic : dict
        Dictionary of modality-specific reference AnnData dictionaries.

        Example:
        {
            "gene": {"H1": adata_gene_H1, "G2": adata_gene_G2},
            "image": {"H1": adata_img_H1, "G2": adata_img_G2},
        }

    n_pcs_dic : dict or None, optional
        Dictionary specifying the number of PCs for each modality.
        If a modality is not found, `default_pcs_num` is used.

    default_pcs_num : int, optional
        Default number of PCs used for PCA embedding.

    label_key : str, optional
        Column in `adata.obs` containing annotated tissue labels.

    min_spots : int, optional
        Minimum number of spots/cells required to evaluate a section.

    exclude_regions : tuple, optional
        Labels to exclude before ARI calculation.

    exclude_mode : {"exact", "contains"}, optional
        Whether to exclude labels by exact matching or substring matching.

    random_state : int, optional
        Random seed for PCA and K-means.

    print_results : bool, optional
        Whether to print progress and ARI results.

    Returns
    -------
    ari_df : pandas.DataFrame
        Section-level ARI evaluation results.

    avg_ari : pandas.DataFrame
        Modality-level ARI summary across successful reference sections.

    ranked_modalities : list
        Modalities ranked by average ARI in descending order.
    """

    if n_pcs_dic is None:
        n_pcs_dic = {}

    ari_records = []

    for modality, ref_dic in modality_ref_dic.items():

        pcs_num = n_pcs_dic.get(modality, default_pcs_num)

        if print_results:
            print("\n============================================================")
            print(f"Evaluating modality: {modality}")
            print("============================================================")

        for section, adata in ref_dic.items():

            if label_key not in adata.obs.columns:
                raise ValueError(
                    f"{modality} - {section}: label_key='{label_key}' "
                    "is not found in adata.obs."
                )

            if adata.n_obs < min_spots:
                ari_records.append(
                    {
                        "modality": modality,
                        "section": section,
                        "n_obs": adata.n_obs,
                        "n_features": adata.n_vars,
                        "n_labels": np.nan,
                        "pcs_num": pcs_num,
                        "ari": np.nan,
                        "status": f"skipped: fewer than {min_spots} spots/cells",
                    }
                )
                continue

            labels = adata.obs[label_key].astype(str)

            valid_mask = get_valid_label_mask(
            	labels = labels,
            	exclude_regions = exclude_regions,
            	exclude_mode = exclude_mode
            	)

            if valid_mask.sum() < min_spots:
                ari_records.append(
                    {
                        "modality": modality,
                        "section": section,
                        "n_obs": valid_mask.sum(),
                        "n_features": adata.n_vars,
                        "n_labels": np.nan,
                        "pcs_num": pcs_num,
                        "ari": np.nan,
                        "status": "skipped: insufficient valid labels",
                    }
                )
                continue

            adata_use = adata[valid_mask.values, :].copy()
            y_true = adata_use.obs[label_key].astype(str).values

            unique_labels = pd.Series(y_true).unique()
            n_clusters = len(unique_labels)

            if n_clusters < 2:
                ari_records.append(
                    {
                        "modality": modality,
                        "section": section,
                        "n_obs": adata_use.n_obs,
                        "n_features": adata_use.n_vars,
                        "n_labels": n_clusters,
                        "pcs_num": pcs_num,
                        "ari": np.nan,
                        "status": "skipped: fewer than 2 labels",
                    }
                )
                continue

            if n_clusters >= adata_use.n_obs:
                ari_records.append(
                    {
                        "modality": modality,
                        "section": section,
                        "n_obs": adata_use.n_obs,
                        "n_features": adata_use.n_vars,
                        "n_labels": n_clusters,
                        "pcs_num": pcs_num,
                        "ari": np.nan,
                        "status": "skipped: n_clusters >= n_obs",
                    }
                )
                continue

            try:
                modality_pcs = compute_pca_embedding(
                    input_adata=adata_use,
                    pcs_num=pcs_num,
                    random_state=random_state,
                    sample_name=f"{modality} - {section}",
                )

                y_pred = kmeans_clustering(
                    features_matrix=modality_pcs,
                    n_clusters=n_clusters,
                    random_state=random_state,
                    kmeans_key="kmeans_clusters",
                )

                ari = adjusted_rand_score(y_true, y_pred)

                ari_records.append(
                    {
                        "modality": modality,
                        "section": section,
                        "n_obs": adata_use.n_obs,
                        "n_features": adata_use.n_vars,
                        "n_labels": n_clusters,
                        "pcs_num": pcs_num,
                        "ari": ari,
                        "status": "success",
                    }
                )

                if print_results:
                    print(
                        f"{section}: n_labels={n_clusters}, "
                        f"ARI={ari:.4f}"
                    )

            except Exception as e:
                ari_records.append(
                    {
                        "modality": modality,
                        "section": section,
                        "n_obs": adata_use.n_obs,
                        "n_features": adata_use.n_vars,
                        "n_labels": n_clusters,
                        "pcs_num": pcs_num,
                        "ari": np.nan,
                        "status": f"failed: {str(e)}",
                    }
                )

                if print_results:
                    print(f"{section}: failed - {str(e)}")

    ari_df = pd.DataFrame(ari_records)

    success_df = ari_df[ari_df["status"] == "success"].copy()

    if success_df.empty:
        raise ValueError(
            "No modality was successfully evaluated. Please check the input "
            "reference AnnData dictionaries and label_key."
        )

    avg_ari = (
        success_df
        .groupby("modality", as_index=False)
        .agg(
            average_ari=("ari", "mean"),
            median_ari=("ari", "median"),
            min_ari=("ari", "min"),
            max_ari=("ari", "max"),
            n_sections=("section", "nunique"),
        )
        .sort_values("average_ari", ascending=False)
        .reset_index(drop=True)
    )

    ranked_modalities = avg_ari["modality"].tolist()

    return ari_df, avg_ari, ranked_modalities


def select_modalities_by_ari(
    avg_ari,
    ranked_modalities,
    included_modalities=None,
    hard_threshold=0.3,
    alpha=0.8,
    selection_criterion="relative",
):
    """
    Select informative modalities based on modality-level ARI summaries.

    This function applies three possible selection rules:
        1. hard threshold,
        2. relative threshold,
        3. overlap between hard and relative thresholds.

    Parameters
    ----------
    avg_ari : pandas.DataFrame
        Modality-level ARI summary. Must contain columns:
        "modality" and "average_ari".

    ranked_modalities : list
        Modalities ranked by average ARI in descending order.

    included_modalities : list or None, optional
        Modalities included in the current evaluation.

    hard_threshold : float, optional
        Minimum average ARI required for the hard-threshold rule.

    alpha : float, optional
        Relative threshold multiplier. A modality is selected if its average
        ARI is at least `alpha * max_average_ari`.

    selection_criterion : {"hard", "relative", "both"}, optional
        Criterion used for the final modality selection.

    Returns
    -------
    selected_modalities : list
        Final selected modalities.

    selected_modalities_hard : list
        Modalities selected by the hard-threshold rule.

    selected_modalities_relative : list
        Modalities selected by the relative-threshold rule.

    selection_info : dict
        Dictionary recording selection parameters and thresholds.
    """

    if included_modalities is None:
        included_modalities = avg_ari["modality"].tolist()

    if selection_criterion not in ["hard", "relative", "both"]:
        raise ValueError(
            "selection_criterion must be one of "
            "{'hard', 'relative', 'both'}."
        )

    if avg_ari.empty:
        raise ValueError("avg_ari is empty. No modalities can be selected.")

    if len(ranked_modalities) == 0:
        raise ValueError("ranked_modalities is empty. No fallback is available.")

    # ------------------------------------------------------------
    # Selection rule 1: hard-threshold rule
    # ------------------------------------------------------------
    selected_modalities_hard = avg_ari.loc[
        avg_ari["average_ari"] >= hard_threshold,
        "modality",
    ].tolist()

    if len(selected_modalities_hard) == 0:
            selected_modalities_hard = ["Gene"]

    # ------------------------------------------------------------
    # Selection rule 2: relative-threshold rule
    # ------------------------------------------------------------
    max_average_ari = avg_ari["average_ari"].max()
    relative_threshold = max_average_ari * alpha

    selected_modalities_relative = avg_ari.loc[
        avg_ari["average_ari"] >= relative_threshold,
        "modality",
    ].tolist()

    # ------------------------------------------------------------
    # Final modality selection
    # ------------------------------------------------------------
    if selection_criterion == "hard":
        selected_modalities = selected_modalities_hard

    elif selection_criterion == "relative":
        selected_modalities = selected_modalities_relative

    elif selection_criterion == "both":
        selected_modalities = [
            modality for modality in selected_modalities_hard
            if modality in selected_modalities_relative
        ]

        if len(selected_modalities) == 0:
            selected_modalities = ["Gene"]

    selection_info = {
        "selection_criterion": selection_criterion,
        "hard_threshold": hard_threshold,
        "alpha": alpha,
        "relative_threshold": relative_threshold,
        "max_average_ari": max_average_ari,
    }

    return (
        selected_modalities,
        selected_modalities_hard,
        selected_modalities_relative,
        selection_info,
    )


def select_informative_modalities(
    included_modalities,
    ref_gene_dic=None,
    ref_image_dic=None,
    ref_protein_dic=None,
    label_key="label",
    hard_threshold=0.3,
    alpha=0.8,
    selection_criterion="both",
    n_pcs_dic=None,
    default_pcs_num=30,
    random_state=0,
    min_spots=10,
    exclude_regions=("nan", "unknown"),
    exclude_mode="exact",
    print_results=True,
):
    """
    Select informative modalities based on unsupervised clustering agreement
    with reference ground-truth labels.

    For each included modality and each reference section, this function:
        1. computes PCA embeddings from the modality-specific AnnData.X,
        2. runs KMeans clustering with the number of clusters equal to the
           number of true labels,
        3. evaluates clustering agreement using adjusted Rand index, ARI,
        4. averages ARI across reference sections for each modality,
        5. ranks modalities by average ARI,
        6. selects informative modalities using the specified criterion.

    Parameters
    ----------
    included_modalities : list of str
        Modalities included in the target dataset.

        Supported values are:
        - "Gene"
        - "Image"
        - "Protein"

        Example
        -------
        included_modalities = ["Gene", "Image", "Protein"]

    ref_gene_dic : dict or None, optional
        Dictionary of gene-expression reference AnnData objects.

        Example
        -------
        {
            "H1": gene_adata_H1,
            "G2": gene_adata_G2,
            "E1": gene_adata_E1,
        }

    ref_image_dic : dict or None, optional
        Dictionary of image-feature reference AnnData objects.

    ref_protein_dic : dict or None, optional
        Dictionary of protein-feature reference AnnData objects.

    label_key : str, default="label"
        Column in `adata.obs` containing ground-truth tissue labels.

    hard_threshold : float, default=0.2
        Minimum average ARI required for a modality to be selected under the
        hard-threshold rule.

    alpha : float, default=0.8
        Relative threshold factor. Under the relative-threshold rule, modalities
        with average ARI >= max_average_ARI * alpha are selected.

    selection_criterion : {"hard", "relative", "both"}, default="both"
        Criterion used to select informative modalities.

        - "hard":
            Select modalities passing `hard_threshold`.

        - "relative":
            Select modalities with average ARI >= max_average_ARI * alpha.

        - "both":
            Select the overlap between the hard-threshold and relative-threshold
            selected modalities.

    n_pcs_dic : dict or None, optional
        Number of PCs used for each modality.

        Example
        -------
        {
            "Gene": 30,
            "Image": 20,
            "Protein": 10,
        }

        If a modality is not included in `n_pcs_dic`, `default_pcs_num` is used.

    default_pcs_num : int, default=30
        Default number of PCs for modalities not specified in `n_pcs_dic`.

    random_state : int, default=0
        Random seed used for PCA and KMeans.

    min_spots : int, default=10
        Minimum number of spots/cells required in a reference section to evaluate
        a modality.

    print_results : bool, default=True
        Whether to print summary results.

    Returns
    -------
    results_dic : dict
        Dictionary containing modality evaluation and selection results.

        Keys include:

        - "ari_df":
            Per-section and per-modality ARI results.

        - "avg_ari":
            Average ARI summary for each modality.

        - "ranked_modalities":
            Modalities ranked by decreasing average ARI.

        - "selected_modalities_hard":
            Modalities selected by the hard-threshold rule.

        - "selected_modalities_relative":
            Modalities selected by the relative-threshold rule.

        - "selected_modalities":
            Final selected modalities according to `selection_criterion`.

        - "selection_params":
            Parameters used for modality selection.
    """

    # ------------------------------------------------------------
    # Check inputs
    # ------------------------------------------------------------
    supported_modalities = ["Gene", "Image", "Protein"]

    included_modalities = list(included_modalities)

    invalid_modalities = [
        modality for modality in included_modalities
        if modality not in supported_modalities
    ]

    if len(invalid_modalities) > 0:
        raise ValueError(
            f"Unsupported modalities found: {invalid_modalities}. "
            f"Supported modalities are {supported_modalities}."
        )

    if selection_criterion not in ["hard", "relative", "both"]:
        raise ValueError(
            "selection_criterion must be one of: "
            "'hard', 'relative', or 'both'."
        )

    if n_pcs_dic is None:
        n_pcs_dic = {}

    modality_ref_dic = {
        "Gene": ref_gene_dic,
        "Image": ref_image_dic,
        "Protein": ref_protein_dic,
    }

    # Keep only included modalities
    modality_ref_dic = {
        modality: modality_ref_dic[modality]
        for modality in included_modalities
    }

    for modality, ref_dic in modality_ref_dic.items():
        if ref_dic is None:
            raise ValueError(
                f"{modality} is included in included_modalities, "
                f"but its reference dictionary is None."
            )

        if not isinstance(ref_dic, dict) or len(ref_dic) == 0:
            raise ValueError(
                f"{modality} reference dictionary must be a non-empty dict."
            )

    # ------------------------------------------------------------
    # Compute ARI for each modality and each reference section
    # ------------------------------------------------------------
    ari_df, avg_ari, ranked_modalities = evaluate_modality_ari(
        modality_ref_dic=modality_ref_dic,
        n_pcs_dic=n_pcs_dic,
        default_pcs_num=default_pcs_num,
        label_key=label_key,
        min_spots=min_spots,
        exclude_regions=exclude_regions,
        exclude_mode=exclude_mode,
        random_state=random_state,
        print_results=print_results,
    )

    # ------------------------------------------------------------
    # Modality selection
    # ------------------------------------------------------------
    (
        selected_modalities,
        selected_modalities_hard,
        selected_modalities_relative,
        selection_info,
    ) = select_modalities_by_ari(
        avg_ari=avg_ari,
        ranked_modalities=ranked_modalities,
        included_modalities=included_modalities,
        hard_threshold=hard_threshold,
        alpha=alpha,
        selection_criterion=selection_criterion,
    )

    # ------------------------------------------------------------
    # Store results
    # ------------------------------------------------------------
    results_dic = {
        "ari_df": ari_df,
        "avg_ari": avg_ari,
        "ranked_modalities": ranked_modalities,
        "selected_modalities_hard": selected_modalities_hard,
        "selected_modalities_relative": selected_modalities_relative,
        "selected_modalities": selected_modalities,
        "selection_criterion": selection_criterion,
        "selection_info": selection_info,
        "selection_params": {
            "included_modalities": included_modalities,
            "label_key": label_key,
            "hard_threshold": hard_threshold,
            "alpha": alpha,
            "relative_threshold": selection_info["relative_threshold"],
            "selection_criterion": selection_criterion,
            "n_pcs_dic": n_pcs_dic,
            "default_pcs_num": default_pcs_num,
            "random_state": random_state,
            "min_spots": min_spots,
            "exclude_regions": exclude_regions,
            "exclude_mode": exclude_mode
        },
    }

    if print_results:
        print("\n============================================================")
        print("Modality informativeness summary")
        print("============================================================")
        print(avg_ari)

        print("\nRanked modalities:")
        print(ranked_modalities)

        print("\nSelected by hard-threshold rule:")
        print(selected_modalities_hard)

        print("\nSelected by relative-threshold rule:")
        print(selected_modalities_relative)

        print(f"\nFinal selected modalities using criterion='{selection_criterion}':")
        print(selected_modalities)

    return results_dic


#============================================================================
# Part 2. Determine dimension reduction approach (PCA vs. selected features)
#============================================================================
def evaluate_dim_reduction_for_section(
    ref_section: str,
    selected_modalities: Sequence[str],
    dim_reduction_method: str = "pca",
    ref_gene_dic: Optional[Mapping[str, Any]] = None,
    ref_image_dic: Optional[Mapping[str, Any]] = None,
    ref_protein_dic: Optional[Mapping[str, Any]] = None,
    features_dic: Optional[Mapping[str, Any]] = None,
    features_format: str = "section",
    pcs_num_dic: Optional[Mapping[str, int]] = None,
    default_pcs_num: int = 30,
    label_key: str = "label",
    exclude_regions: Sequence[str] = ("nan", "unknown"),
    exclude_mode: str = "contains",
    scale_embedding: bool = True,
    random_state: int = 0,
    align_by_obs_names: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate one dimension-reduction method for one reference section.

    The logic is unchanged from the original implementation:

    1. get true labels from the reference Gene AnnData;
    2. exclude invalid labels;
    3. set ``n_clusters`` to the number of unique valid labels;
    4. integrate selected modalities using PCA or selected features;
    5. run KMeans on the valid spots;
    6. compute ARI against the true labels.

    The only organizational update is that modality integration and clustering
    are now delegated to shared helpers.
    """

    label_adata = get_ref_modality_adata(
        ref_section=ref_section,
        modality="Gene",
        ref_gene_dic=ref_gene_dic,
        ref_image_dic=ref_image_dic,
        ref_protein_dic=ref_protein_dic,
    )

    if label_key not in label_adata.obs.columns:
        raise KeyError(
            f"{ref_section}: label_key={label_key!r} is not found in adata.obs."
        )

    labels = label_adata.obs[label_key].astype(str)

    valid_mask = get_valid_label_mask(
        labels=labels,
        exclude_regions=exclude_regions,
        exclude_mode=exclude_mode,
    )

    true_labels = labels.loc[valid_mask].to_numpy()
    n_clusters = len(np.unique(true_labels))

    if n_clusters < 2:
        raise ValueError(
            f"{ref_section}: at least two unique labels are required for ARI."
        )

    integrated_embedding, modality_embedding_dic = integrate_modalities_for_section(
        ref_section=ref_section,
        selected_modalities=selected_modalities,
        dim_reduction_method=dim_reduction_method,
        ref_gene_dic=ref_gene_dic,
        ref_image_dic=ref_image_dic,
        ref_protein_dic=ref_protein_dic,
        features_dic=features_dic,
        features_format=features_format,
        pcs_num_dic=pcs_num_dic,
        default_pcs_num=default_pcs_num,
        scale_embedding=scale_embedding,
        random_state=random_state,
        align_by_obs_names=align_by_obs_names,
    )

    if integrated_embedding.shape[0] != label_adata.n_obs:
        raise ValueError(
            f"{ref_section}: integrated embedding has "
            f"{integrated_embedding.shape[0]} observations, but label AnnData has "
            f"{label_adata.n_obs} observations. Please check row alignment."
        )

    integrated_embedding_eval = integrated_embedding[valid_mask.to_numpy(), :]

    pred_labels, cluster_info = cluster_integrated_embedding(
        integrated_embedding=integrated_embedding_eval,
        clustering_config={
            "clustering_method": "kmeans",
            "n_clusters": n_clusters,
            "random_state": random_state,
        },
        cluster_key="dim_reduction_eval_cluster",
    )

    ari = adjusted_rand_score(true_labels, pred_labels)

    result = {
        "ref_section": ref_section,
        "dim_reduction_method": dim_reduction_method,
        "features_format": features_format,
        "selected_modalities": list(selected_modalities),
        "n_clusters": n_clusters,
        "n_valid_obs": int(valid_mask.sum()),
        "n_integrated_features": int(integrated_embedding.shape[1]),
        "ari": ari,
        "cluster_info": cluster_info,
    }

    return result



def evaluate_dim_reduction_method(
    ref_section_list: Sequence[str],
    selected_modalities: Sequence[str],
    dim_reduction_method: str = "pca",
    ref_gene_dic: Optional[Mapping[str, Any]] = None,
    ref_image_dic: Optional[Mapping[str, Any]] = None,
    ref_protein_dic: Optional[Mapping[str, Any]] = None,
    features_dic: Optional[Mapping[str, Any]] = None,
    features_format: str = "section",
    pcs_num_dic: Optional[Mapping[str, int]] = None,
    default_pcs_num: int = 30,
    label_key: str = "label",
    exclude_regions: Sequence[str] = ("nan", "unknown"),
    exclude_mode: str = "contains",
    scale_embedding: bool = True,
    random_state: int = 0,
    align_by_obs_names: bool = False,
    print_results: bool = True,
) -> Tuple[pd.DataFrame, float]:
    """
    Evaluate one dimension-reduction method across reference sections.

    Returns
    -------
    ari_df : pandas.DataFrame
        Section-level ARI table.

    average_ari : float
        Average ARI across reference sections.
    """

    result_list = []

    for ref_section in ref_section_list:
        result = evaluate_dim_reduction_for_section(
            ref_section=ref_section,
            selected_modalities=selected_modalities,
            dim_reduction_method=dim_reduction_method,
            ref_gene_dic=ref_gene_dic,
            ref_image_dic=ref_image_dic,
            ref_protein_dic=ref_protein_dic,
            features_dic=features_dic,
            features_format=features_format,
            pcs_num_dic=pcs_num_dic,
            default_pcs_num=default_pcs_num,
            label_key=label_key,
            exclude_regions=exclude_regions,
            exclude_mode=exclude_mode,
            scale_embedding=scale_embedding,
            random_state=random_state,
            align_by_obs_names=align_by_obs_names,
        )

        result_list.append(result)

        if print_results:
            print(
                f"{ref_section} | {dim_reduction_method} | "
                f"ARI = {result['ari']:.4f} | "
                f"n_clusters = {result['n_clusters']} | "
                f"n_features = {result['n_integrated_features']}"
            )

    ari_df = pd.DataFrame(result_list)
    average_ari = ari_df["ari"].mean()

    if print_results:
        print(
            f"\nAverage ARI for {dim_reduction_method}: "
            f"{average_ari:.4f}"
        )

    return ari_df, average_ari



def determine_dimension_reduction_method(
    ref_section_list: Sequence[str],
    selected_modalities: Sequence[str],
    ref_gene_dic: Optional[Mapping[str, Any]] = None,
    ref_image_dic: Optional[Mapping[str, Any]] = None,
    ref_protein_dic: Optional[Mapping[str, Any]] = None,
    features_dic: Optional[Mapping[str, Any]] = None,
    features_format: str = "section",
    pcs_num_dic: Optional[Mapping[str, int]] = None,
    default_pcs_num: int = 30,
    candidate_methods: Sequence[str] = ("pca", "selected_features"),
    label_key: str = "label",
    exclude_regions: Sequence[str] = ("nan", "unknown"),
    exclude_mode: str = "contains",
    scale_embedding: bool = True,
    random_state: int = 0,
    align_by_obs_names: bool = False,
    print_results: bool = True,
) -> Dict[str, Any]:
    """
    Determine the best dimension-reduction method using reference sections.

    This function compares candidate methods by average ARI across reference
    sections and selects the method with the highest average ARI.
    """

    if selected_modalities is None or len(selected_modalities) == 0:
        raise ValueError("selected_modalities cannot be None or empty.")

    for method in candidate_methods:
        if method not in SUPPORTED_REDUCTION_METHODS:
            raise ValueError(
                f"Unsupported method: {method}. "
                "Supported methods are 'pca' and 'selected_features'."
            )

    if "selected_features" in candidate_methods and features_dic is None:
        raise ValueError(
            "features_dic is required when candidate_methods contains "
            "'selected_features'."
        )

    ari_df_dic = {}
    average_ari_dic = {}

    for method in candidate_methods:
        if print_results:
            print("\n============================================================")
            print(f"Evaluating dimension reduction method: {method}")
            print("============================================================")

        ari_df, average_ari = evaluate_dim_reduction_method(
            ref_section_list=ref_section_list,
            selected_modalities=selected_modalities,
            dim_reduction_method=method,
            ref_gene_dic=ref_gene_dic,
            ref_image_dic=ref_image_dic,
            ref_protein_dic=ref_protein_dic,
            features_dic=features_dic,
            features_format=features_format,
            pcs_num_dic=pcs_num_dic,
            default_pcs_num=default_pcs_num,
            label_key=label_key,
            exclude_regions=exclude_regions,
            exclude_mode=exclude_mode,
            scale_embedding=scale_embedding,
            random_state=random_state,
            align_by_obs_names=align_by_obs_names,
            print_results=print_results,
        )

        ari_df_dic[method] = ari_df
        average_ari_dic[method] = average_ari

    summary_df = pd.DataFrame(
        {
            "dim_reduction_method": list(average_ari_dic.keys()),
            "average_ari": list(average_ari_dic.values()),
        }
    )

    summary_df = summary_df.sort_values(
        "average_ari",
        ascending=False,
    ).reset_index(drop=True)

    best_method = summary_df.loc[0, "dim_reduction_method"]

    if print_results:
        print("\n============================================================")
        print("Dimension reduction method selection summary")
        print("============================================================")
        print(summary_df)
        print(f"\nBest dimension reduction method: {best_method}")

    result = {
        "best_method": best_method,
        "summary_df": summary_df,
        "ari_df_dic": ari_df_dic,
        "average_ari_dic": average_ari_dic,
        "features_format": features_format,
        "align_by_obs_names": align_by_obs_names,
    }

    return result

