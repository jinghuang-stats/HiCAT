import re
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from scipy.sparse import issparse
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import List, Tuple, Union, Any


def rank_genes_groups(
	input_adata, 
	target, 
	label_key, 
	non_target="rest", 
	two_sides: bool = False, 
	logged: bool = True, 
	verbose: bool = False):
    """
    Rank marker genes for a target group against either all remaining groups
    or a user-specified set of non-target groups.

    This function wraps `scanpy.tl.rank_genes_groups` and returns a tidy
    pandas DataFrame containing ranked genes, adjusted p-values, detection
    fractions, mean expression values, and fold-change-like summaries.

    Parameters
    ----------
    input_adata : anndata.AnnData
        Input AnnData object. The expression matrix `input_adata.X` is used
        for differential expression and summary statistics.

    target : str, int, or category-like
        Label of the target group in `input_adata.obs[label_key]`.

    label_key : str
        Column name in `input_adata.obs` containing group labels.

    non_target : "rest" or list-like, default="rest"
        Defines the comparison group.

        - If `"rest"`, the target group is compared against all other cells.
        - If list-like, the target group is compared only against cells whose
          labels are included in `non_target`.

    two_sides : bool, default=False
        If False, return only genes enriched in the target group relative to
        the comparison group.

        If True, return two DataFrames:

        - `df_target`: genes ranked for the target group.
        - `df_rest`: genes ranked for the non-target/rest group.

    logged : bool, default=True
        Whether `input_adata.X` is assumed to contain log-transformed expression.

        If True, fold change is computed as:

            exp(mean_log_expression_target - mean_log_expression_rest)

        This is not necessarily identical to the ratio of arithmetic means on
        the original raw expression scale, especially if `X` contains log1p-
        normalized values.

        If False, fold change is computed as:

            mean_expression_target / mean_expression_rest

    Returns
    -------
    pandas.DataFrame
        If `two_sides=False`, returns one DataFrame with marker genes for
        the target group.

    tuple of pandas.DataFrame
        If `two_sides=True`, returns `(df_target, df_rest)`.

    Notes
    -----
    The returned columns include:

    - `genes`: ranked gene names.
    - `in_group_fraction`: fraction of target cells with expression > 0.
    - `out_group_fraction`: fraction of comparison cells with expression > 0.
    - `in_out_group_ratio`: ratio of detection fractions.
    - `in_group_mean_exp`: mean expression in target cells.
    - `out_group_mean_exp`: mean expression in comparison cells.
    - `fold_change`: fold-change-like expression ratio.
    - `pvals_adj`: adjusted p-values from Scanpy.

    Potential memory note:
    This function converts the selected expression matrix into a dense
    pandas DataFrame. For very large AnnData objects, this may require
    substantial memory.
    """

    # --------------------------
    # Input validation
    # --------------------------
    if label_key not in input_adata.obs.columns:
        raise ValueError(f"`label_key='{label_key}'` is not found in `input_adata.obs`.")

    labels = input_adata.obs[label_key]

    if target not in set(labels):
        raise ValueError(
            f"`target={target}` is not found in `input_adata.obs['{label_key}']`."
        )

    adata = input_adata.copy()

    # Use an internal column name to avoid overwriting user data.
    group_key = "_rank_genes_binary_target"

    # --------------------------
    # Define target vs comparison cells
    # --------------------------
    if non_target == "rest":
        adata.obs[group_key] = (adata.obs[label_key] == target).astype(int).astype(str)
        print("Data contains:", set(adata.obs[label_key]))

    else:
    	if isinstance(non_target, str):
    		non_target_labels = [non_target]
    	else:
        	non_target_labels = list(non_target)

        keep_labels = non_target_labels + [target]
        adata = adata[adata.obs[label_key].isin(keep_labels)].copy()

        adata.obs[group_key] = (adata.obs[label_key] == target).astype(int).astype(str)

        if verbose:
        	print("Data contains:", set(adata.obs[label_key]))

    # Make sure both groups are present.
    group_counts = adata.obs[group_key].value_counts()

    if "1" not in group_counts or "0" not in group_counts:
        raise ValueError(
            "Both target and comparison groups must contain at least one cell."
        )

    # --------------------------
    # Run differential expression
    # --------------------------
    sc.tl.rank_genes_groups(adata, groupby=group_key, reference="rest", n_genes=adata.shape[1], method='wilcoxon')

    def _build_result_df(group):
        """
        Build summary DataFrame for one ranked group.

        group='1' means target vs rest.
        group='0' means rest vs target.
        """

        # Get Scanpy DE results safely.
        de_df = sc.get.rank_genes_groups_df(adata, group=group)

        genes = de_df["names"].tolist()
        pvals_adj = de_df["pvals_adj"].tolist()

        # Convert expression matrix to dense DataFrame.
        if issparse(adata.X):
            expr = pd.DataFrame(
                adata.X.toarray(),
                index=adata.obs[group_key].tolist(),
                columns=adata.var_names,
            )
        else:
            expr = pd.DataFrame(
                adata.X,
                index=adata.obs[group_key].tolist(),
                columns=adata.var_names,
            )

        # Keep the genes in Scanpy-ranked order.
        expr = expr.loc[:, genes]

        # Mean expression by binary group.
        mean_expr = expr.groupby(level=0).mean()

        # Fraction of cells with nonzero expression.
        detected = expr > 0
        fraction_expr = detected.groupby(level=0).sum() / detected.groupby(level=0).count()

        # For group='1':
        #   in_group = target, out_group = rest
        #
        # For group='0':
        #   in_group = rest, out_group = target
        in_group = group
        out_group = "0" if group == "1" else "1"

        in_mean = mean_expr.loc[in_group]
        out_mean = mean_expr.loc[out_group]

        in_fraction = fraction_expr.loc[in_group]
        out_fraction = fraction_expr.loc[out_group]

        # log1p transformation
		if logged:
		    fold_change = np.expm1(in_mean.values) / (np.expm1(out_mean.values) + 1e-9)
		else:
		    fold_change = in_mean.values / (out_mean.values + 1e-9)

        result_df = pd.DataFrame(
            {
                "genes": genes,
                "in_group_fraction": in_fraction.tolist(),
                "out_group_fraction": out_fraction.tolist(),
                "in_out_group_ratio": (in_fraction / (out_fraction + 1e-9)).tolist(),
                "in_group_mean_exp": in_mean.tolist(),
                "out_group_mean_exp": out_mean.tolist(),
                "fold_change": fold_change.tolist(),
                "pvals_adj": pvals_adj,
            }
        )

        return result_df

    # Target group markers.
    df_target = _build_result_df(group="1")

    if not two_sides:
        return df_target

    # Rest/non-target group markers.
    df_rest = _build_result_df(group="0")

    return df_target, df_rest


def filter_ranked_genes(
    df,
    pvals_adj: float = 0.05,
    min_in_out_group_ratio: float = 1.0,
    min_in_group_fraction: float = 0.0,
    min_fold_change: float = 1.15,
    gene_num: int = 10,
    sort_by: str = "fold_change",
    ascending: bool = False,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Filter ranked genes based on statistical and expression criteria.

    Parameters
    ----------
    df
        Output dataframe from rank_genes_groups.

    Returns
    -------
    genes
        Selected genes after filtering.

    df_filtered
    	filtered dataframe ranked by fold_change
    """

    required_cols = [
        "genes",
        "pvals_adj",
        "in_out_group_ratio",
        "in_group_fraction",
        "fold_change",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(
            f"rank_genes_groups output is missing required columns: {missing_cols}"
        )

    df_filtered = df[
        (df["pvals_adj"] <= pvals_adj)
        & (df["in_out_group_ratio"] >= min_in_out_group_ratio)
        & (df["in_group_fraction"] >= min_in_group_fraction)
        & (df["fold_change"] >= min_fold_change)
    ].copy()

    df_filtered = df_filtered.sort_values(by=sort_by, ascending=ascending)

    genes = df_filtered["genes"].tolist()
    genes = genes[: min(gene_num, len(genes))]

    return genes, df_filtered


def get_region_genes(input_adata, region, label_key, gene_num=10, min_fold_change=1.15, min_in_out_group_ratio=1.0, min_in_group_fraction=0.5, pvals_adj=0.05, print_results=True):
    """
    Rank and filter genes for one target region, then store selected genes.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.
    region : str
        Target region / label to compare against the rest.
    label_key : str
        Column name in test_adata.obs containing region labels.
    gene_num : int
        Maximum number of genes to return.
    min_fold_change : float
        Minimum fold-change threshold.
    min_in_out_group_ratio : float
        Minimum in/out group ratio threshold.
    min_in_group_fraction : float
        Minimum fraction of cells/spots expressing the gene in the target group.
    pvals_adj : float
        Maximum adjusted p-value threshold.
    print_results : bool
        Whether to print intermediate ranking/filtering results.

    Returns
    -------
    region_genes : list
        Selected filtered genes for the region.
    df1_filtered : pandas.DataFrame
        Filtered marker gene dataframe.
    """
    df1 = rank_genes_groups(input_adata=input_adata, target=region, label_key=label_key, non_target="rest", two_sides=False, logged=True)

    if print_results:
    	print(f"\n----------- Region: {region} -----------")
        
        print("Sorted by in_out_group_ratio:")
        print(df1.sort_values(by="in_out_group_ratio", ascending=False))

        print("Sorted by fold_change:")
        print(df1.sort_values(by="fold_change", ascending=False))

    region_genes, df1_filtered = filter_ranked_genes(
        df=df1,
        pvals_adj=pvals_adj,
        min_in_out_group_ratio=min_in_out_group_ratio,
        min_in_group_fraction=min_in_group_fraction,
        min_fold_change=min_fold_change,
        gene_num=gene_num,
    	)

    return region_genes, df1_filtered


def select_region_markers_across_samples(
    ref_adata_dic,
    label_key="label",
    gene_num=10,
    min_fold_change=1.15,
    min_in_out_group_ratio=1.0,
    min_in_group_fraction=0.5,
    pvals_adj=0.05,
    exclude_labels=("nan", "unknown"),
    exclude_mode="contains",
    print_results=True
):
    """
    Select region-specific marker genes across multiple samples.

    This function loops through each sample and each tissue region,
    then calls get_region_genes() to identify region-specific genes.

    Parameters
    ----------
    ref_adata_dic : dict
        Dictionary of sample-specific AnnData objects.
        Example:
        {
            "sample1": adata1,
            "sample2": adata2
        }

    label_key : str
        Column in adata.obs containing tissue region labels.

    gene_num : int
        Maximum number of marker genes to return per region.

    min_fold_change : float
        Minimum fold-change threshold.

    min_in_out_group_ratio : float
        Minimum in/out group ratio threshold.

    min_in_group_fraction : float
        Minimum fraction of spots/cells expressing the gene in the target region.

    pvals_adj : float
        Maximum adjusted p-value threshold.

    exclude_labels : tuple
        Labels to exclude from marker gene selection.

    exclude_mode: {"contains", "exact"}
    	Exclusion mode.

    print_results : bool
        Whether to print intermediate results.

    Returns
    -------
    d_g_all : dict
        Nested dictionary storing selected genes for each sample and region.
        Example:
        {
            "sample1": {
                "region1": ["geneA", "geneB"],
                "region2": ["geneC", "geneD"]
            },
            "sample2": {
                "region1": ["geneA", "geneE"]
            }
        }

    gene_list_all : dict
        Dictionary storing the union of selected marker genes for each sample.
        Example:
        {
            "sample1": ["geneA", "geneB", "geneC", "geneD"],
            "sample2": ["geneA", "geneE"]
        }

    """

    d_g_all = {}
    gene_list_all = {}

    for sample_name, adata_sample in ref_adata_dic.items():

        if print_results:
            print(f"\n================== Sample: {sample_name} ==================")

        d_g = {}
        df_filtered_dic = {}

        # Get tissue regions for this sample
        tissue_regions = adata_sample.obs[label_key].value_counts().index.tolist()

        valid_regions = []

        # Remove nan / unknown labels safely
        for region in tissue_regions:
        	region_str = str(region).lower()

        	if exclude_mode == "contains":
        		is_excluded = any(exclude_label in region_str for exclude_label in exclude_labels)
        	elif exclude_mode == "exact":
        		is_excluded = region_str in exclude_labels
        	else:
        		raise ValueError("exclude_mode must be either 'contains' or 'exact'.")

        	if not is_excluded:
        		valid_regions.append(region)

        valid_regions = sorted(valid_regions)

        if print_results:
            print(f"Included valid tissue regions: {valid_regions}")

        # Select marker genes for each region
        for region in valid_regions:

            region_genes, _ = get_region_genes(
                input_adata=adata_sample,
                region=region,
                label_key=label_key,
                gene_num=gene_num,
                min_fold_change=min_fold_change,
                min_in_out_group_ratio=min_in_out_group_ratio,
                min_in_group_fraction=min_in_group_fraction,
                pvals_adj=pvals_adj,
                print_results=print_results
            )

            d_g[region] = region_genes

        # Store sample-level region marker genes
        d_g_all[sample_name] = d_g

        # Get union of selected genes across all regions in this sample
        gene_list = sorted(
            set(gene for genes in d_g.values() for gene in genes)
        )

        gene_list_all[sample_name] = gene_list

        if print_results:
            print(f"\nSelected marker genes for {sample_name}:")
            print(d_g)
            print(f"Number of unique selected genes: {len(gene_list)}")

    return d_g_all, gene_list_all


def kmeans_clustering(features_matrix, n_clusters=5, random_state=0, kmeans_key="kmeans_clusters"):
    """Run KMeans clustering with basic validity checks."""
    features_matrix = features_matrix.toarray() if issparse(features_matrix) else np.asarray(features_matrix)

    if features_matrix.shape[0] < 2:
        raise ValueError("KMeans requires at least 2 observations.")
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2.")
    if n_clusters >= features_matrix.shape[0]:
        raise ValueError("n_clusters must be smaller than the number of observations.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    y_pred = kmeans.fit_predict(features_matrix)

    print("========== KMeans Clustering Results ==========")
    print(pd.Series(y_pred).value_counts())

    return y_pred


def leiden_clustering(
    features_matrix,
    resolution=0.5,
    n_neighbors=10,
    random_state=0,
    leiden_key="leiden_clusters",
    return_info=False,
):
    """
    Run Leiden clustering on a feature matrix.

    Parameters
    ----------
    features_matrix : array-like or sparse matrix
        Input feature matrix with shape `(n_obs, n_features)`.

    resolution : float, default=0.5
        Leiden clustering resolution. Larger values usually generate more clusters.

    n_neighbors : int, default=10
        Number of nearest neighbors used to construct the neighborhood graph.
        The actual value is adjusted to `min(n_neighbors, n_obs - 1)`.

    random_state : int, default=0
        Random seed used for neighbor graph construction and Leiden clustering.

    leiden_key : str, default="leiden_clusters"
        Column name used internally to store Leiden cluster labels.

    return_info : bool, default=False
        If True, return both cluster labels and a dictionary containing
        clustering parameters.

    Returns
    -------
    y_pred : numpy.ndarray
        One-dimensional array of Leiden cluster labels.

    cluster_info : dict, optional
        Returned only when `return_info=True`.
        Contains clustering method, resolution, requested n_neighbors,
        adjusted n_neighbors, and random_state.
    """

    features_matrix = (
        features_matrix.toarray()
        if issparse(features_matrix)
        else np.asarray(features_matrix)
    )

    resolution = float(resolution)
    n_neighbors = int(n_neighbors)
    random_state = int(random_state)

    if features_matrix.shape[0] < 2:
        raise ValueError("Leiden clustering requires at least 2 observations.")

    n_neighbors_used = min(n_neighbors, features_matrix.shape[0] - 1)

    if n_neighbors_used < 1:
        raise ValueError("n_neighbors must be at least 1 after adjustment.")

    tmp = ad.AnnData(features_matrix)

    sc.pp.neighbors(
        tmp,
        n_neighbors=n_neighbors_used,
        random_state=random_state,
    )

    sc.tl.leiden(
        tmp,
        resolution=resolution,
        key_added=leiden_key,
        random_state=random_state,
    )

    y_pred = tmp.obs[leiden_key].astype(int).to_numpy()

    print("========== Leiden Clustering Results ==========")
    print(pd.Series(y_pred).value_counts())

    if return_info:
        cluster_info = {
            "clustering_method": "leiden",
            "resolution": resolution,
            "n_neighbors": n_neighbors,
            "n_neighbors_used": n_neighbors_used,
            "random_state": random_state,
        }

        return y_pred, cluster_info

    return y_pred


def compute_pca_embedding(input_adata, pcs_num=30, random_state=0, sample_name=None):
    """
    Compute PCA embeddings from AnnData.X.

    This function safely adjusts the number of principal components based on
    the dimensions of the input AnnData object, converts sparse matrices to
    dense arrays when needed, and returns the PCA-transformed feature matrix.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object. PCA is computed from `input_adata.X`.

    pcs_num : int, default=30
        Requested number of principal components.

    random_state : int, default=0
        Random seed used for PCA.

    sample_name : str or None, default=None
        Optional sample or tissue-section name used only for clearer error messages.

    Returns
    -------
    gene_pcs : numpy.ndarray
        PCA-transformed matrix with shape `(n_obs, pcs_num_use)`.

    Raises
    ------
    ValueError
        If PCA cannot be performed because the adjusted number of components is < 1.
    """

    section_msg = f" for {sample_name}" if sample_name is not None else ""

    pcs_num_use = min(
        pcs_num,
        input_adata.shape[0] - 1,
        input_adata.shape[1]
    )

    if pcs_num_use < 1:
        raise ValueError(
            f"Cannot perform PCA{section_msg}; adjusted pcs_num is < 1."
        )
    else:
    	print(f"Actual number of principal components used after adjustment: {pcs_num_use}")

    pca = PCA(n_components=pcs_num_use, random_state=random_state)
    X = input_adata.X.toarray() if issparse(input_adata.X) else np.asarray(input_adata.X)
    gene_pcs = pca.fit_transform(X)

    return gene_pcs


def get_gene_vector(input_adata, gene):
    """
    Return one gene expression vector from an AnnData object as a 1D NumPy array.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.
    gene : str
        Gene name to extract.

    Returns
    -------
    gene_vector : np.ndarray
        1D NumPy array of gene expression values across spots/cells.
    """
    if gene not in input_adata.var_names:
        raise KeyError(f"Gene {gene!r} is not present in input_adata.var_names.")

    X_gene = input_adata[:, gene].X

    if issparse(X_gene):
        gene_vector = X_gene.toarray().ravel()
    else:
        gene_vector = np.asarray(X_gene).ravel()

    return gene_vector


def compute_modality_embedding(
    input_adata,
    dim_reduction_method="pca",
    selected_features=None,
    pcs_num=30,
    scale_embedding=True,
    random_state=0,
    sample_name=None,
    modality_name=None,
):
    """
    Compute reduced embedding for one modality.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object for one modality.

    dim_reduction_method : {"pca", "selected_features"}, default="pca"
        Dimension reduction method.

    selected_features : list or None
        Selected feature names. Required when
        dim_reduction_method="selected_features".

    pcs_num : int, default=30
        Number of principal components when dim_reduction_method="pca".

    scale_embedding : bool, default=True
        Whether to standardize the reduced embedding.

        This is recommended before concatenating different modalities.

    random_state : int, default=0
        Random seed.

    sample_name : str or None
        Optional sample name.

    modality_name : str or None
        Optional modality name.

    Returns
    -------
    embedding : numpy.ndarray
        Reduced modality embedding.
    """

    if dim_reduction_method == "pca":
        embedding = compute_pca_embedding(
            input_adata=input_adata,
            pcs_num=pcs_num,
            random_state=random_state,
            sample_name=sample_name,
        )

    elif dim_reduction_method == "selected_features":
        embedding = compute_selected_feature_embedding(
            input_adata=input_adata,
            selected_features=selected_features,
            sample_name=sample_name,
            modality_name=modality_name,
        )

    else:
        raise ValueError(
            "dim_reduction_method must be either 'pca' or 'selected_features'."
        )

    if scale_embedding:
        embedding = StandardScaler().fit_transform(embedding)

    return embedding


def get_ref_modality_adata(
    ref_section,
    modality,
    ref_gene_dic=None,
    ref_image_dic=None,
    ref_protein_dic=None,
):
    """
    Get the reference AnnData object for one section and one modality.

    Parameters
    ----------
    ref_section : str
        Reference section name.

    modality : {"Gene", "Image", "Protein"}
        Modality to retrieve.

    ref_gene_dic : dict or None, optional
        Dictionary of gene-based reference AnnData objects.

    ref_image_dic : dict or None, optional
        Dictionary of image-based reference AnnData objects.

    ref_protein_dic : dict or None, optional
        Dictionary of protein-based reference AnnData objects.

    Returns
    -------
    modality_adata : AnnData
        AnnData object for the requested reference section and modality.

    Raises
    ------
    ValueError
        If the requested modality is unsupported or the corresponding
        modality dictionary is None.

    KeyError
        If ref_section is not found in the corresponding modality dictionary.
    """

    if modality == "Gene":
        modality_dic = ref_gene_dic

    elif modality == "Image":
        modality_dic = ref_image_dic

    elif modality == "Protein":
        modality_dic = ref_protein_dic

    else:
        raise ValueError(
            f"Unsupported modality: {modality}. "
            "Expected one of {'Gene', 'Image', 'Protein'}."
        )

    if modality_dic is None:
        raise ValueError(
            f"{modality} modality was requested, but its dictionary is None."
        )

    if ref_section not in modality_dic:
        raise KeyError(
            f"ref_section='{ref_section}' is not found in the "
            f"{modality} modality dictionary."
        )

    modality_adata = modality_dic[ref_section]

    return modality_adata


def get_valid_label_mask(
    labels,
    exclude_regions=("nan", "unknown"),
    exclude_mode="contains",
):
    """
    Construct valid label mask by excluding regions.

    Parameters
    ----------
    labels : pandas.Series
        Label vector.

    exclude_regions : tuple or list
        Region labels or substrings to exclude.

    exclude_mode : {"contains", "exact"}
        Whether to exclude labels by exact matching or substring matching.

    Returns
    -------
    valid_mask : pandas.Series
        Boolean mask indicating valid labels.
    """

    labels = labels.astype(str)

    if exclude_regions is None or len(exclude_regions) == 0:
        return pd.Series(True, index=labels.index)

    if exclude_mode == "exact":
        valid_mask = ~labels.isin(exclude_regions)

    elif exclude_mode == "contains":
        pattern = "|".join(
            re.escape(str(region))
            for region in exclude_regions
        )
        valid_mask = ~labels.str.contains(
            pattern,
            case=False,
            na=False,
        )

    else:
        raise ValueError(
            "exclude_mode must be either 'exact' or 'contains'."
        )

    return valid_mask


def sudo_to_spot_annotation(
    spot_obs,
    sudo_obs,
    num_nbs,
    spot_x_key,
    spot_y_key,
    sudo_x_key,
    sudo_y_key,
    annotation_key,
    small_region_adjustment=False,
    small_region_thres=0.25,
    dominant_region_thres=0.5,
    novel_label="novel_cluster",
    unknown_label="nan",
    neighbor_mode="knn",
    copy=True,
    print_results=True,
):
    """
    Transfer annotations from sudo-level observations to spot-level observations.

    For each spot, nearby sudo observations are identified. The spot annotation is
    assigned based on the majority annotation among neighboring sudo observations.

    Optionally, this function can adjust for sparse/small regions. If the most common
    local label is a large region but the second most common label is a globally small
    region with sufficient local proportion, the small region can be selected instead.

    Parameters
    ----------
    spot_obs : pd.DataFrame
        Spot-level observation dataframe.
    sudo_obs : pd.DataFrame
        Sudo-level observation dataframe.
    num_nbs : int
        Number of nearest sudo observations used for annotation if neighbor_mode="knn".
        If neighbor_mode="radius_quantile", this controls the global distance quantile.
    spot_x_col, spot_y_key : str
        Column names for spot-level x/y coordinates.
    sudo_x_col, sudo_y_key : str
        Column names for sudo-level x/y coordinates.
    annotation_key : str
        Column name of sudo-level annotations to transfer.
    small_region_adjustment : bool, default=False
        Whether to apply sparse/small-region adjustment.
    small_region_thres : float, default=0.25
        Threshold used to define globally small regions and local small-region support.
    dominant_region_thres : float, default=0.5
        Apply small-region adjustment only if one region dominates the sudo data.
    novel_label : str, default="novel_cluster"
        Label to skip when assigning the final annotation if another label is available.
    unknown_label : str, default="nan"
        Initial annotation for spots without nearby sudo observations.
    neighbor_mode : {"knn", "radius_quantile"}, default="knn"
        - "knn": use exactly num_nbs nearest sudo observations for each spot.
        - "radius_quantile": original global distance-threshold logic.
    copy : bool, default=True
        Whether to copy spot_obs before modification.
    print_results : bool, default=True
        Whether to print annotation summaries.

    Returns
    -------
    spot_obs : pd.DataFrame
        Updated spot-level observation dataframe.
    annotation_key : str
        Name of the transferred annotation column.
    """

    if copy:
        spot_obs = spot_obs.copy()

    required_spot_cols = [spot_x_key, spot_y_key]
    required_sudo_cols = [sudo_x_key, sudo_y_key, annotation_key]

    missing_spot_cols = [col for col in required_spot_cols if col not in spot_obs.columns]
    missing_sudo_cols = [col for col in required_sudo_cols if col not in sudo_obs.columns]

    if len(missing_spot_cols) > 0:
        raise KeyError(f"Missing columns in spot_obs: {missing_spot_cols}")

    if len(missing_sudo_cols) > 0:
        raise KeyError(f"Missing columns in sudo_obs: {missing_sudo_cols}")

    if num_nbs <= 0:
        raise ValueError("num_nbs must be a positive integer.")

    # ------------------------------------------------------------
    # Obtain spatial coordinates
    # ------------------------------------------------------------
    spot_coords = spot_obs[[spot_x_key, spot_y_key]].to_numpy()
    sudo_coords = sudo_obs[[sudo_x_key, sudo_y_key]].to_numpy()

    # ------------------------------------------------------------
    # Identify tissue regions and initialize probability columns
    # ------------------------------------------------------------
    tissue_sections = (
        pd.Series(sudo_obs[annotation_key])
        .dropna()
        .value_counts()
        .index
        .tolist()
    )

    sections_prob = [section + "_prob" for section in tissue_sections]

    for col in sections_prob:
        spot_obs[col] = 0.0

    spot_obs[annotation_key] = unknown_label

    # ------------------------------------------------------------
    # Identify globally small regions, if requested
    # ------------------------------------------------------------
    tissue_prop = pd.Series(sudo_obs[annotation_key]).value_counts(normalize=True)

    if small_region_adjustment and tissue_prop.max() >= dominant_region_thres:
        small_prop_regions = tissue_prop[tissue_prop < small_region_thres].index.tolist()
        small_prop_regions = [
            region for region in small_prop_regions
            if region != novel_label
        ]
    else:
        small_prop_regions = []

    if print_results and small_region_adjustment:
        print(
            f"Based on the proportion threshold of {small_region_thres}, "
            f"small regions include: {', '.join(small_prop_regions)}"
        )

    # ------------------------------------------------------------
    # Find sudo neighbors for each spot
    # ------------------------------------------------------------
    if neighbor_mode == "knn":
        tree = cKDTree(sudo_coords)

        k = min(num_nbs, sudo_obs.shape[0])
        _, neighbor_indices = tree.query(spot_coords, k=k)

        if k == 1:
            neighbor_indices = neighbor_indices[:, None]

    elif neighbor_mode == "radius_quantile":

        dists = cdist(spot_coords, sudo_coords, metric="euclidean")
        dists_threshold = np.quantile(dists.flatten(), num_nbs / sudo_obs.shape[0])
        neighbor_indices = [
            np.where(dists[i, :] < dists_threshold)[0]
            for i in range(spot_obs.shape[0])
        ]

    else:
        raise ValueError("neighbor_mode must be either 'knn' or 'radius_quantile'.")

    # ------------------------------------------------------------
    # Infer spot-level annotations
    # ------------------------------------------------------------
    for i in range(spot_obs.shape[0]):

        if neighbor_mode == "knn":
            sudo_indices = neighbor_indices[i]
        else:
            sudo_indices = neighbor_indices[i]

        if len(sudo_indices) == 0:
            continue

        pred_tmp = sudo_obs[annotation_key].iloc[sudo_indices]
        pred_labels_prop = pred_tmp.value_counts(normalize=True)

        pred_labels = pred_labels_prop.index.tolist()

        # Store local annotation probabilities
        for label in pred_labels:
            prob_col = label + "_prob"
            if prob_col in spot_obs.columns:
                spot_obs.loc[spot_obs.index[i], prob_col] = pred_labels_prop[label]

        # Case 1: skip novel_cluster if another candidate exists
        if pred_labels[0] == novel_label and len(pred_labels) > 1:
            final_label = pred_labels[1]

        # Case 2: optionally adjust for sparse/small regions
        elif (
            small_region_adjustment
            and len(pred_labels) > 1
            and pred_labels[0] not in small_prop_regions
            and pred_labels[1] in small_prop_regions
            and pred_labels_prop[pred_labels[1]] > small_region_thres
        ):
            final_label = pred_labels[1]

            if print_results:
                print("************ small region proportion adjustment ************")
                print(
                    f"{final_label} with a local proportion of "
                    f"{round(pred_labels_prop[final_label], 2)}"
                )

        # Case 3: default majority vote
        else:
            final_label = pred_labels[0]

        spot_obs.loc[spot_obs.index[i], annotation_key] = final_label

    spot_obs[annotation_key] = spot_obs[annotation_key].astype("category")

    if print_results:
        print("======================= Tissue region proportions in spot data =======================")
        print(pd.Series(spot_obs[annotation_key]).value_counts(normalize=True))

        print("======================= Tissue region proportions in sudo data =======================")
        print(pd.Series(sudo_obs[annotation_key]).value_counts(normalize=True))

    return spot_obs, annotation_key







