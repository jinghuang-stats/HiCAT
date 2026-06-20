import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import issparse

# 1. rank_genes_groups
# 2. select_genes: based on filtering parameters: pvals_adj | min_in_out_group_ratio | min_in_group_fraction | min_fold_change | gene_num
# 3. 

# untestified
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
        print("Sorted by in_out_group_ratio:")
        print(df1.sort_values(by="in_out_group_ratio", ascending=False))

        print("Sorted by fold_change:")
        print(df1.sort_values(by="fold_change", ascending=False))

    df1_filtered = df1[
        (df1["pvals_adj"] <= pvals_adj)
        & (df1["fold_change"] >= min_fold_change)
        & (df1["in_out_group_ratio"] >= min_in_out_group_ratio)
        & (df1["in_group_fraction"] >= min_in_group_fraction)
    ]

    df1_filtered = df1_filtered.sort_values(by="fold_change", ascending=False)

    if print_results:
        print(f"After applying filtering parameters -> {df1_filtered.shape[0]} genes")

    region_genes = df1_filtered["genes"].tolist()[: min(df1_filtered.shape[0], gene_num)]

    return region_genes, df1_filtered


# untestified
def rank_genes_groups(input_adata, target, label_key, non_target="rest", two_sides=False, logged=True):
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
        non_target_labels = list(non_target)

        keep_labels = non_target_labels + [target]
        adata = adata[adata.obs[label_key].isin(keep_labels)].copy()

        adata.obs[group_key] = (adata.obs[label_key] == target).astype(int).astype(str)
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

        if logged:
            fold_change = np.exp(in_mean.values - out_mean.values)
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



