import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import issparse

# Local package imports
from .preprocess import *
from .util import *


# ============================================================
# 1. Reference section weights
# ============================================================
def get_valid_regions(
    input_adata,
    label_key="label",
    min_prop=0.05,
    exclude_labels=("nan", "unknown"),
    exclude_mode="contains",
):
    """
    Identify valid tissue regions in one AnnData object.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.
    label_key : str
        Region annotation column in input_adata.obs.
    min_prop : float
        Minimum region proportion required.
    exclude_labels : tuple
        Labels to exclude.
    exclude_mode : {"contains", "exact"}
        Whether to exclude labels by substring matching or exact matching.

    Returns
    -------
    filtered_regions : list
        Valid tissue regions.
    """

    if label_key not in input_adata.obs.columns:
        raise KeyError(f"{label_key!r} is not in input_adata.obs.")

    region_props = input_adata.obs[label_key].value_counts(normalize=True)

    filtered_regions = []

    for region, prop in region_props.items():
        region_str = str(region).lower()

        if exclude_mode == "contains":
            is_excluded = any(exclude_label in region_str for exclude_label in exclude_labels)
        elif exclude_mode == "exact":
            is_excluded = region_str in exclude_labels
        else:
            raise ValueError("exclude_mode must be either 'contains' or 'exact'.")

        if (not is_excluded) and (prop > min_prop):
            filtered_regions.append(region)

    return filtered_regions


def infer_reference_weights(
    ref_adata_dic,
    ref_section_list=None,
    label_key="label",
    min_prop=0.05,
    exclude_labels=("nan", "unknown"),
    exclude_mode="contains",
    weight_key="weight",
    print_results=True,
):
    """
    Infer reference-section weights based on tissue-region diversity.

    Weight formula:
        weight = sqrt(number of valid regions in section / maximum number of valid regions)

    This generalizes the weighting logic used in the HER2+ BC and Brain Visium scripts.
    """

    if ref_section_list is None:
        ref_section_list = list(ref_adata_dic.keys())

    region_dic = {}
    max_region_num = 0

    for section in ref_section_list:
        if print_results:
            print(f"------------------- {section} -------------------")

        filtered_regions = get_valid_regions(
            input_adata=ref_adata_dic[section],
            label_key=label_key,
            min_prop=min_prop,
            exclude_labels=exclude_labels,
            exclude_mode=exclude_mode,
        )

        region_dic[section] = filtered_regions
        max_region_num = max(max_region_num, len(filtered_regions))

        if print_results:
            print("Filtered tissue regions:")
            print(filtered_regions)
            print(f"Number of regions: {len(filtered_regions)}")

    weights = pd.DataFrame(
        np.zeros((len(ref_section_list), 1)),
        index=ref_section_list,
        columns=[weight_key],
    )

    for section in ref_section_list:
        if max_region_num == 0:
            region_weight = 1.0
        else:
            region_weight = math.sqrt(len(region_dic[section]) / max_region_num)

        weights.loc[section, weight_key] = region_weight

    if print_results:
        print("======================= Reference weights =======================")
        print(weights)

    return weights, region_dic


# ============================================================
# 2. Query-reference similarity
# ============================================================
def compute_gene_distribution_similarity(
    ref_adata,
    qry_adata,
    gene,
    method="ks",
):
    """
    Compute one-gene distribution similarity between reference and query.

    For method="ks":
        Kolmogorov-Smirnov (KS) statistic ranges from 0 to 1.
        measures the maximum difference between the two eCDFs.
        similarity = 1 - KS statistic.
        Larger value means more similar.
    """

    rv_ref = get_gene_vector(ref_adata, gene)
    rv_qry = get_gene_vector(qry_adata, gene)

    if method == "ks":
        ks_stat = stats.ks_2samp(rv_ref, rv_qry).statistic # two-sample Kolmogorov-Smirnov test
        similarity = 1 - ks_stat
    else:
        raise ValueError("Currently only method='ks' is supported.")

    return similarity


def compute_pairwise_ref_query_similarity(
    ref_adata,
    qry_adata,
    gene_list,
    qry_section,
    summary_key="average",
    method="ks"
):
    """
    Compute marker-gene similarity between one reference and one query section.

    Returns
    -------
    similarity_res : pd.DataFrame
        One-row DataFrame with gene-level similarities and average score.
    """

    available_genes = [
        gene for gene in gene_list
        if gene in ref_adata.var_names and gene in qry_adata.var_names
    ]

    similarity_res = pd.DataFrame(
        np.zeros((1, len(available_genes))),
        index=[qry_section],
        columns=available_genes,
    )

    for gene in available_genes:
        similarity_res.loc[qry_section, gene] = compute_gene_distribution_similarity(
            ref_adata=ref_adata,
            qry_adata=qry_adata,
            gene=gene,
            method=method,
        )

    if len(available_genes) > 0:
        similarity_res[summary_key] = similarity_res.loc[qry_section, available_genes].mean()
    else:
        similarity_res[summary_key] = np.nan

    return similarity_res


def compute_reference_similarity(
    ref_adata_dic,
    qry_adata_dic,
    gene_list_All,
    weights=None,
    ref_section_list=None,
    qry_section_list=None,
    weight_key="weight",
    similarity_key="KS_similarity",
    weighted_similarity_key=None,
    summary_key="average",
    method="ks",
    sort_by="weighted",
    print_results=True,
):
    """
    Compute query-reference similarity for reference selection.

    Parameters
    ----------
    ref_adata_dic : dict
        Reference AnnData dictionary.
    qry_adata_dic : dict
        Query AnnData dictionary.
    gene_list_All : dict
        Reference-section marker genes.
    weights : pd.DataFrame or None
        Reference weights from infer_reference_weights().
        If None, all references receive weight 1.
    sort_by : {"similarity", "weighted"}
        Whether to sort references by raw similarity or weighted similarity.

    Returns
    -------
    d_s_All : dict
        Nested dictionary containing gene-level similarity results.
    similarity_summary_dic : dict
        Summary DataFrame for each query section.
    ranked_refs_dic : dict
        Ranked reference sections for each query section.
    """

    if ref_section_list is None:
        ref_section_list = list(ref_adata_dic.keys())

    if qry_section_list is None:
        qry_section_list = list(qry_adata_dic.keys())

    if weighted_similarity_key is None:
        weighted_similarity_key = "weighted_" + similarity_key

    if weights is None:
        weights = pd.DataFrame(
            np.ones((len(ref_section_list), 1)),
            index=ref_section_list,
            columns=[weight_key],
        )

    d_s_All = {}
    similarity_summary_dic = {}
    ranked_refs_dic = {}

    for qry_section in qry_section_list:
        if print_results:
            print(f"==================== Query section: {qry_section} ====================")

        d_s = {}

        similarity_summary = pd.DataFrame(
            np.zeros((len(ref_section_list), 3)),
            index=ref_section_list,
            columns=[similarity_key, weight_key, weighted_similarity_key],
        )

        qry_adata_test = qry_adata_dic[qry_section]

        for ref_section in ref_section_list:
            ref_adata_test = ref_adata_dic[ref_section]
            gene_list = gene_list_All[ref_section]

            similarity_res = compute_pairwise_ref_query_similarity(
                ref_adata=ref_adata_test,
                qry_adata=qry_adata_test,
                gene_list=gene_list,
                qry_section=qry_section,
                summary_key=summary_key,
                method=method,
            )

            d_s[ref_section] = similarity_res

            raw_similarity = similarity_res.loc[qry_section, summary_key]
            ref_weight = weights.loc[ref_section, weight_key]

            similarity_summary.loc[ref_section, similarity_key] = raw_similarity
            similarity_summary.loc[ref_section, weight_key] = ref_weight
            similarity_summary.loc[ref_section, weighted_similarity_key] = (
                raw_similarity * ref_weight
            )

            if print_results:
                print(f"========== {qry_section} vs. {ref_section} ==========")
                print(similarity_res)

        d_s_All[qry_section] = d_s

        if sort_by == "similarity":
            similarity_summary = similarity_summary.sort_values(
                by=similarity_key,
                ascending=False,
            )
        elif sort_by == "weighted":
            similarity_summary = similarity_summary.sort_values(
                by=weighted_similarity_key,
                ascending=False,
            )
        else:
            raise ValueError("sort_by must be either 'similarity' or 'weighted'.")

        similarity_summary_dic[qry_section] = similarity_summary
        ranked_refs_dic[qry_section] = similarity_summary.index.tolist()

        if print_results:
            print(f"========== Similarity summary for {qry_section} ==========")
            print(similarity_summary)

    return d_s_All, similarity_summary_dic, ranked_refs_dic


# ============================================================
# 3. Select query-specific references
# ============================================================
def select_refs_for_each_query(
    similarity_summary_dic,
    selection_mode="cutoff",
    alpha=0.9,
    top_k=3,
    min_similarity_level=0.7,
    score_key=None,
    similarity_key="KS_similarity",
    weighted_similarity_key=None,
    sort_by="weighted",
    print_results=True,
):
    """
    Select reference sections for each query section based on similarity scores.

    Selection logic
    ---------------
    A reference is selected only if it satisfies BOTH:

        1. score >= min_similarity_level

    and

        2. the main selection rule:
            - if selection_mode="cutoff":
                score >= alpha * max_score
            - if selection_mode="top_k":
                reference is among the top_k ranked references

    If no reference has score >= min_similarity_level, the function reports this
    and keeps only the reference with the largest similarity score.

    Parameters
    ----------
    similarity_summary_dic : dict
        Dictionary of similarity summary DataFrames.
        Example:
            similarity_summary_dic[query_section] = DataFrame indexed by ref sections.

    selection_mode : {"cutoff", "top_k"}
        Reference selection mode.

    alpha : float, default=0.9
        Used when selection_mode="cutoff".
        The cutoff is alpha * maximum similarity score for each query.

    top_k : int, default=3
        Used when selection_mode="top_k".
        Number of top-ranked references to keep.

    min_similarity_level : float, default=0.7
        Lowest acceptable similarity level.

    score_key : str or None
        Column used for reference selection.
        If None, the function determines it from sort_by.

    similarity_key : str
        Raw similarity column name.

    weighted_similarity_key : str or None
        Weighted similarity column name.

    sort_by : {"weighted", "similarity"}
        Determines default score_key when score_key=None.

    print_results : bool
        Whether to print selected references.

    Returns
    -------
    selected_refs_dic : dict
        Dictionary mapping each query section to selected reference sections.

    selection_summary_dic : dict
        Dictionary of updated similarity summary DataFrames with selection indicators.
    """

    if weighted_similarity_key is None:
        weighted_similarity_key = "weighted_" + similarity_key

    if score_key is None:
        if sort_by == "weighted":
            score_key = weighted_similarity_key
        elif sort_by == "similarity":
            score_key = similarity_key
        else:
            raise ValueError("sort_by must be either 'weighted' or 'similarity'.")

    if selection_mode not in ["cutoff", "top_k"]:
        raise ValueError("selection_mode must be either 'cutoff' or 'top_k'.")

    if selection_mode == "cutoff":
        if alpha <= 0:
            raise ValueError("alpha must be positive when selection_mode='cutoff'.")

    if selection_mode == "top_k":
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer when selection_mode='top_k'.")

    selected_refs_dic = {}
    selection_summary_dic = {}

    for qry_section, similarity_summary in similarity_summary_dic.items():

        summary_df = similarity_summary.copy()

        if score_key not in summary_df.columns:
            raise KeyError(
                f"{score_key!r} is not in similarity_summary_dic[{qry_section!r}].columns."
            )

        # Sort references by selected score.
        summary_df = summary_df.sort_values(by=score_key, ascending=False)

        scores = summary_df[score_key]
        valid_scores = scores.dropna()

        if valid_scores.shape[0] == 0:
            raise ValueError(
                f"No valid similarity scores found for query section {qry_section!r}."
            )

        max_score = valid_scores.max()
        best_ref = valid_scores.idxmax()

        # --------------------------------------------------------
        # References passing the lowest similarity level
        # --------------------------------------------------------
        refs_above_min_level = valid_scores[
            valid_scores >= min_similarity_level
        ].index.tolist()

        # --------------------------------------------------------
        # Main selection rule
        # --------------------------------------------------------
        if selection_mode == "cutoff":
            similarity_cutoff = alpha * max_score

            refs_by_main_rule = valid_scores[
                valid_scores >= similarity_cutoff
            ].index.tolist()

        elif selection_mode == "top_k":
            similarity_cutoff = np.nan

            refs_by_main_rule = valid_scores.head(
                min(top_k, valid_scores.shape[0])
            ).index.tolist()

        # --------------------------------------------------------
        # Final selected references: intersection, not union
        # --------------------------------------------------------
        if len(refs_above_min_level) == 0:
            selected_refs = [best_ref]
            no_ref_above_min_level = True

            if print_results:
                print(
                    f"[Warning] Query {qry_section}: no reference has "
                    f"{score_key} >= {min_similarity_level}. "
                    f"Only keeping the best reference: {best_ref} "
                    f"with score {round(max_score, 4)}."
                )

        else:
            selected_refs = [
                ref for ref in refs_by_main_rule
                if ref in refs_above_min_level
            ]

            no_ref_above_min_level = False

            # This should rarely happen if top_k >= 1 or alpha <= 1,
            # but this fallback makes the function safer.
            if len(selected_refs) == 0:
                best_ref_above_min = valid_scores.loc[refs_above_min_level].idxmax()
                selected_refs = [best_ref_above_min]

                if print_results:
                    print(
                        f"[Warning] Query {qry_section}: no reference passes both "
                        f"the minimum similarity level and the main selection rule. "
                        f"Keeping the best reference above the minimum level: "
                        f"{best_ref_above_min}."
                    )

        # --------------------------------------------------------
        # Add selection indicators to summary table
        # --------------------------------------------------------
        summary_df["above_min_similarity_level"] = summary_df.index.isin(
            refs_above_min_level
        )

        summary_df["selected_by_main_rule"] = summary_df.index.isin(
            refs_by_main_rule
        )

        summary_df["selected"] = summary_df.index.isin(selected_refs)

        summary_df["selection_cutoff"] = similarity_cutoff
        summary_df["max_score"] = max_score
        summary_df["min_similarity_level"] = min_similarity_level
        summary_df["no_ref_above_min_level"] = no_ref_above_min_level

        selected_refs_dic[qry_section] = selected_refs
        selection_summary_dic[qry_section] = summary_df

        if print_results:
            print(f"==================== {qry_section} selected references ====================")
            print(f"Selection score key: {score_key}")
            print(f"Selection mode: {selection_mode}")

            if selection_mode == "cutoff":
                print(f"Alpha: {alpha}")
                print(f"Maximum score: {round(max_score, 4)}")
                print(f"Alpha cutoff: {round(similarity_cutoff, 4)}")

            if selection_mode == "top_k":
                print(f"Top k: {top_k}")

            print(f"Minimum similarity level: {min_similarity_level}")
            print(f"References above minimum level: {refs_above_min_level}")
            print(f"References selected by main rule: {refs_by_main_rule}")
            print(f"Final selected references: {selected_refs}")

    return selected_refs_dic, selection_summary_dic


def select_references_pipeline(
    ref_adata_dic,
    qry_adata_dic,
    ref_section_list=None,
    qry_section_list=None,
    label_key="label",
    low_exp_thres=0.05,
    normalize=True,
    normalization_method="min_max",
    min_region_prop=0.05,
    exclude_labels=("nan", "unknown"),
    exclude_mode="contains",
    pvals_adj=0.05,
    min_in_out_group_ratio=1.0,
    min_in_group_fraction=0.5,
    min_fold_change=1.10,
    gene_num=10,
    weight_key="weight",
    similarity_key="KS_similarity",
    summary_key="average",
    similarity_method="ks",
    sort_by="weighted",
    selection_mode="cutoff",
    alpha=0.9,
    top_k=3,
    min_similarity_level=0.7,
    selection_score_key=None,
    preprocess_ref=True,
    preprocess_qry=True,
    print_results=True,
):
    """
    General reference-selection pipeline.

    This pipeline:
        1. preprocesses reference/query AnnData objects,
        2. infers reference-section weights based on tissue-region diversity,
        3. selects reference-section marker genes,
        4. computes query-reference similarity,
        5. selects candidate reference sections for each query section.

    Returns
    -------
    results_dic : dict
        Dictionary containing:
            - ref_adata_dic
            - qry_adata_dic
            - weights
            - region_dic
            - d_g_All
            - gene_list_All
            - d_s_All
            - similarity_summary_dic
            - ranked_refs_dic
            - selected_refs_dic
            - selection_summary_dic
            - params
    """

    # --------------------------------------------------------
    # Step 0. Resolve section lists
    # --------------------------------------------------------
    if ref_section_list is None:
        ref_section_list = list(ref_adata_dic.keys())

    if qry_section_list is None:
        qry_section_list = list(qry_adata_dic.keys())

    missing_ref_sections = [
        section for section in ref_section_list
        if section not in ref_adata_dic
    ]

    missing_qry_sections = [
        section for section in qry_section_list
        if section not in qry_adata_dic
    ]

    if len(missing_ref_sections) > 0:
        raise KeyError(
            f"The following ref_section_list entries are not in ref_adata_dic: "
            f"{missing_ref_sections}"
        )

    if len(missing_qry_sections) > 0:
        raise KeyError(
            f"The following qry_section_list entries are not in qry_adata_dic: "
            f"{missing_qry_sections}"
        )

    # Keep only requested sections.
    ref_adata_dic = {
        section: ref_adata_dic[section]
        for section in ref_section_list
    }

    qry_adata_dic = {
        section: qry_adata_dic[section]
        for section in qry_section_list
    }

    # --------------------------------------------------------
    # Step 1. Preprocess reference and query AnnData objects
    # --------------------------------------------------------
    if preprocess_ref:
        ref_adata_dic = preprocess_adata_dic(
            adata_dic=ref_adata_dic,
            section_list=ref_section_list,
            low_exp_thres=low_exp_thres,
            normalize=normalize,
            normalization_method=normalization_method,
            print_results=print_results,
        )

    if preprocess_qry:
        qry_adata_dic = preprocess_adata_dic(
            adata_dic=qry_adata_dic,
            section_list=qry_section_list,
            low_exp_thres=low_exp_thres,
            normalize=normalize,
            normalization_method=normalization_method,
            print_results=print_results,
        )

    # --------------------------------------------------------
    # Step 2. Infer reference weights
    # --------------------------------------------------------
    weights, region_dic = infer_reference_weights(
        ref_adata_dic=ref_adata_dic,
        ref_section_list=ref_section_list,
        label_key=label_key,
        min_prop=min_region_prop,
        exclude_labels=exclude_labels,
        exclude_mode=exclude_mode,
        weight_key=weight_key,
        print_results=print_results,
    )

    # --------------------------------------------------------
    # Step 3. Select reference-section marker genes
    # --------------------------------------------------------
    d_g_All, gene_list_All = select_region_markers_across_samples(
        adata_dic=ref_adata_dic,
        sample_list=ref_section_list,
        label_key=label_key,
        gene_num=gene_num,
        min_fold_change=min_fold_change,
        min_in_out_group_ratio=min_in_out_group_ratio,
        min_in_group_fraction=min_in_group_fraction,
        pvals_adj=pvals_adj,
        exclude_labels=exclude_labels,
        exclude_mode=exclude_mode,
        print_results=print_results,
    )

    # --------------------------------------------------------
    # Step 4. Compute query-reference similarity
    # --------------------------------------------------------
    d_s_All, similarity_summary_dic, ranked_refs_dic = compute_reference_similarity(
        ref_adata_dic=ref_adata_dic,
        qry_adata_dic=qry_adata_dic,
        gene_list_All=gene_list_All,
        weights=weights,
        ref_section_list=ref_section_list,
        qry_section_list=qry_section_list,
        weight_key=weight_key,
        similarity_key=similarity_key,
        summary_key=summary_key,
        method=similarity_method,
        sort_by=sort_by,
        print_results=print_results,
    )

    # --------------------------------------------------------
    # Step 5. Select final references for each query
    # --------------------------------------------------------
    weighted_similarity_key = "weighted_" + similarity_key

    selected_refs_dic, selection_summary_dic = select_refs_for_each_query(
        similarity_summary_dic=similarity_summary_dic,
        selection_mode=selection_mode,
        alpha=alpha,
        top_k=top_k,
        min_similarity_level=min_similarity_level,
        score_key=selection_score_key,
        similarity_key=similarity_key,
        weighted_similarity_key=weighted_similarity_key,
        sort_by=sort_by,
        print_results=print_results,
    )

    # --------------------------------------------------------
    # Step 6. Store results
    # --------------------------------------------------------
    params = {
        "ref_section_list": ref_section_list,
        "qry_section_list": qry_section_list,
        "low_exp_thres": low_exp_thres,
        "normalize": normalize,
        "normalization_method": normalization_method,
        "min_region_prop": min_region_prop,
        "exclude_labels": exclude_labels,
        "exclude_mode": exclude_mode,
        "min_fold_change": min_fold_change,
        "min_in_out_group_ratio": min_in_out_group_ratio,
        "min_in_group_fraction": min_in_group_fraction,
        "pvals_adj": pvals_adj,
        "gene_num": gene_num,
        "weight_key": weight_key,
        "similarity_key": similarity_key,
        "summary_key": summary_key,
        "similarity_method": similarity_method,
        "sort_by": sort_by,
        "selection_mode": selection_mode,
        "alpha": alpha,
        "top_k": top_k,
        "min_similarity_level": min_similarity_level,
        "selection_score_key": selection_score_key,
        "preprocess_ref": preprocess_ref,
        "preprocess_qry": preprocess_qry,
    }

    results_dic = {
        "ref_adata_dic": ref_adata_dic,
        "qry_adata_dic": qry_adata_dic,
        "weights": weights,
        "region_dic": region_dic,
        "d_g_All": d_g_All,
        "gene_list_All": gene_list_All,
        "d_s_All": d_s_All,
        "similarity_summary_dic": similarity_summary_dic,
        "ranked_refs_dic": ranked_refs_dic,
        "selected_refs_dic": selected_refs_dic,
        "selection_summary_dic": selection_summary_dic,
        "params": params,
    }

    return results_dic



