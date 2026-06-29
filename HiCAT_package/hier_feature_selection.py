#=======================================================================
# Part 1. Select hierarchical features
#=======================================================================
def select_hier_genes(
    ref_adata_dic: Dict[str, Any],
    hier_tree,
    anchor_scenario: str,
    filtering_paras: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Select hierarchical gene sets across reference samples using a fitted HierTree object.

    This function connects tree inference results with hierarchical gene selection.

    Parameters
    ----------
    ref_adata_dic
        Dictionary of AnnData objects.

        Example:
            {
                "ref_sample1": adata1,
                "ref_sample2": adata2,
                ...
            }

    hier_tree
        A HierTree object obtained from tree inference.

        Required methods:
            hier_tree.get_split_pairs(order="root_to_leaf")
            hier_tree.get_regions(node)
            hier_tree.root_node

    anchor_scenario
        Either "nn_based" or "quantile_based".

        This controls the default filtering parameters.
        User-provided values in filtering_paras will override the defaults.

    filtering_paras
        Dictionary of filtering parameters.

        Required keys:
            label_key
            pvals_adj
            min_in_out_group_ratio
            min_in_group_fraction
            min_fold_change
            gene_num

        Optional keys:
            two_sides
            logged
            split_order
            verbose

    Returns
    -------
    results_dic
        Nested dictionary containing selected genes for each sample.

        results_dic[sample_name]["hier_genes_dic"]
            Direction-specific genes for each binary separation.

        results_dic[sample_name]["hier_genenum"]
            Number of selected genes for each parent split.

        results_dic[sample_name]["split_info"]
            Metadata for each binary split.
    """

    if anchor_scenario not in ["nn_based", "quantile_based"]:
        raise ValueError(
            "anchor_scenario must be either 'nn_based' or 'quantile_based'."
        )

    # ------------------------------------------------------------------
    # Required parameters
    # ------------------------------------------------------------------
    label_key = filtering_paras["label_key"]
    rank_genes_groups = filtering_paras["rank_genes_groups"]

    # ------------------------------------------------------------------
    # Scenario-specific defaults
    # ------------------------------------------------------------------
    if anchor_scenario == "nn_based":
        default_paras = {
            "pvals_adj": 0.05,
            "min_in_out_group_ratio": 1.0,
            "min_in_group_fraction": 0.0,
            "min_fold_change": 1.15,
            "gene_num": 10,
        }

    else:  # quantile_based
        default_paras = {
            "pvals_adj": 0.05,
            "min_in_out_group_ratio": 1.0,
            "min_in_group_fraction": 0.0,
            "min_fold_change": 1.15,
            "gene_num": 5,
        }

    # ------------------------------------------------------------------
    # User-provided parameters override defaults
    # ------------------------------------------------------------------
    pvals_adj = filtering_paras.get(
        "pvals_adj", default_paras["pvals_adj"]
    )

    min_in_out_group_ratio = filtering_paras.get(
        "min_in_out_group_ratio",
        default_paras["min_in_out_group_ratio"],
    )

    min_in_group_fraction = filtering_paras.get(
        "min_in_group_fraction",
        default_paras["min_in_group_fraction"],
    )

    min_fold_change = filtering_paras.get(
        "min_fold_change",
        default_paras["min_fold_change"],
    )

    gene_num = filtering_paras.get(
        "gene_num",
        default_paras["gene_num"],
    )

    two_sides = filtering_paras.get("two_sides", True)
    logged = filtering_paras.get("logged", True)
    verbose = filtering_paras.get("verbose", True)

    split_order = filtering_paras.get("split_order", "root_to_leaf")

    # ------------------------------------------------------------------
    # Use HierTree object to get binary split information
    # ------------------------------------------------------------------
    split_pairs = hier_tree.get_split_pairs(order=split_order)

	results_dic = {
	    "anchor_scenario": anchor_scenario,
	    "root_node": hier_tree.root_node,
	    "filtering_paras": {
	        "pvals_adj": pvals_adj,
	        "min_in_out_group_ratio": min_in_out_group_ratio,
	        "min_in_group_fraction": min_in_group_fraction,
	        "min_fold_change": min_fold_change,
	        "gene_num": gene_num,
	        "two_sides": two_sides,
	        "logged": logged,
	        "split_order": split_order,
	    },
	    "split_info": {},
	    "hier_genes_dic": {},
	    "hier_genenum": {},
	}

    # ==================================================================
    # Iterate over samples
    # ==================================================================
    for sample_name, gene_adata in ref_adata_dic.items():

        if verbose:
            print("\n" + "=" * 80)
            print(f"Selecting hierarchical genes for sample: {sample_name}")
            print(f"Anchor scenario: {anchor_scenario}")
            print(f"Root node: {hier_tree.root_node}")
            print("=" * 80)

        if label_key not in gene_adata.obs.columns:
            raise ValueError(
                f"label_key='{label_key}' is not found in "
                f"ref_adata_dic['{sample_name}'].obs."
            )

        hier_genes_dic = {}
        hier_genenum = {}
        split_info = {}

        # ==============================================================
        # Iterate over binary splits from inferred tree
        # ==============================================================
        for parent_node, child_node_1, child_node_2 in split_pairs:

            child_1_regions = hier_tree.get_regions(child_node_1)
            child_2_regions = hier_tree.get_regions(child_node_2)

            included_regions = child_1_regions + child_2_regions

            if verbose:
                print("\n" + "-" * 60)
                print(f"Parent node: {parent_node}")
                print(f"Binary split: {child_node_1} vs {child_node_2}")
                print(f"{child_node_1} regions: {child_1_regions}")
                print(f"{child_node_2} regions: {child_2_regions}")

            # ----------------------------------------------------------
            # Subset spots belonging to this binary split
            # ----------------------------------------------------------
            region_mask = gene_adata.obs[label_key].isin(included_regions)

            split_key_1 = f"{child_node_1}_vs_{child_node_2}"
            split_key_2 = f"{child_node_2}_vs_{child_node_1}"

            if parent_node not in results_dic["split_info"]:
            	results_dic["split_info"][parent_node] ={
                	"parent_node": parent_node,
                	"child_node_1": child_node_1,
                	"child_node_2": child_node_2,
                	"child_1_regions": child_1_regions,
                	"child_2_regions": child_2_regions,
                	"split_key_1": split_key_1,
                	"split_key_2": split_key_2,
            	}

            if region_mask.sum() == 0:
                if verbose:
                    print(
                        f"Skipping {parent_node}: no spots found for "
                        f"{child_node_1} or {child_node_2}."
                    )

                hier_genes_dic[f"{child_node_1}_vs_{child_node_2}"] = []
                hier_genes_dic[f"{child_node_2}_vs_{child_node_1}"] = []
                hier_genenum[parent_node] = (0, 0)

                continue

            adata_sub = gene_adata[region_mask].copy()

            # ----------------------------------------------------------
            # Define binary target label
            # 1: child_node_1 regions
            # 0: child_node_2 regions
            # ----------------------------------------------------------
            adata_sub.obs["target"] = (
                adata_sub.obs[label_key].isin(child_1_regions)
            ).astype(int)

            target_counts = adata_sub.obs["target"].value_counts().to_dict()

            if verbose:
                print(f"Target counts: {target_counts}")

            # Need both classes for binary DE
            if not ({0, 1}.issubset(set(target_counts.keys()))):
                if verbose:
                    print(
                        f"Skipping {parent_node}: only one class is present."
                    )

                hier_genes_dic[f"{child_node_1}_vs_{child_node_2}"] = []
                hier_genes_dic[f"{child_node_2}_vs_{child_node_1}"] = []
                hier_genenum[parent_node] = (0, 0)

                continue

            # ----------------------------------------------------------
            # Differential expression / gene ranking
            # ----------------------------------------------------------
            df1, df0 = rank_genes_groups(
                input_adata=adata_sub,
                target=1,
                label_key="target",
                non_target="rest",
                two_sides=two_sides,
                logged=logged,
            )

            # ----------------------------------------------------------
            # child_node_1-enriched genes
            # ----------------------------------------------------------
            child_1_genes, _ = filter_ranked_genes(
                df=df1,
                pvals_adj=pvals_adj,
                min_in_out_group_ratio=min_in_out_group_ratio,
                min_in_group_fraction=min_in_group_fraction,
                min_fold_change=min_fold_change,
                gene_num=gene_num,
            )

            # ----------------------------------------------------------
            # child_node_2-enriched genes
            # ----------------------------------------------------------
            child_2_genes, _ = filter_ranked_genes(
                df=df0,
                pvals_adj=pvals_adj,
                min_in_out_group_ratio=min_in_out_group_ratio,
                min_in_group_fraction=min_in_group_fraction,
                min_fold_change=min_fold_change,
                gene_num=gene_num,
            )

            hier_genes_dic[split_key_1] = child_1_genes
            hier_genes_dic[split_key_2] = child_2_genes

            hier_genenum[parent_node] = (
                len(child_1_genes),
                len(child_2_genes),
            )

            if verbose:
                print(f"{split_key_1}: {len(child_1_genes)} genes")
                print(f"{split_key_2}: {len(child_2_genes)} genes")

        results_dic["hier_genes_dic"][sample_name] = hier_genes_dic
        results_dic["hier_genenum"][sample_name] = hier_genenum

    return results_dic


def get_hierarchy_split_info_dic(hier_tree, order="root_to_leaf"):
    """
    Get split information for every binary split in a HierTree object.

    Each internal parent node defines one binary split:

        parent_node -> child_node_1 vs child_node_2

    Parameters
    ----------
    hier_tree : HierTree
        Fitted hierarchical tree object.

    order : {"root_to_leaf", "leaf_to_root"}, default="root_to_leaf"
        Order used to retrieve split pairs from the tree.

    Returns
    -------
    split_info_dic : dict
        Dictionary keyed by parent_node.

        Example:
            {
                "node_0": {
                    "parent_node": "node_0",
                    "child_node_1": "node_1",
                    "child_node_2": "node_2",
                    "child_1_regions": [...],
                    "child_2_regions": [...],
                    "split_key_1": "node_1_vs_node_2",
                    "split_key_2": "node_2_vs_node_1",
                },
                ...
            }
    """

    split_pairs = hier_tree.get_split_pairs(order=order)

    split_info_dic = {}

    for parent_node, child_node_1, child_node_2 in split_pairs:

        child_1_regions = hier_tree.get_regions(child_node_1)
        child_2_regions = hier_tree.get_regions(child_node_2)

        split_key_1 = f"{child_node_1}_vs_{child_node_2}"
        split_key_2 = f"{child_node_2}_vs_{child_node_1}"

        split_info_dic[parent_node] = {
            "parent_node": parent_node,
            "child_node_1": child_node_1,
            "child_node_2": child_node_2,
            "child_1_regions": child_1_regions,
            "child_2_regions": child_2_regions,
            "included_regions": child_1_regions + child_2_regions,
            "split_key_1": split_key_1,
            "split_key_2": split_key_2,
        }

    return split_info_dic


def get_split_info_by_parent_node(hier_tree, parent_node, order="root_to_leaf"):
    """
    Retrieve split information for one parent node.

    Parameters
    ----------
    hier_tree : HierTree
        Fitted hierarchical tree object.

    parent_node : str
        Internal node whose binary split should be retrieved.

    order : {"root_to_leaf", "leaf_to_root"}, default="root_to_leaf"
        Split ordering used internally.

    Returns
    -------
    split_info : dict
        Split metadata for the requested parent_node.
    """

    split_info_dic = get_hierarchy_split_info_dic(
        hier_tree=hier_tree,
        order=order,
    )

    if parent_node not in split_info_dic:
        raise ValueError(
            f"parent_node='{parent_node}' is not an internal split node. "
            f"Available parent nodes are: {list(split_info_dic.keys())}."
        )

    return split_info_dic[parent_node]


def _union_preserve_order(feature_lists, count_num=1):
    """
    Keep features that appear in at least `count_num` feature lists.

    Features are ordered by:
    1. Higher shared count first.
    2. Earlier first appearance if counts are tied.

    Parameters
    ----------
    feature_lists : list of list-like
        A list containing multiple feature lists.

    count_num : int, default=1
        Minimum number of feature lists in which a feature must appear
        to be kept. If count_num=1, this behaves like a union, but the
        returned features are ordered by shared count.

    Returns
    -------
    selected_features : list
        Features appearing in at least `count_num` lists, ordered by
        decreasing shared count.
    """

    if count_num < 1:
        raise ValueError("count_num must be at least 1.")

    feature_counts = {}
    first_seen_order = {}
    order_idx = 0

    for features in feature_lists:
        if features is None:
            continue

        # Count each feature only once within the same feature list
        seen_in_current_list = set()

        for feature in features:
            if feature in seen_in_current_list:
                continue

            seen_in_current_list.add(feature)

            if feature not in feature_counts:
                feature_counts[feature] = 0
                first_seen_order[feature] = order_idx
                order_idx += 1

            feature_counts[feature] += 1

    selected_features = [
        feature
        for feature, count in feature_counts.items()
        if count >= count_num
    ]

    selected_features = sorted(
        selected_features,
        key=lambda feature: (-feature_counts[feature], first_seen_order[feature])
    )

    return selected_features


def construct_features_dic_for_parent_split(
    hier_tree,
    parent_node,
    gene_results_dic=None,
    image_results_dic=None,
    protein_results_dic=None,
    ref_section_list=None,
    output_format="section",
    count_num=1,
    min_features=1,
    split_order="root_to_leaf",
    feature_result_key="hier_genes_dic",
    strict=False,
    verbose=True,
):
    """
    Construct features_dic for one hierarchy split specified by parent_node.

    This generalizes construct_features_dic_from_first_split().

    Parameters
    ----------
    hier_tree : HierTree
        Fitted hierarchical tree object.

    parent_node : str
        Internal node defining the binary split.

    gene_results_dic, image_results_dic, protein_results_dic : dict or None
        Hierarchical feature selection results for different modalities.

        Each result dictionary should contain:

            results_dic[feature_result_key][ref_section][split_key]

        By default:

            feature_result_key = "hier_genes_dic"

    ref_section_list : list or None, optional
        Reference sections to include. If None, sections are inferred from
        the provided result dictionaries.

    output_format : {"section", "modality"}, default="section"
        Output structure.

        If "section":

            {
                "H1": {
                    "Gene": [...],
                    "Image": [...],
                    "Protein": [...]
                },
                "G2": {
                    "Gene": [...],
                    "Image": [...],
                    "Protein": [...]
                }
            }

        If "modality":

            {
                "Gene": [...],
                "Image": [...],
                "Protein": [...]
            }

    count_num : int, default=1
        Minimum number of reference sections in which a feature must appear
        to be kept.

        This is mainly meaningful when output_format="modality", because
        features are aggregated across reference sections.

    min_features : int, default=1
        Minimum number of selected features required.

    split_order : {"root_to_leaf", "leaf_to_root"}, default="root_to_leaf"
        Split ordering used to retrieve split metadata.

    feature_result_key : str, default="hier_genes_dic"
        Key storing hierarchical feature dictionaries inside each result object.

    strict : bool, default=True
        If True, raise errors when a section or split key is missing.
        If False, skip missing entries.

    verbose : bool, default=True
        Whether to print summary information.

    Returns
    -------
    features_dic : dict
        Feature dictionary for the requested parent split.

    split_info : dict
        Metadata for the requested parent split.
    """

    if output_format not in {"section", "modality"}:
        raise ValueError("output_format must be either 'section' or 'modality'.")

    if count_num < 1:
        raise ValueError("count_num must be at least 1.")

    modality_results_dic = {
        "Gene": gene_results_dic,
        "Image": image_results_dic,
        "Protein": protein_results_dic,
    }

    modality_results_dic = {
        modality: results_dic
        for modality, results_dic in modality_results_dic.items()
        if results_dic is not None
    }

    if len(modality_results_dic) == 0:
        raise ValueError(
            "At least one of gene_results_dic, image_results_dic, or "
            "protein_results_dic must be provided."
        )

    for modality, results_dic in modality_results_dic.items():
        if feature_result_key not in results_dic:
            raise KeyError(
                f"{modality} results_dic must contain key "
                f"'{feature_result_key}'."
            )

    split_info = get_split_info_by_parent_node(
        hier_tree=hier_tree,
        parent_node=parent_node,
        order=split_order,
    )

    split_key_1 = split_info["split_key_1"]
    split_key_2 = split_info["split_key_2"]

    if verbose:
        print("\nUsing hierarchy split:")
        print(f"Parent node: {split_info['parent_node']}")
        print(f"Split 1: {split_key_1}")
        print(f"Split 2: {split_key_2}")
        print(f"Child 1 regions: {split_info['child_1_regions']}")
        print(f"Child 2 regions: {split_info['child_2_regions']}")

    # ------------------------------------------------------------
    # Infer reference sections
    # ------------------------------------------------------------
    if ref_section_list is None:
        ref_section_list = []
        seen_sections = set()

        for results_dic in modality_results_dic.values():
            for ref_section in results_dic[feature_result_key].keys():
                if ref_section not in seen_sections:
                    seen_sections.add(ref_section)
                    ref_section_list.append(ref_section)

    # ============================================================
    # Output format 1: section-specific features
    # ============================================================
    if output_format == "section":

        features_dic = {}

        for ref_section in ref_section_list:

            features_dic[ref_section] = {}

            for modality, results_dic in modality_results_dic.items():

                if ref_section not in results_dic[feature_result_key]:
                    if strict:
                        raise KeyError(
                            f"ref_section='{ref_section}' is not found in "
                            f"{modality} results_dic['{feature_result_key}']."
                        )
                    else:
                        continue

                section_feature_dic = results_dic[feature_result_key][ref_section]

                missing_keys = [
                    split_key
                    for split_key in [split_key_1, split_key_2]
                    if split_key not in section_feature_dic
                ]

                if len(missing_keys) > 0:
                    if strict:
                        raise KeyError(
                            f"Missing split keys for ref_section='{ref_section}', "
                            f"modality='{modality}': {missing_keys}."
                        )
                    else:
                        continue

                section_features = _union_preserve_order(
                    feature_lists=[
                        section_feature_dic[split_key_1],
                        section_feature_dic[split_key_2],
                    ],
                    count_num=1,
                )

                if len(section_features) < min_features:
                    if strict:
                        raise ValueError(
                            f"{ref_section}, {modality}: only "
                            f"{len(section_features)} features were selected, "
                            f"but min_features={min_features}."
                        )
                    else:
                        continue

                features_dic[ref_section][modality] = section_features

                if verbose:
                    print(
                        f"{ref_section}, {modality}: selected "
                        f"{len(section_features)} features."
                    )

    # ============================================================
    # Output format 2: modality-level features aggregated across sections
    # ============================================================
    else:

        features_dic = {}

        for modality, results_dic in modality_results_dic.items():

            section_level_feature_lists = []

            for ref_section in ref_section_list:

                if ref_section not in results_dic[feature_result_key]:
                    if strict:
                        raise KeyError(
                            f"ref_section='{ref_section}' is not found in "
                            f"{modality} results_dic['{feature_result_key}']."
                        )
                    else:
                        continue

                section_feature_dic = results_dic[feature_result_key][ref_section]

                missing_keys = [
                    split_key
                    for split_key in [split_key_1, split_key_2]
                    if split_key not in section_feature_dic
                ]

                if len(missing_keys) > 0:
                    if strict:
                        raise KeyError(
                            f"Missing split keys for ref_section='{ref_section}', "
                            f"modality='{modality}': {missing_keys}."
                        )
                    else:
                        continue

                # First combine the two directional feature lists within each section.
                section_features = _union_preserve_order(
                    feature_lists=[
                        section_feature_dic[split_key_1],
                        section_feature_dic[split_key_2],
                    ],
                    count_num=1,
                )

                if len(section_features) > 0:
                    section_level_feature_lists.append(section_features)

            # Then keep features shared by at least count_num reference sections.
            modality_features = _union_preserve_order(
                feature_lists=section_level_feature_lists,
                count_num=count_num,
            )

            if len(modality_features) < min_features:
                if strict:
                    raise ValueError(
                        f"{modality}: only {len(modality_features)} features "
                        f"were selected, but min_features={min_features}."
                    )
                else:
                    continue

            features_dic[modality] = modality_features

            if verbose:
                print(
                    f"{modality}: selected {len(modality_features)} features "
                    f"shared by at least {count_num} reference section(s)."
                )

    return features_dic, split_info


def construct_features_dic_for_all_splits(
    hier_tree,
    gene_results_dic=None,
    image_results_dic=None,
    protein_results_dic=None,
    ref_section_list=None,
    output_format="section",
    count_num=1,
    min_features=1,
    split_order="root_to_leaf",
    feature_result_key="hier_genes_dic",
    strict=False,
    verbose=True,
):
    """
    Construct features_dic for every binary split in a HierTree object.

    Returns a dictionary keyed by parent_node.

    Parameters
    ----------
    hier_tree : HierTree
        Fitted hierarchical tree object.

    gene_results_dic, image_results_dic, protein_results_dic : dict or None
        Hierarchical feature selection results.

    ref_section_list : list or None, optional
        Reference sections to include.

    output_format : {"section", "modality"}, default="section"
        Output format for each split-specific features_dic.

    count_num : int, default=1
        Minimum number of reference sections in which a feature must appear
        to be kept. Mainly useful when output_format="modality".

    min_features : int, default=1
        Minimum number of selected features required.

    split_order : {"root_to_leaf", "leaf_to_root"}, default="root_to_leaf"
        Order to iterate over hierarchy splits.

    feature_result_key : str, default="hier_genes_dic"
        Key storing selected hierarchical features.

    strict : bool, default=True
        If True, raise errors for missing sections, missing split keys,
        or too few features.

    verbose : bool, default=True
        Whether to print progress.

    Returns
    -------
    all_split_features_dic : dict
        Dictionary keyed by parent_node.

        Example:
            {
                "node_0": {
                    "split_info": {...},
                    "features_dic": {...}
                },
                "node_1": {
                    "split_info": {...},
                    "features_dic": {...}
                }
            }
    """

    split_info_dic = get_hierarchy_split_info_dic(
        hier_tree=hier_tree,
        order=split_order,
    )

    all_split_features_dic = {}

    for parent_node in split_info_dic.keys():

        if verbose:
            print("\n" + "=" * 80)
            print(f"Constructing features_dic for parent_node: {parent_node}")
            print("=" * 80)

        features_dic, split_info = construct_features_dic_for_parent_split(
            hier_tree=hier_tree,
            parent_node=parent_node,
            gene_results_dic=gene_results_dic,
            image_results_dic=image_results_dic,
            protein_results_dic=protein_results_dic,
            ref_section_list=ref_section_list,
            output_format=output_format,
            count_num=count_num,
            min_features=min_features,
            split_order=split_order,
            feature_result_key=feature_result_key,
            strict=strict,
            verbose=verbose,
        )

        all_split_features_dic[parent_node] = {
            "split_info": split_info,
            "features_dic": features_dic,
        }

    return all_split_features_dic


def get_features_dic_by_parent_node(
    all_split_features_dic,
    parent_node,
    return_split_info=False,
):
    """
    Retrieve features_dic for one hierarchy split by parent_node.

    Parameters
    ----------
    all_split_features_dic : dict
        Output from construct_features_dic_for_all_splits().

    parent_node : str
        Parent node defining the binary split.

    return_split_info : bool, default=False
        Whether to also return split metadata.

    Returns
    -------
    features_dic : dict
        Split-specific features_dic.

    split_info : dict, optional
        Returned only when return_split_info=True.
    """

    if parent_node not in all_split_features_dic:
        raise ValueError(
            f"parent_node='{parent_node}' is not found. "
            f"Available parent nodes are: {list(all_split_features_dic.keys())}."
        )

    features_dic = all_split_features_dic[parent_node]["features_dic"]
    split_info = all_split_features_dic[parent_node]["split_info"]

    if return_split_info:
        return features_dic, split_info

    return features_dic




