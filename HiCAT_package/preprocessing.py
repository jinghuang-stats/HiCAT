import os
import cv2
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import MinMaxScaler


def extract_scribble_masks(
    image_path,
    annotated_image_path,
    label_color_dict,
    selected_labels=None,
    color_tolerance=30,
    resize_min_size=1000,
    min_contour_area=1000,
    output_dir=None,
    save_individual_masks=True,
):
    """
    Extract pathologist scribble masks from an annotated histology image.

    Parameters
    ----------
    image_path : str
        Path to the original histology image.

    annotated_image_path : str
        Path to the annotated image containing pathologist scribbles.

    label_color_dict : dict
        Dictionary mapping label names to RGB colors.
        Example:
        {
            "invasive_cancer": [236, 28, 36],
            "connective_tissue": [63, 72, 203]
        }

    selected_labels : list or None
        Labels to extract for this specific tissue section.
        If None, all labels in label_color_dict will be used.

    color_tolerance : int or dict
        Allowed RGB deviation when matching scribble colors.
        If int, the same tolerance is used for all labels and channels.
        If dict, it should map label names to RGB tolerances.
        Example:
        {
            "invasive_cancer": [30, 30, 30],
            "connective_tissue": [25, 25, 25]
        }

    resize_min_size : int
        Resize annotated image so that its shorter side equals this value.
        This speeds up contour detection.

    min_contour_area : int
        Minimum contour area to keep. Smaller detected regions are removed.

    output_dir : str or None
        Directory to save masks. If None, masks are not saved.

    save_individual_masks : bool
        Whether to save one binary mask per label.

    Returns
    -------
    ref_mask : np.ndarray
        Final label mask resized to the original image size.
        Shape: original image height x original image width.
        Values:
            0 = background / unlabeled
            1, 2, 3, ... = tissue labels

    label_id_dict : dict
        Dictionary mapping integer mask values to label names.
        Example:
        {
            0: "nan",
            1: "invasive_cancer",
            2: "connective_tissue"
        }

    d_mask : dict
        Dictionary mapping label names to binary masks at resized annotated-image resolution.
    """

    # ------------------------------------------------------------
    # Read original and annotated images
    # ------------------------------------------------------------
    img = cv2.imread(image_path)
    img_annotated = cv2.imread(annotated_image_path)

    if img is None:
        raise ValueError(f"Cannot read image_path: {image_path}")

    if img_annotated is None:
        raise ValueError(f"Cannot read annotated_image_path: {annotated_image_path}")

    # cv2 reads images as BGR, but user-provided colors are usually RGB
    original_height, original_width = img.shape[:2]

    # ------------------------------------------------------------
    # Decide which labels to extract
    # ------------------------------------------------------------
    if selected_labels is None:
        selected_labels = list(label_color_dict.keys())

    missing_labels = [label for label in selected_labels if label not in label_color_dict]
    if len(missing_labels) > 0:
        raise ValueError(f"These selected labels are not in label_color_dict: {missing_labels}")

    # ------------------------------------------------------------
    # Resize annotated image for faster processing
    # ------------------------------------------------------------
    resize_factor = resize_min_size / np.min(img.shape[:2])
    resize_width = int(img.shape[1] * resize_factor)
    resize_height = int(img.shape[0] * resize_factor)

    img_annotated_resized = cv2.resize(
        img_annotated,
        (resize_width, resize_height),
        interpolation=cv2.INTER_AREA
    )

    # ------------------------------------------------------------
    # Prepare output folder
    # ------------------------------------------------------------
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Extract binary mask for each label
    # ------------------------------------------------------------
    d_mask = {}

    for label in selected_labels:
        r, g, b = label_color_dict[label]

        if isinstance(color_tolerance, dict):
            r_tol, g_tol, b_tol = color_tolerance.get(label, [30, 30, 30])
        else:
            r_tol = g_tol = b_tol = color_tolerance

        # Because OpenCV image is BGR:
        b_channel = img_annotated_resized[:, :, 0]
        g_channel = img_annotated_resized[:, :, 1]
        r_channel = img_annotated_resized[:, :, 2]

        color_mask = (
            (b_channel > b - b_tol) & (b_channel < b + b_tol) &
            (g_channel > g - g_tol) & (g_channel < g + g_tol) &
            (r_channel > r - r_tol) & (r_channel < r + r_tol)
        ).astype(np.uint8)

        # Find contours from the binary color mask
        contours, _ = cv2.findContours(
            color_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Keep only sufficiently large contours
        mask = np.zeros(color_mask.shape, dtype=np.uint8)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_contour_area:
                cv2.drawContours(mask, [cnt], -1, 1, thickness=-1)

        d_mask[label] = mask

        if output_dir is not None and save_individual_masks:
            save_path = os.path.join(output_dir, f"{label}_mask.jpg")
            cv2.imwrite(save_path, mask * 255)

    # ------------------------------------------------------------
    # Merge individual masks into one reference mask
    # ------------------------------------------------------------
    ref_mask_resized = np.zeros(img_annotated_resized.shape[:2], dtype=np.uint8)

    label_id_dict = {0: "nan"}

    for idx, label in enumerate(selected_labels, start=1):
        mask = d_mask[label]
        ref_mask_resized[mask != 0] = idx
        label_id_dict[idx] = label

    # ------------------------------------------------------------
    # Resize final mask back to original image size
    # Important: use INTER_NEAREST for label masks
    # ------------------------------------------------------------
    ref_mask = cv2.resize(
        ref_mask_resized,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )

    if output_dir is not None:
        ref_mask_path = os.path.join(output_dir, "ref_mask.jpg")
        cv2.imwrite(ref_mask_path, ref_mask * 40)

    return ref_mask, label_id_dict, d_mask


def assign_spot_labels(
    obs_df,
    ref_mask,
    label_id_dict,
    x_col="x",
    y_col="y",
    spot_id_col=None,
):
    """
    Assign tissue labels to spots based on their x/y coordinates and ref_mask.

    Parameters
    ----------
    obs_df : pandas.DataFrame
        Spot metadata dataframe, such as adata.obs.

    ref_mask : np.ndarray
        Final annotation mask at the original image size.
        Shape: height x width.

    label_id_dict : dict
        Dictionary mapping integer mask IDs to label names.
        Example:
        {
            0: "nan",
            1: "invasive_cancer",
            2: "connective_tissue"
        }

    x_col : str
        Column name for x coordinates.

    y_col : str
        Column name for y coordinates.

    spot_id_col : str or None
        Optional column containing spot IDs.
        If None, obs_df.index is used.

    Returns
    -------
    spot_label_df : pandas.DataFrame
        Dataframe with spot ID, x, y, label ID, and label name.
    """

    results = []

    height, width = ref_mask.shape[:2]

    for spot_idx, row in obs_df.iterrows():
        x = int(row[x_col])
        y = int(row[y_col])

        # Check whether coordinates are inside the image
        if x < 0 or x >= width or y < 0 or y >= height:
            label_id = 0
            label_name = "nan"
        else:
            label_id = int(ref_mask[y, x]) # check x and y coordinates
            label_name = label_id_dict.get(label_id, "nan")

        if spot_id_col is None:
            spot_id = spot_idx
        else:
            spot_id = row[spot_id_col]

        results.append({
            "spot_id": spot_id,
            "x": x,
            "y": y,
            "label_id": label_id,
            "label": label_name,
        })

    spot_label_df = pd.DataFrame(results)

    return spot_label_df


def filter_low_exp_genes(input_adata, low_exp_thres=0.02):
    """
    Filter genes based on the fraction of spots/cells with non-zero expression.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.

    low_exp_thres : float
        Minimum fraction of spots/cells where a gene must be expressed.

    Returns
    -------
    filtered_genes : list
        Genes passing the expression frequency threshold.
    """

    X = input_adata.X

    if issparse(X):
        nonzero_exp_frac = np.asarray((X > 0).mean(axis=0)).ravel()
    else:
        nonzero_exp_frac = (X > 0).mean(axis=0)

    gene_names = input_adata.var.index.to_numpy()
    filtered_genes = gene_names[nonzero_exp_frac >= low_exp_thres].tolist()

    return filtered_genes


def normalize_adata(input_adata, method="min_max", copy=True):
    """
    Normalize AnnData.X within one sample.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.

    method : str
        Normalization method.
        Currently supports:
            "min_max" : scale each gene to [0, 1] within the sample
            None or "none" : no normalization

    copy : bool
        Whether to return a copied AnnData object.

    Returns
    -------
    output_adata : AnnData
        Normalized AnnData object.
    """

    if copy:
        output_adata = input_adata.copy()
    else:
        output_adata = input_adata

    if method is None or method == "none":
        return output_adata

    if method == "min_max":
        X = output_adata.X

        # MinMaxScaler usually expects dense matrix
        if issparse(X):
            X = X.toarray()
        else:
            X = np.asarray(X)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        output_adata.X = X_scaled

        return output_adata

    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def construct_ref_adata_dic(
    ref_section_list,
    data_path,
    dataset_name,
    file_template="{section}.h5ad",
    label_key="label",
    sample_key="sample",
    low_exp_thres=0.02,
    filter_low_exp=True,
    normalize=True,
    normalize_method="min_max",
    integrate_filtered=True,
    print_results=True
):
    """
    Construct filtered reference AnnData dictionary and merged AnnData object.

    This function performs:
        1. read reference AnnData objects,
        2. filter lowly expressed genes within each sample,
        3. identify common genes shared across samples,
        4. subset each sample to common genes,
        5. optionally perform min-max normalization within each sample,
        6. integrate all reference samples into one AnnData object.

    Parameters
    ----------
    ref_section_list : list
        List of reference tissue section/sample names.
        Example:
        ["H1", "G2", "E1"]

    data_path : str
        Directory containing reference AnnData files.

    dataset_name : str
    	The name of analyzed dataset

    file_template : str
        Template for AnnData file names.
        Use "{section}" as placeholder.

        Example:
        "{section}.h5ad"
        "sudo_HER2+BC_{section}_log_s=1_res=50_nbr=10_k=2.h5ad"
        "Brain_Visium_{section}_normalize+log_with_labels.h5ad"

    label_key : str
        Column in adata.obs containing tissue region labels.

    sample_key : str
        Column name used to store sample IDs after merging.

    low_exp_thres : float
        Minimum fraction of spots/cells with non-zero expression.

    filter_low_exp : bool
        Whether to filter lowly expressed genes within each sample.

    normalize : bool
        Whether to normalize each sample after gene filtering and common-gene selection.

    normalize_method : str
        Normalization method passed to normalize_adata().
        Default is "min_max".

    integrate_filtered : bool
        If True, all_adata is constructed from filtered and normalized reference samples.
        If False, all_adata is constructed from original reference samples restricted to common genes (after filtering).

    print_results : bool
        Whether to print processing information.

    Returns
    -------
    results : dict
        Dictionary containing:

        {
            "ref_adata_dic_raw": original reference AnnData dictionary,
            "ref_adata_dic_filtered": filtered and optionally normalized AnnData dictionary,
            "all_adata": merged AnnData object,
            "common_genes": common genes after filtering,
            "sample_names": sample names
        }
    """

    # ------------------------------------------------------------
    # 1. Read reference AnnData objects
    # ------------------------------------------------------------
    ref_adata_dic_raw = {}

    for section in ref_section_list:
        file_name = file_template.format(section=section, dataset_name=dataset_name)
        file_path = os.path.join(data_path, file_name)

        if print_results:
            print(f"Reading: {file_path}")

        adata = sc.read(file_path)

        if label_key not in adata.obs.columns:
            raise ValueError(
                f"{label_key} is not found in adata.obs for section: {section}"
            )

        ref_adata_dic_raw[section] = adata

        if print_results:
            print(f"{section}: raw shape = {adata.shape}")

    # ------------------------------------------------------------
    # 2. Filter lowly expressed genes within each sample
    # ------------------------------------------------------------
    ref_adata_dic_tmp = {}

    for section, adata in ref_adata_dic_raw.items():

        adata_tmp = adata.copy()

        if filter_low_exp:
            filtered_genes = filter_low_exp_genes(
                input_adata=adata_tmp,
                low_exp_thres=low_exp_thres
            )

            adata_tmp = adata_tmp[
                :,
                adata_tmp.var.index.isin(filtered_genes)
            ].copy()

            if print_results:
                print(
                    f"{section}: {len(filtered_genes)} genes retained "
                    f"after low-expression filtering."
                )

        ref_adata_dic_tmp[section] = adata_tmp

    # ------------------------------------------------------------
    # 3. Identify common genes shared across filtered samples
    # ------------------------------------------------------------
    common_genes = None

    for section in ref_section_list:
        genes = set(ref_adata_dic_tmp[section].var.index.tolist())

        if common_genes is None:
            common_genes = genes
        else:
            common_genes = common_genes & genes

    common_genes = sorted(common_genes)

    if len(common_genes) == 0:
        raise ValueError("No common genes found across reference samples.")

    if print_results:
        print(f"Number of common genes across samples: {len(common_genes)}")

    # ------------------------------------------------------------
    # 4. Subset to common genes and normalize within each sample
    # ------------------------------------------------------------
    ref_adata_dic_filtered = {}

    for section in ref_section_list:

        adata_tmp = ref_adata_dic_tmp[section][
            :,
            ref_adata_dic_tmp[section].var.index.isin(common_genes)
        ].copy()

        # Keep the same gene order across samples
        adata_tmp = adata_tmp[:, common_genes].copy()

        if normalize:
            adata_tmp = normalize_adata(
                input_adata=adata_tmp,
                method=normalize_method,
                copy=True
            )

        ref_adata_dic_filtered[section] = adata_tmp

        if print_results:
            print(f"{section}: filtered shape = {adata_tmp.shape}")

    # ------------------------------------------------------------
    # 5. Construct all_adata
    # ------------------------------------------------------------
    all_adata_list = []

    if integrate_filtered:
        # Use filtered and normalized data
        for section in ref_section_list:
            all_adata_list.append(ref_adata_dic_filtered[section].copy())
    else:
        # Use raw data, but restricted to common genes
        for section in ref_section_list:
            adata_raw = ref_adata_dic_raw[section][
                :,
                ref_adata_dic_raw[section].var.index.isin(common_genes)
            ].copy()

            adata_raw = adata_raw[:, common_genes].copy()

            all_adata_list.append(adata_raw)

    all_adata = ad.concat(
        all_adata_list,
        axis=0,
        join="inner",
        label=sample_key,
        keys=ref_section_list
    )

    all_adata.var["genes"] = all_adata.var.index.tolist()

    if print_results:
        print(f"Merged all_adata shape = {all_adata.shape}")
        print(f"Sample key added to all_adata.obs: {sample_key}")

    results = {
        "ref_adata_dic_raw": ref_adata_dic_raw,
        "ref_adata_dic_filtered": ref_adata_dic_filtered,
        "all_adata": all_adata,
        "common_genes": common_genes,
        "sample_names": ref_section_list
    }

    return results


def construct_merged_scaled_adata_and_gene_df(
    adata_dic,
    tissue_section_list,
    total_genes_list,
    merged_key="sample",
    normalize_method="min_max",
    print_results=True,
):
    """
    Construct a merged scaled AnnData object and gene-expression dataframe
    using the union of selected subtype marker genes across tissue sections.

    For each tissue section, this function:
        1. checks whether the section exists in `adata_dic`,
        2. selects genes available from `total_genes_list`,
        3. normalizes the selected gene-expression matrix within that section,
        4. concatenates all scaled AnnData objects across sections,
        5. converts the merged expression matrix into a pandas DataFrame.

    Parameters
    ----------
    adata_dic : dict
        Dictionary of tissue-section AnnData objects.

        Example:
        {
            "H1": adata_H1,
            "G2": adata_G2,
            "E1": adata_E1,
        }

    tissue_section_list : list
        List of tissue-section names to merge.

    total_genes_list : list
        Union of selected subtype marker genes across tissue sections.
        Only genes present in each section will be used for that section.

    merged_key : str, default="sample"
        Column name added to `.obs` of the merged AnnData object to record
        the tissue-section/source sample.

    normalize_method : str or None, default="min_max"
        Normalization method passed to `normalize_adata`.

    print_results : bool, default=True
        Whether to print merged AnnData variable information and sample counts.

    Returns
    -------
    merged_adata_sca : AnnData
        Concatenated and scaled AnnData object across all tissue sections.

    gene_df : pandas.DataFrame
        Dense gene-expression dataframe from `merged_adata_sca.X`.
        Rows are spots/cells and columns are genes.

    Raises
    ------
    KeyError
        If a tissue section in `tissue_section_list` is not present in `adata_dic`.

    ValueError
        If `total_genes_list` is empty, or if no selected genes are available
        in a given tissue section.
    """

    if total_genes_list is None or len(total_genes_list) == 0:
        raise ValueError("total_genes_list is empty.")

    adata_sca_list = []

    for tissue_section in tissue_section_list:
        if tissue_section not in adata_dic:
            raise KeyError(f"{tissue_section!r} is not present in adata_dic.")

        test_gene = adata_dic[tissue_section]

        available_genes = [
            g for g in total_genes_list
            if g in test_gene.var_names
        ]

        if len(available_genes) == 0:
            raise ValueError(
                f"No genes from total_genes_list are available in {tissue_section}."
            )

        test_gene_sub = test_gene[:, available_genes].copy()

        test_gene_sub_sca = normalize_adata(
            test_gene_sub,
            method=normalize_method,
        )

        adata_sca_list.append(test_gene_sub_sca)

    merged_adata_sca = ad.concat(
        adata_sca_list,
        axis=0,
        join="inner",
        label=merged_key,
        keys=tissue_section_list,
    )

    merged_adata_sca.var["genes"] = merged_adata_sca.var.index.tolist()

    if print_results:
        print(merged_adata_sca.var)
        print(merged_adata_sca.obs[merged_key].value_counts())

    if issparse(merged_adata_sca.X):
        X = merged_adata_sca.X.toarray()
    else:
        X = np.asarray(merged_adata_sca.X)

    gene_df = pd.DataFrame(
        X,
        index=merged_adata_sca.obs.index,
        columns=merged_adata_sca.var_names,
    )

    return merged_adata_sca, gene_df


def make_nonnegative_adata(input_adata, copy=True):
    """
    Shift AnnData.X to be non-negative feature-wise.

    For each gene/feature, subtract its minimum value across all spots/cells.
    This preserves relative differences within each feature while ensuring
    all values are >= 0.

    Parameters
    ----------
    input_adata : AnnData
        Input AnnData object.

    copy : bool, default=True
        Whether to return a copied AnnData object.

    Returns
    -------
    output_adata : AnnData
        AnnData object with non-negative X.
    """

    output_adata = input_adata.copy() if copy else input_adata

    X = output_adata.X

    if issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X)

    X_min = X.min(axis=0)
    X_nonneg = X - X_min

    output_adata.X = X_nonneg

    return output_adata








