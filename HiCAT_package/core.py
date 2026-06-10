from .data import HiCATResult
from .tree_inference import build_hier_tree
from .reference_selection import select_refs
from .label_transfer import hier_label_transfer


class HiCAT:
    """
    Main workflow class for running the HiCAT pipeline.

    The class provides a high-level interface for:

    1. building a hierarchical label tree,
    2. selecting suitable reference sections,
    3. transferring labels from references to query data,
    4. storing all intermediate and final results in a HiCATResult object.
    """

    def __init__(self, config=None):
        """
        Initialize a HiCAT workflow object.

        Parameters
        ----------
        config : dict or None, optional
            Optional configuration dictionary for controlling the workflow.
            This can later store parameters such as the transfer scenario,
            reference selection settings, or heterogeneity analysis options.
        """
        self.config = config
        self.result = None

    def fit_transfer(
        self,
        references,
        query,
        label_key,
        scenario="within_study",
    ):
        """
        Run hierarchical label transfer from reference datasets to a query dataset.

        Parameters
        ----------
        references : list of anndata.AnnData
            Reference AnnData objects containing known labels.

        query : anndata.AnnData
            Query AnnData object to be annotated.

        label_key : str
            Column name in `.obs` that contains the reference labels.

        scenario : str, default="within_study"
            Label transfer scenario. For example:

            - "within_study"
            - "cross_study"
            - "cross_platform"

            The exact supported values depend on the implementation of
            `hier_label_transfer`.

        Returns
        -------
        self : HiCAT
            Returns the fitted HiCAT object. The full result is stored in
            `self.result`.
        """

        tree_result = build_hier_tree(
            reference_adatas=references,
            query_adata=query,
            label_key=label_key,
        )

        reference_result = select_refs(
            reference_adatas=references,
            query_adata=query,
            label_key=label_key,
        )

        transfer_result = hier_label_transfer(
            reference_adatas=reference_result.selected_references,
            query_adata=query,
            tree_result=tree_result,
            label_key=label_key,
            scenario=scenario,
        )

        self.result = HiCATResult(
            processed_references=references,
            preprocessed_query=query,
            tree_result=tree_result,
            reference_result=reference_result,
            transfer_result=transfer_result,
        )

        return self




