from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class TreeResult:
	tree: Any = None
	distance_matrix: Optional[Any] = None
	label_profiles: Optional[Dict[str, Any]] = None
	node_features: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
	node_to_labels: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class ReferenceSelectionResult:
	selected_refs: List[Any] = field(default_factory=list)
	reference_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModalitySelectionResult:
	selected_modalities: List[str] = field(default_factory=list)
	modality_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class TransferResult:
	query_adata: Optional[Any] = None
	predicted_labels: Optional[Any] = None
	node_predictions: Dict[str, Any] = field(default_factory=dict)
	anchor_pairs: Dict[str, Any] = field(default_factory=dict)
	selected_modalities: List[str] = field(default_factory=list)
	reduction_method: Optional[str] = None
	scenario: Optional[str] = None


@dataclass
class HeterogeneityResult:
	region_scores: Dict[str, float] = field(default_factory=dict)
	ref_subtype_pred: Dict[str, Any] = field(default_factory=dict)
	subtype_markers: Dict[str, list[str]] = field(default_factory=dict)


@dataclass
class HiCATResult:
    processed_references: Optional[List[Any]] = None
    preprocessed_query: Optional[Any] = None
    tree_result: Optional[TreeResult] = None
    reference_result: Optional[ReferenceSelectionResult] = None
    transfer_result: Optional[TransferResult] = None



