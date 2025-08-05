from dataclasses import dataclass, field
import numpy as np
from typing import Literal


@dataclass
class CellNode:
    id: str
    position: np.ndarray  # shape (2,)
    neighbors: dict = field(default_factory=lambda: {i: None for i in range(6)})


@dataclass
class HexGraphConfig:
    """
    Configuration settings for the HexLatticeGraph class.

    This dataclass defines all tunable hyperparameters that control
    prediction, clustering, conflict resolution, and graph structure growth.

    Attributes:
        neighbour_pos_tolerance (float):
            Maximum allowed distance (in pixels) to consider two nodes as neighbors.
            Used in `build_edges` to decide if a predicted neighbor position matches an existing node.

        bidrectional_assignment (bool):
            If True, all edges are added bidirectionally (i.e., mutual links between nodes).
            Used in `build_edges`, `_add_node_with_edge_assignments`, and conflict resolution logic.

        image_size (tuple[int, int] | None):
            Optional (width, height) bounds of the image space.
            Used in `_in_bounds` to filter out predictions that lie outside the image.

        max_iterations (int):
            Maximum number of growth iterations when calling `grow_graph_iteratively`.

        curve_aware_candidate_pred (bool):
            If True, attempts to use three-point curve prediction for missing nodes using linear regression.
            Falls back to lattice vector prediction if unavailable.
            Used in `collect_missing_neighbor_candidates`.

        cluster_vote_threshold (int):
            Minimum number of supporting predictions (votes) required to keep a clustered candidate.
            Used in `cluster_and_filter_candidates`.

        cluster_pred_eps (float):
            Distance threshold for DBSCAN clustering of raw predictions (not yet validated).
            Used in `cluster_and_filter_candidates`.

        pred_merge_dist (float):
            Maximum distance between a predicted node and an existing one to allow merging.
            Used in `resolve_prediction_conflicts` for both singleton and small clusters.

        cluster_conflict_pred_eps (float):
            Distance threshold for DBSCAN used specifically in conflict resolution clustering.
            Allows grouping of predictions that may conflict.
            Used in `resolve_prediction_conflicts`.

        min_dist_nodes (float):
            Minimum distance between newly predicted nodes and:
                - existing nodes
                - other validated predictions
            If violated, predictions are discarded and set to CONFLICT.
            Used in `_final_conflict_check` and optionally during conflict resolution.

        prefer_method (Literal["curve", "lattice_vector"] | None):
            Preferred prediction method when both are available in a cluster.
            If set, `cluster_and_filter_candidates` filters clusters by this method first.

        debug (bool):
            Enables debug logging and visualizations when True.
            Debug hooks are found in `grow_graph_iteratively` and related methods.
    """

    neighbour_pos_tolerance: float = 20.5
    bidrectional_assignment: bool = False
    image_size: tuple[int, int] | None = None
    max_iterations: int = 15
    curve_aware_candidate_pred: bool = True
    cluster_vote_threshold: int = 2
    cluster_pred_eps: float = 17.0
    pred_merge_dist: float = 27.5
    cluster_conflict_pred_eps: float = 28.5
    min_dist_nodes: float = 33.5
    prefer_method: Literal["curve", "lattice_vector"] | None = "curve"
    debug: bool = False
