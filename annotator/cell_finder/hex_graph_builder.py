import numpy as np
import uuid
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN, KMeans
from collections import defaultdict

from scipy.spatial import KDTree

from cell_finder.models import HexGraphConfig, CellNode
from cell_finder.utils import visualize_hex_lattice_graph


class HexLatticeGraph:
    def __init__(self, seed_points, lattice_vectors, config: HexGraphConfig):
        self.seed_points = seed_points
        self.nodes = {}
        self.pos_index = []
        self.id_index = []
        self.vecs = [np.array(v) for v in lattice_vectors]
        self.config = config
        self.graph_changed = False

    @property
    def cell_positions(self) -> list[tuple[int, int, int, float]]:
        return [(*cell, 24, 1.0) for cell in self.pos_index]

    def grow_graph_iteratively(self) -> None:
        for pt in self.seed_points:
            self.add_node(pt)

        self.build_edges()

        for i in range(self.config.max_iterations):
            self.graph_changed = False

            raw_candidates = self.collect_missing_neighbor_candidates()
            if not raw_candidates:
                break

            validated = self.cluster_and_filter_candidates(raw_candidates)
            if not validated:
                break

            if self.config.debug:
                visualize_hex_lattice_graph(self, i, "validated", raw_candidates, validated)

            conflict_resolved = self.resolve_prediction_conflicts(validated=validated)

            for pos, _ in conflict_resolved:
                self.add_node(position=pos)
                self.graph_changed = True

            self.build_edges()

            if not self.graph_changed:
                break

    def resolve_prediction_conflicts(self, validated: list[dict]) -> list[dict]:
        if not validated:
            return []

        positions = np.array([pos for pos, _ in validated])
        clustering = DBSCAN(eps=self.config.cluster_conflict_pred_eps, min_samples=1).fit(positions)
        labels = clustering.labels_

        grouped = defaultdict(list)
        for idx, label in enumerate(labels):
            grouped[label].append(validated[idx])

        final_validated = []
        occupied = KDTree(np.array(self.pos_index)) if self.pos_index else None

        for group in grouped.values():

            if len(group) == 1:
                merged_pos, support = group[0]
                nearby = self._get_nearby_existing_nodes(merged_pos, r=self.config.pred_merge_dist, node_tree=occupied)

                if nearby:
                    if len(nearby) > 1:
                        self._set_predictors_edges_to_conflict(support)
                        continue

                    existing_id = self.id_index[nearby[0]]
                    existing_node = self.nodes[existing_id]

                    if self._existing_has_open_edges(existing_node, support):
                        self._merge_predicted_with_existing(support, existing_id)
                        continue
                    else:
                        self._set_predictors_edges_to_conflict(support)
                        continue

                extended_nearby = self._get_nearby_existing_nodes(
                    merged_pos, r=self.config.min_dist_nodes, node_tree=occupied
                )
                if extended_nearby:
                    self._set_predictors_edges_to_conflict(support)
                    continue

                final_validated.append(group[0])
                continue

            if len(group) > 3:
                for _, support in group:
                    self._set_predictors_edges_to_conflict(support)
                continue

            center = np.mean([pos for pos, _ in group], axis=0)
            nearby = self._get_nearby_existing_nodes(center, r=self.config.pred_merge_dist, node_tree=occupied)
            if nearby:
                for _, support in group:
                    self._set_predictors_edges_to_conflict(support)
                continue

            conflict, all_support = self._merge_predcitions_and_check_conflict(group)

            if conflict:
                self._set_predictors_edges_to_conflict(all_support)
                continue

            self._add_node_with_edge_assignments(center, all_support)
            self.graph_changed = True

        return self._final_conflict_check(final_validated, occupied)

    def _merge_predcitions_and_check_conflict(self, group):
        seen = set()
        conflict = False
        all_support = []
        for _, support in group:
            for s in support:
                pair = (s["source"], s["dir"])
                if pair in seen:
                    conflict = True
                else:
                    seen.add(pair)
                all_support.append(s)
        return conflict, all_support

    def _merge_predicted_with_existing(self, support: list[dict], existing_id: str) -> None:
        for s in support:
            pid = s["source"]
            dir_idx = s["dir"]
            if pid in self.nodes:
                self.nodes[pid].neighbors[dir_idx] = existing_id
                reverse_dir = self._get_reverse_direction(dir_idx)
                self.nodes[existing_id].neighbors[reverse_dir] = pid
                self.graph_changed = True

    def _final_conflict_check(
        self, final_validated: list[tuple[np.ndarray, list[dict]]], existing: KDTree | None
    ) -> list[dict]:
        if not final_validated:
            return []

        validated_positions = [pos for pos, _ in final_validated]
        internal_tree = KDTree(validated_positions) if validated_positions else None

        kept = []

        for i, (pos, support) in enumerate(final_validated):
            conflict = False

            if existing is not None:
                existing_close = existing.query_ball_point(pos, r=self.config.min_dist_nodes)
                if existing_close:
                    conflict = True

            if not conflict and internal_tree is not None:
                close_indices = internal_tree.query_ball_point(pos, r=self.config.min_dist_nodes)
                if any(idx != i for idx in close_indices):
                    conflict = True

            if conflict:
                self._set_predictors_edges_to_conflict(support)
            else:
                kept.append((pos, support))

        return kept

    def _get_reverse_direction(self, dir_idx: int) -> int:
        return (dir_idx + 3) % 6

    def _existing_has_open_edges(self, existing_node, support: list[dict]) -> bool:
        for s in support:
            dir_idx = s["dir"]
            reverse_dir = self._get_reverse_direction(dir_idx)
            if existing_node.neighbors.get(reverse_dir) is not None:
                return False
        return True

    def _get_nearby_existing_nodes(self, pos, r: float, node_tree: KDTree | None) -> list:
        if node_tree is None:
            return []
        return node_tree.query_ball_point(pos, r=r)

    def _set_predictors_edges_to_conflict(self, support: list[dict]):
        for s in support:
            pid = s["source"]
            dir_idx = s["dir"]
            if pid in self.nodes:
                self.nodes[pid].neighbors[dir_idx] = "CONFLICT"

    def _add_node_with_edge_assignments(self, position: np.ndarray, support: list[dict]):
        rounded_pos = np.round(position).astype(int)
        node = self.add_node(rounded_pos)

        for s in support:
            pred_id = s["source"]
            dir_idx = s["dir"]

            if pred_id not in self.nodes:
                continue

            self.nodes[pred_id].neighbors[dir_idx] = node.id

            if self.config.bidrectional_assignment:
                reverse_dir = self._get_reverse_direction(dir_idx)
                self.nodes[node.id].neighbors[reverse_dir] = pred_id

        return node

    def add_node(self, position):
        rounded_pos = np.round(position).astype(int)
        node_id = str(uuid.uuid4())
        node = CellNode(id=node_id, position=rounded_pos)
        self.nodes[node_id] = node
        self.pos_index.append(node.position)
        self.id_index.append(node_id)
        return node

    def _in_bounds(self, pos) -> bool:
        if self.config.image_size is None:
            return True
        x, y = pos
        width, height = self.config.image_size
        return 0 <= x < width and 0 <= y < height

    def build_edges(self) -> None:

        positions = np.array(self.pos_index)

        for _, node_id in enumerate(self.id_index):
            origin = self.nodes[node_id]
            origin_pos = origin.position

            for dir_index, vec in enumerate(self.vecs + [-v for v in self.vecs]):

                if origin.neighbors[dir_index] is not None:
                    continue

                predicted_pos = origin_pos + vec

                if not self._in_bounds(predicted_pos):
                    origin.neighbors[dir_index] = "OUT_OF_BOUNDS"
                    continue

                distances = np.linalg.norm(positions - predicted_pos, axis=1)
                close_indices = np.where(distances <= self.config.neighbour_pos_tolerance)[0]

                if len(close_indices) > 0:

                    neighbor_id = self.id_index[close_indices[0]]
                    neighbor_node = self.nodes[neighbor_id]
                    reverse_dir = self._get_reverse_direction(dir_index)
                    reverse_value = neighbor_node.neighbors[reverse_dir]

                    if reverse_value is not None and reverse_value != origin.id:
                        origin.neighbors[dir_index] = "CONFLICT"
                        continue

                    origin.neighbors[dir_index] = neighbor_id

                    if self.config.bidrectional_assignment:
                        self.nodes[neighbor_id].neighbors[reverse_dir] = node_id

    def collect_missing_neighbor_candidates(self):

        candidates = []
        directions = self.vecs + [-v for v in self.vecs]

        for node_id, node in self.nodes.items():
            pos = node.position

            for dir_idx in range(6):
                method = "curve"
                neighbor_id = node.neighbors[dir_idx]

                if neighbor_id is not None:
                    continue

                pred_pos = None

                if self.config.curve_aware_candidate_pred:
                    back_dir = self._get_reverse_direction(dir_idx)
                    prev1_id = node.neighbors[back_dir]

                    if self._is_uuid(prev1_id):
                        prev1 = self.nodes[prev1_id]
                        prev2_id = prev1.neighbors[back_dir]

                        if self._is_uuid(prev2_id):
                            prev2 = self.nodes[prev2_id]

                            pts = np.array([prev2.position, prev1.position, pos])

                            X = np.arange(3).reshape(-1, 1)
                            model_x = LinearRegression().fit(X, pts[:, 0])
                            model_y = LinearRegression().fit(X, pts[:, 1])

                            next_x = model_x.predict([[3]])[0]
                            next_y = model_y.predict([[3]])[0]
                            pred_pos = np.array([next_x, next_y])

                if pred_pos is None:
                    vec = directions[dir_idx]
                    pred_pos = pos + vec
                    method = "lattice_vector"

                if not self._in_bounds(pred_pos):
                    continue

                candidates.append((pred_pos, node_id, dir_idx, method))

        return candidates

    def _is_uuid(self, neighbor_ref: str | None) -> bool:
        try:
            uuid.UUID(str(neighbor_ref))
            return True
        except ValueError:
            return False

    def cluster_and_filter_candidates(self, candidates):

        if not candidates:
            return []

        pos_array = np.array([pos for (pos, *_) in candidates])

        clustering = DBSCAN(eps=self.config.cluster_pred_eps, min_samples=1).fit(pos_array)
        labels = clustering.labels_

        clustered = defaultdict(list)

        for idx, label in enumerate(labels):
            clustered[label].append(candidates[idx])

        filtered = []

        for cluster in clustered.values():
            if len(cluster) < self.config.cluster_vote_threshold:
                continue

            if self.config.prefer_method:
                method = self.config.prefer_method
                method_groups = defaultdict(list)
                for item in cluster:
                    method = item[3]
                    method_groups[method].append(item)

                if method in method_groups and len(method_groups[method]) >= self.config.cluster_vote_threshold:
                    used_cluster = method_groups[method]
                else:
                    used_cluster = cluster
            else:
                used_cluster = cluster

            positions = np.array([item[0] for item in used_cluster])
            mean_pos = positions.mean(axis=0)

            support = [{"source": item[1], "dir": item[2], "method": item[3], "pos": item[0]} for item in used_cluster]

            filtered.append((mean_pos, support))

        return filtered

    def get_node_by_position(self, x: int, y: int):
        target = np.array([x, y])
        for node in self.nodes.values():
            if np.array_equal(node.position, target):
                return node
        return None

    @classmethod
    def estimate_lattice_vectors_by_angle_clustering(
        cls,
        seed_points_xy: np.ndarray,
        k: int = 6,
        n_axes: int = 3,
        expected_spacing: float = 48.0,
        tolerance: float = 15.0,
    ) -> np.ndarray:

        min_dist = expected_spacing - tolerance
        max_dist = expected_spacing + tolerance

        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(seed_points_xy)
        distances, indices = nbrs.kneighbors(seed_points_xy)

        raw_vectors = []
        for i, idxs in enumerate(indices):
            origin = seed_points_xy[i]
            for j in idxs[1:]:  # skip self
                target = seed_points_xy[j]
                vec = target - origin
                dist = np.linalg.norm(vec)
                if min_dist <= dist <= max_dist:
                    raw_vectors.append(vec)

        if not raw_vectors:
            raise ValueError("No valid neighbor vectors found in the given distance range.")

        raw_vectors = np.array(raw_vectors)

        angles = np.arctan2(raw_vectors[:, 1], raw_vectors[:, 0])
        angles_deg = np.degrees(angles) % 180  # collapse opposite directions

        angle_kmeans = KMeans(n_clusters=n_axes, n_init=10, random_state=42)
        angle_kmeans.fit(angles_deg.reshape(-1, 1))
        centroid_angles = np.sort(angle_kmeans.cluster_centers_.flatten())  # degrees

        unit_vectors = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in centroid_angles])

        avg_dist = np.mean([np.linalg.norm(v) for v in raw_vectors])
        lattice_vectors = unit_vectors * avg_dist

        return lattice_vectors
