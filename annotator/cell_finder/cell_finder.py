from pathlib import Path
import cv2
from numpy.typing import NDArray
import json
import uuid
import numpy as np
from typing import Callable

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
from dataclasses import asdict, replace

from cell_finder.utils import save_cell_find_config_to_json, show_cells_on_image
from cell_finder.hex_graph_builder import HexLatticeGraph
from cell_finder.models import HexGraphConfig


class CellFinder:
    def __init__(self, input_dir: Path, output_path: Path | None = None) -> None:
        self.input_dir = input_dir
        assert self.input_dir.is_dir(), "images path does not exist"
        self.output_path = output_path if output_path else self.input_dir
        assert self.output_path.is_dir(), "output path does not exist"
        self.image_paths = self._find_images()

    def run_with_graph_building(self, threshold, scale_factor, graph_config_base):

        results = self.run_with_template_matching(threshold=threshold, scale_factor=scale_factor)

        for img_name, data in results.items():
            path, duration, num, color_img, matches = data
            seed_points = np.array([(x, y) for x, y, _, _ in matches])
            lattice_vectors = HexLatticeGraph.estimate_lattice_vectors_by_angle_clustering(seed_points)
            h, w = color_img.shape[:2]
            graph_config = replace(graph_config_base, image_size=(w, h))

            graph = HexLatticeGraph(seed_points=seed_points, lattice_vectors=lattice_vectors, config=graph_config)
            graph.grow_graph_iteratively()

            results[img_name] = (path, duration, num, color_img, graph.cell_positions)

            if graph_config.debug:
                show_cells_on_image(color_img, graph.cell_positions)

        return results

    def run_with_template_matching(self, threshold: float = 0.725, scale_factor: float = 0.425):
        template_folder = Path(__file__).parent / "pattern_matching"
        return self.run(
            "template_matching", template_folder=template_folder, threshold=threshold, scale_factor=scale_factor
        )

    def run_with_hough_transform(
        self,
        dp: float = 1.2,
        min_dist: float = 20,
        param1: float = 50,
        param2: float = 30,
        min_radius: int = 5,
        max_radius: int = 50,
    ):
        return self.run(
            method="circle_hough_transform",
            dp=dp,
            min_dist=min_dist,
            param1=param1,
            param2=param2,
            min_radius=min_radius,
            max_radius=max_radius,
        )

    def run_with_gen_hough_transform(
        self,
        min_angle: float = 0,
        max_angle: float = 360,
        angle_step: float = 1,
        min_scale: float = 0.9,
        max_scale: float = 1.1,
        scale_step: float = 0.05,
    ):
        return self.run(
            method="generalized_hough_transform",
            min_angle=min_angle,
            max_angle=max_angle,
            angle_step=angle_step,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_step=scale_step,
        )

    def run_hybrid_detection(self, allowed_overlap: float = 0.05):
        template_folder = Path(__file__).parent / "pattern_matching"
        tmpl_results = self.run(method="template_matching", template_folder=template_folder)
        cht_results = self.run(method="circle_hough_transform", min_dist=30, dp=1.5, max_radius=32, min_radius=18)

        for img_name in tmpl_results:
            tmpl_data = tmpl_results[img_name]
            cht_data = cht_results.get(img_name)

            if cht_data is None:
                continue

            tmpl_img_path, _, tmpl_num_cells, color_image, tmpl_matches = tmpl_data
            _, _, cht_num_cells, _, cht_matches = cht_data

            combined = list(tmpl_matches)
            for x_c, y_c, r_c, s_c in cht_matches:
                if all(
                    np.hypot(x_c - x_t, y_c - y_t) > ((r_c + r_t) * (1 - allowed_overlap))
                    for x_t, y_t, r_t, s_t in tmpl_matches
                ):
                    combined.append((x_c, y_c, r_c, s_c))
            self._save_cells_to_json(combined, tmpl_img_path)

    def run(
        self, method: str = "template_matching", max_workers: int = 4, **method_kwargs
    ) -> dict[str, tuple[Path, float, int, NDArray, list[tuple]]]:

        detection_fn, supression_fn = self._get_detection_function(method)

        if not self.image_paths:
            print("No images found to process.")
            return {}

        def process_image(img_path: Path) -> tuple[Path, float, int, NDArray, list[tuple]]:
            start_time = time.time()
            gray_image, color_image = self._load_image_and_prepare(img_path)
            gray_image = self._apply_clahe(gray_image)
            matches = detection_fn(gray_image, **method_kwargs)
            if supression_fn:
                matches = supression_fn(matches)
            duration = time.time() - start_time
            return img_path, duration, len(matches), color_image, matches

        combined_results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_image, img_path) for img_path in self.image_paths]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Running {method}"):
                img_path, duration, num_cells, color_image, matches = future.result()
                combined_results[img_path.name] = (img_path, duration, num_cells, color_image, matches)

        return combined_results

    def _get_detection_function(self, method: str) -> tuple[Callable, Callable]:
        match method:
            case "template_matching":
                return self._template_matching, self._non_max_suppression
            case "circle_hough_transform":
                return self._circle_hough_transform, self._nms_circles
            case "generalized_hough_transform":
                return self._generalized_hough_transform, self._nms_circles
            case _:
                raise ValueError(
                    f"Detection method '{method}' is not supported. "
                    f"Choose from 'template_matching', 'circle_hough_transform'."
                )

    def _template_matching(
        self,
        gray_image,
        template_folder: Path,
        threshold: float = 0.725,
        scale_factor: float = 0.425,
    ) -> list[tuple]:
        assert template_folder.is_dir(), f"Template folder does not exist: {template_folder}"

        results = []
        for template_path in template_folder.glob("*.png"):
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue

            w, h = template.shape[::-1]
            result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)

            for pt in zip(*locations[::-1]):
                score = result[pt[1], pt[0]]
                center_x = pt[0] + w // 2
                center_y = pt[1] + h // 2
                radius = int(min(w, h) * scale_factor)
                results.append((center_x, center_y, radius, score))

        return results

    def _circle_hough_transform(
        self,
        gray_image,
        dp: float = 1.2,
        min_dist: float = 20,
        param1: float = 50,
        param2: float = 30,
        min_radius: int = 5,
        max_radius: int = 50,
    ) -> list[tuple]:
        detected_circles = cv2.HoughCircles(
            gray_image,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        results = []
        if detected_circles is not None and len(detected_circles) > 0:
            detected_circles = np.around(detected_circles[0, :]).astype(int)
            for x, y, r in detected_circles:
                results.append((x, y, r, 1.0))  # 1.0 is just a dummy score

        return results

    def _generalized_hough_transform(
        self,
        gray_image: np.ndarray,
        min_angle: float = 0,
        max_angle: float = 360,
        angle_step: float = 1,
        min_scale: float = 0.9,
        max_scale: float = 1.1,
        scale_step: float = 0.05,
    ) -> list[tuple[int, int, int, float]]:

        template = self.generate_hexagon_template()

        ght = cv2.createGeneralizedHoughGuil()
        ght.setTemplate(template)

        ght.setMinAngle(min_angle)
        ght.setMaxAngle(max_angle)
        ght.setAngleStep(angle_step)
        ght.setMinScale(min_scale)
        ght.setMaxScale(max_scale)
        ght.setScaleStep(scale_step)

        ght.setCannyLowThresh(50)
        ght.setCannyHighThresh(150)

        positions, votes = ght.detect(gray_image)
        if positions is None:
            return []

        results = []
        for (x, y, scale, angle), vote in zip(positions, votes):
            radius = int(24 * scale)
            results.append((int(x), int(y), radius, float(vote)))

        return results

    def generate_hexagon_template(self, radius: int = 24, margin: int = 20) -> np.ndarray:

        size = 2 * (radius + margin)
        center = (size // 2, size // 2)

        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        points = np.array(
            [(int(center[0] + radius * np.cos(angle)), int(center[1] + radius * np.sin(angle))) for angle in angles],
            dtype=np.int32,
        )
        points = points.reshape((-1, 1, 2))
        points_list = [points]

        img = np.zeros((size, size), dtype=np.uint8)
        cv2.polylines(img, points_list, isClosed=True, color=255, thickness=2)
        return img

    def _nms_circles(self, matches, min_dist=20, radius_tolerance=0.3, exp_radius=24, allowed_overlap_ratio=0.075):
        if len(matches) == 0:
            return []
        filtered_by_size = [
            m for m in matches if exp_radius * (1 - radius_tolerance) <= m[2] <= exp_radius * (1 + radius_tolerance)
        ]
        min_dist = exp_radius * 2 * (1 - allowed_overlap_ratio)
        kept = []
        for x, y, r, s in filtered_by_size:
            if all(np.hypot(x - xk, y - yk) >= min_dist for xk, yk, _, _ in kept):
                kept.append((x, y, r, s))
        return kept

    def _non_max_suppression(self, matches, overlap_thresh=0.3):
        if len(matches) == 0:
            return []

        boxes = np.array([[x - r, y - r, x + r, y + r] for x, y, r, _ in matches])
        scores = np.array([score for *_, score in matches])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= overlap_thresh)[0]
            order = order[inds + 1]

        return [matches[i] for i in keep]

    def _apply_clahe(self, img, clipLimit=2.0, tileGridSize=(8, 8)):

        if img.dtype in [np.float32, np.float64]:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        return clahe.apply(img)

    def _find_images(self) -> list[Path]:
        """TODO: This needs to be extended in order to find only
        background images. see background_img_annotator.py"""
        return sorted(self.input_dir.glob("*.png"))

    def _load_image_and_prepare(self, path: Path) -> tuple[NDArray, NDArray]:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return gray, color

    def save_artifacts(self, method: str, results: dict, parameters: dict):
        for result in results.values():
            path, _, _, _, matches = result
            self._save_cells_to_json(matches, path)

        save_cell_find_config_to_json(self.output_path, method=method, **parameters)

    def _save_cells_to_json(self, results, file_path: Path):
        cell_data = [
            {
                "id": str(uuid.uuid4()),
                "center_x": int(x),
                "center_y": int(y),
                "radius": int(r),
                "label": "unlabeled",
            }
            for x, y, r, s in results
        ]
        output_file = self.output_path / f"{file_path.stem}.json"
        with open(output_file, "w") as f:
            json.dump(cell_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run CellFinder with a chosen detection method.")
    parser.add_argument("input_path", type=str, help="Path to input image directory.")
    parser.add_argument(
        "--output_path", type=str, default=None, help="Optional path to output directory. Defaults to input path."
    )

    subparsers = parser.add_subparsers(dest="method", required=True, help="Detection method to use.")

    for method_name in [
        "template_matching",
        "circle_hough_transform",
        "hybrid",
    ]:
        subparsers.add_parser(method_name, help=f"Use {method_name} detection method.")

    graph_parser = subparsers.add_parser("graph_building", help="Use graph-based detection.")
    graph_parser.add_argument(
        "--curve_aware",
        action="store_true",
        help="curve-aware mode to estimate lattice (only valid with graph_building).",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else input_path

    cell_finder = CellFinder(input_dir=input_path, output_path=output_path)

    if args.method == "template_matching":
        threshold = 0.725
        scale_factor = 0.425
        parameters = {"threshold": threshold, "scale_factor": scale_factor}
        results = cell_finder.run_with_template_matching(threshold=threshold, scale_factor=scale_factor)
        cell_finder.save_artifacts(method="pattern matching", results=results, parameters=parameters)

    elif args.method == "circle_hough_transform":
        parameters = {"dp": 1.5, "min_dist": 30, "param1": 50, "param2": 30, "max_radius": 32, "min_radius": 18}
        results = cell_finder.run_with_hough_transform(min_dist=30, dp=1.5, max_radius=32, min_radius=18)
        cell_finder.save_artifacts(method="circle hough transform", results=results, parameters=parameters)

    elif args.method == "hybrid":
        cell_finder.run_hybrid_detection()

    elif args.method == "generalized_hough_transform":
        """This is currently not supported, since the results are too poor"""

        parameters = {
            "min_angle": 0,
            "max_angle": 360,
            "angle_step": 4,
            "min_scale": 0.85,
            "max_scale": 1.2,
            "scale_step": 0.025,
        }
        results = cell_finder.run_with_gen_hough_transform(
            min_scale=0.85, max_scale=1.2, scale_step=0.025, min_angle=0, max_angle=360, angle_step=4
        )
        cell_finder.save_artifacts(method=args.method, results=results, parameters=parameters)

    elif args.method == "graph_building":
        threshold = 0.725
        scale_factor = 0.425

        graph_config_base = HexGraphConfig(
            curve_aware_candidate_pred=args.curve_aware,
            prefer_method="curve" if args.curve_aware else "lattice_vector",
        )
        parameters = {"threshold": threshold, "scale_factor": scale_factor, "graph_config": asdict(graph_config_base)}

        results = cell_finder.run_with_graph_building(threshold, scale_factor, graph_config_base)

        cell_finder.save_artifacts(method="pattern matching", results=results, parameters=parameters)


if __name__ == "__main__":
    main()
