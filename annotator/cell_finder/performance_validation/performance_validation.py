import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import json
from skimage.io import imread
import argparse

from cell_finder.utils import save_html_performance_report


class CellFindPerformanceValidator:

    def __init__(self, prediction_path: Path, ground_truth_path: Path | None = None) -> None:
        self.pred_path = prediction_path
        assert self.pred_path.is_dir(), "prediction path is not a directory"
        if ground_truth_path:
            self.gt_path = ground_truth_path
            assert self.gt_path.is_dir(), "ground truth path is not a directory"
        else:
            self.gt_path = Path(__file__).parent / "ground_truth"
        self.matched_paths = self._load_paths()
        self.results_path = self.pred_path / "results"

    def run_performance_validation(self, threshold: float = 15.0, visualize: bool = False):
        all_results = []

        for pair in self.matched_paths:
            img_path = pair["img_path"]
            gt_path = pair["gt_label_path"]
            pred_path = pair["pred_label_path"]
            image_name = img_path.stem

            gt_cells = self._load_cells(gt_path)
            pred_cells = self._load_cells(pred_path)

            result = self._match_cells(gt_cells, pred_cells, threshold)
            result["image_name"] = image_name
            result["gt_cells"] = gt_cells
            result["pred_cells"] = pred_cells
            result["img_path"] = img_path
            all_results.append(result)

            print(
                f"[{image_name}] TP: {result['true_positives']} | FP: {result['false_positives']} | FN: {result['false_negatives']}"
            )
            print(
                f"    Precision: {result['precision']:.2f}, Recall: {result['recall']:.2f}, F1: {result['f1_score']:.2f}, Mean Error: {result['mean_localization_error']:.2f}"
            )

        total_tp = sum(r["true_positives"] for r in all_results)
        total_fp = sum(r["false_positives"] for r in all_results)
        total_fn = sum(r["false_negatives"] for r in all_results)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print("\n=== Overall Performance ===")
        print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

        report_name = None
        if visualize:
            config = self._load_predictions_config()
            self._create_out_dir()
            report_name = save_html_performance_report(all_results, config, self.results_path)

        return total_tp, total_fp, total_fn, precision, recall, f1, report_name

    def _create_out_dir(self) -> None:
        Path.mkdir(self.results_path, exist_ok=True)

    def _load_cells(self, path) -> dict[str, int]:
        with open(path, "r") as f:
            cells = json.load(f)
        return cells

    def _load_image(self, image_path: Path) -> np.ndarray:
        return imread(str(image_path))

    def _load_paths(self):
        pngs = {p.stem: p for p in self.gt_path.glob("*.png")}
        gt_jsons = {p.stem: p for p in self.gt_path.glob("*.json")}
        pred_jsons = {p.stem: p for p in self.pred_path.glob("*.json")}

        all_stems = set(pngs.keys()) | set(gt_jsons.keys())

        pairs = []

        for stem in sorted(all_stems):
            img_path = pngs.get(stem)
            label_path = gt_jsons.get(stem)
            pred_path = pred_jsons.get(stem)

            if not img_path:
                print(f"Missing PNG for: {stem}")
                continue
            if not label_path:
                print(f"Missing JSON for: {stem}")
                continue
            if not pred_path:
                print(f"Missing prediction JSON for: {stem}")
                continue

            pairs.append({"img_path": img_path, "gt_label_path": label_path, "pred_label_path": pred_path})

        return pairs

    def _load_predictions_config(self) -> dict:
        config_list = list(self.pred_path.glob("cell_finder_config.json"))
        assert len(config_list) == 1
        with open(config_list[0], "r") as f:
            config = json.load(f)
        return config

    def _euclidean(self, p1: dict[str, int], p2: dict[str, int]):
        return np.sqrt((p1["center_x"] - p2["center_x"]) ** 2 + (p1["center_y"] - p2["center_y"]) ** 2)

    def _match_cells(self, gt_cells, pred_cells, threshold):
        if not gt_cells or not pred_cells:
            return {
                "true_positives": 0,
                "false_positives": len(pred_cells),
                "false_negatives": len(gt_cells),
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "mean_localization_error": None,
            }

        cost_matrix = np.zeros((len(gt_cells), len(pred_cells)))

        for i, gt in enumerate(gt_cells):
            for j, pred in enumerate(pred_cells):
                cost_matrix[i, j] = self._euclidean(gt, pred)

        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

        matches = []
        for i, j in zip(gt_indices, pred_indices):
            distance = cost_matrix[i, j]
            if distance < threshold:
                matches.append((gt_cells[i], pred_cells[j]))

        tp = len(matches)
        fp = len(pred_cells) - tp
        fn = len(gt_cells) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        error = np.mean([self._euclidean(gt, pred) for gt, pred in matches]) if matches else None

        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "mean_localization_error": error,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run performance validator to test performance of the cell finder on labeled data"
    )
    parser.add_argument("input_path", type=str, help="Path to input image directory (images and detected cells).")
    parser.add_argument(
        "--ground_truth_path",
        type=str,
        help="Path to ground truth. If not provided, will take from performance_validation/ground_truth",
    )
    parser.add_argument("--visualize", action="store_true", help="optional parameter to visualize and save report")
    args = parser.parse_args()
    input_path = Path(args.input_path)
    ground_truth_path = Path(args.ground_truth_path) if args.ground_truth_path else None

    validator = CellFindPerformanceValidator(prediction_path=input_path, ground_truth_path=ground_truth_path)
    validator.run_performance_validation(visualize=args.visualize)


if __name__ == "__main__":
    main()
