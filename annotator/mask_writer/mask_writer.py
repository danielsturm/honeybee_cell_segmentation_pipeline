from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from mask_writer.utils import visualize_mask_by_index

LABELS_PATH = Path(__file__).parents[1] / "data" / "label_classes.json"


class MaskWriter:
    def __init__(
        self, input_img_path: Path, input_json_path: Path, mask_out_path: Path, visualize_indices: list[int] = None
    ) -> None:
        self.input_img_path = input_img_path
        assert self.input_img_path.is_dir(), f"images path {self.input_img_path} does not exist"
        self.json_in_path = input_json_path
        assert self.json_in_path.is_dir(), f"json path {self.json_in_path} does not exist"
        self.output_path = mask_out_path
        assert self.output_path.is_dir(), f"output path {self.output_path} does not exist"
        self.data_paths = self._find_data()
        self.masks_out_path = self._create_out_dir()
        self._label_map = self._load_label_map()
        self.visualize_indices = visualize_indices

    def _load_label_map(self) -> list[dict[str, str]]:
        with open(LABELS_PATH) as f:
            label_data = json.load(f)
        return label_data

    def _create_out_dir(self) -> Path:
        out_path = self.output_path / "ground_truth_masks"
        Path.mkdir(out_path, parents=True, exist_ok=True)
        return out_path

    def _find_data(self) -> dict[str, tuple[Path, Path]]:
        result = {}
        png_files = list(self.input_img_path.glob("*.png"))
        for png_file in png_files:
            json_file = self.json_in_path / f"{png_file.stem}.json"
            if json_file.exists():
                result[png_file.stem] = (png_file, json_file)
        return result

    def run(self) -> None:
        label_dict = {entry["name"]: entry["png_index"] for entry in self._label_map}
        files = list(self.data_paths.items())
        for idx, (name, (img_path, json_path)) in enumerate(files):
            img = cv2.imread(str(img_path))
            height, width = img.shape[:2]

            mask = np.zeros((height, width), dtype=np.uint8)

            with open(json_path) as f:
                cells = json.load(f)

            for cell in cells:
                label = cell["label"]
                center_x = cell["center_x"]
                center_y = cell["center_y"]
                radius = cell["radius"]

                png_index = label_dict.get(label, 0) if label != "unlabeled" else 0

                cv2.circle(mask, (center_x, center_y), radius, png_index, -1)

            out_file = self.masks_out_path / f"{name}.png"
            cv2.imwrite(str(out_file), mask)

            if self.visualize_indices is not None and idx in self.visualize_indices:
                visualize_mask_by_index(self.masks_out_path, idx)


def main():
    parser = argparse.ArgumentParser(description="Run to create ground truth masks from json files")
    parser.add_argument("input_path", type=str, help="Path to input data directory.")
    parser.add_argument(
        "--input_json_path", type=str, default=None, help="Optional path to output directory. Defaults to input path."
    )
    parser.add_argument(
        "--masks_output_path", type=str, default=None, help="Optional path to output directory. Defaults to input path."
    )
    parser.add_argument(
        "--visualize_indices",
        type=int,
        nargs="+",
        default=None,
        help="List of indices of images to visualize (space separated).",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    input_json_path = Path(args.input_json_path) if args.input_json_path else input_path
    masks_output_path = Path(args.masks_output_path) if args.masks_output_path else input_path

    mask_creator = MaskWriter(input_path, input_json_path, masks_output_path, args.visualize_indices)
    mask_creator.run()


if __name__ == "__main__":
    main()
