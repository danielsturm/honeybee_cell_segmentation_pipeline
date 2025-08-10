import cv2
from pathlib import Path
import matplotlib.pyplot as plt


def visualize_mask_by_index(output_path: Path, index: int) -> None:
    files = sorted(output_path.glob("*.png"))
    if index < 0 or index >= len(files):
        raise IndexError(f"Index {index} is out of range. Found {len(files)} mask files.")

    img_path = files[index]
    mask = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")

    plt.imshow(mask, cmap="nipy_spectral")
    plt.title(img_path.name)
    plt.axis("off")
    plt.show()
