# Annotation Tool

- Uses Napari as framework (https://napari.org/stable/)
- Has three main components: Cell Finder, Annotation Tool and Segmentation Mask Creator
    - **Cell Finder**: Tries to find as many cells as possible beforehand. First it uses template matching to find cells. Then it reconstructs the missing cells with graph building.
    - **Annotation Tool**: UI to label the cells or add/remove and correct positions and sizes.
    - **Segmentation Mask Creator**: Generates pixelwise labeled segmentation masks from your annotated JSON files. Each mask is saved as a `.png` file, where each pixel value corresponds to a label class (with `0` as background/unlabeled).


## Install dependencies
- I use poetry for dependency management, therefore this differs a bit if you are using a conda environment

### With Conda Envionment
- inside the conda environment:
    - `conda install -c conda-forge poetry`
    - `poetry config virtualenvs.create false` (since you are using a conda environment, poetry should not create it's own)
    - cd into annotator/
    - run `poetry install`

### With Poetry Environment
- install poetry: https://python-poetry.org/docs/#installation
- cd into annotator/
    - run `poetry install`

## How to use

### Cell finder
- Run the Cell Finder to find as many cells as possible beforehand.
- This will created a json file with the found cells, with the same name as the image.
- Take graph_building as method, since the other methods don't perform well.
- **Run**:
```sh
poetry run cell-finder <input_path> [--output_path <output_path>] <method> [--curve_aware]
```

- `<input_path>`: Path to the directory containing input images (`.png` files).
- `--output_path`: (Optional) Path to save results. Defaults to the input path.
- `<method>`: Detection method to use. Options:
    - `template_matching`
    - `circle_hough_transform`
    - `hybrid`
    - `graph_building` (recommended)
- `--curve_aware`: (Optional, only for `graph_building`) Enables curve-aware lattice estimation for curved honeycomb patterns. Needs to be the last flag.

#### Validation of Results

You can validate the detection results against ground truth annotations using the performance validation tool.
This might help to tune parameters and objectively assess detection quality especially if you need to decide if `curve_aware` or `lattice_vector` works better.

```sh
poetry run performance-validation <predictions_path> [--ground_truth_path <ground_truth_path>] [--visualize]
```
- `<predictions_path>`: Required. Path to the directory containing predicted cell JSON files (and images).
- `--ground_truth_path <ground_truth_path>`: Optional. Path to the directory containing ground truth JSON files and images. If omitted, defaults to performance_validation/ground_truth (relative to the script).
- `--visualize`: Optional. If set, generates an HTML report with visualizations of the results.
<!-- -->
- Compares detected cells to ground truth JSON files.
- Reports precision, recall, and F1 score.
- Visualizes true positives, false positives, and false negatives.
- There are a few annotated ground truth images in `cell_finder/performance_validation/ground_truth`

---

### Annotation Tool
- Each image needs a json file with pre detected cells (see cell finder).
- If an image does not have a corresponding labels json, an empty one will be created
- **Labels**: Uses the labels in `data/label_classes.json`. Adjust here if you need other classes or colors.
- **Run**: start the annotation tool from annotator/ with `poetry run annotator </path/to/images>`

#### Interface Overview

- **Image viewer**: Displays the current image.
- **Points layer**: Each point marks a cell to annotate.
- **Brush layer**: Allows selection of multiple points at once.
- **Label menu**: Choose the label from the dropdown menu to assign to selected points.
- **Navigation buttons**: Move between images.
- **Reset button**: Undo the last change with **← Previous / Next →** buttons.
- **Label legend**: Shows all available labels and their colors.

#### Layers

##### Points Layer

- **Add points**: Click on the image.
- **Select points**: Click, use the selection rectangle or use the brush tool.
- **Move points**: Drag selected points to adjust positions.
- **Resize points**: Hold Alt and scroll mouse wheel on selected points.
- **Delete points**: Select and press Delete via del key or use the button.

##### Brush Layer

- **Activate**: Press `B` or select in the layer list.
- **Select points**: Paint over the area; all points under the brush are selected.
- **Adjust brush size**: Use the brush size slider.
- **Note**: Deleting points does not work when brush layer is activated. Select and switch to points layer

#### Key Bindings
make sure to focus the canvas when using key binds. Otherwise it would not work and even use the napari default key bindings.

| Key Combination | Action |
|-----------------|--------|
| `Ctrl+L`        | Toggle lock labeled points mode. Prevents selecting already labeled points (only 'unlabeled' can be selected now) |
| `J`             | Apply selected label to selected points |
| `B`             | Activate brush layer |
| `C`             | Activate points layer |
| `Y`             | Hide points (opacity = 0) |
| `X`             | Show points (opacity = 1) |
| `Ctrl+I`        | Delete all points except selected ones (Inverse deletion). To remove cells outside of the frame |
| `Delete`        | Delete selected points |

#### Output

Annotations are saved as JSON files with the same base name as the image. Each file contains an array of cell objects:

```json
[
  {
    "id": "fc4e04d4-7669-4321-8516-0f37ac1b47e3",
    "center_x": 2176.0,
    "center_y": 2965.0,
    "radius": 24.0,
    "label": "nectar"
  },
  ...
]
```

---

### Segmentation Mask Creator

From the `annotator/` directory, run:

```sh
poetry run mask-writer <input_path> [--input_json_path <json_path>] [--masks_output_path <output_path>] [--visualize_indices N [N ...]]
```

- `<input_path>`: Path to the directory containing your input images (`.png` files).
- `--input_json_path`: (Optional) Path to the directory containing the annotation JSON files. Defaults to `<input_path>`.
- `--masks_output_path`: (Optional) Path to the directory where the output masks will be saved. Defaults to `<input_path>`.
- `--visualize_indices`: (Optional) List of integer indices (space separated) specifying which images' masks to visualize after creation. For example, `--visualize_indices 0 2 5` will show the masks for the 1st, 3rd, and 6th images.

#### Output

- For each image, a mask PNG is created in the `ground_truth_masks` subdirectory of your output path.
- Each pixel in the mask is assigned an integer value according to its label:
    - `0`: background/unlabeled
    - Other values: as defined in `data/label_classes.json`