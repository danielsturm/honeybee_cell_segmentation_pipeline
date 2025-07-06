from dataclasses import dataclass
import numpy as np
import logging
import copy


@dataclass
class Cell:
    id: str
    x: float
    y: float
    diameter: float
    label: str


@dataclass
class AnnotationDTO:
    image: np.ndarray
    image_name: str
    ids: list[str]
    points: np.ndarray
    point_diameters: np.ndarray
    labels: list[str]


class HoneyCombAnnotationData:
    def __init__(self, label_categories: list[str], logger: logging.Logger) -> None:
        self.logger = logger
        self._image: np.ndarray | None = None
        self._image_name: str | None = None
        self.label_categories = label_categories
        self.cells: list[Cell] = []
        self._previous_cell_state: list[Cell] = []

    def update_data(
        self,
        image: np.ndarray,
        image_name: str,
        ids: list[str],
        points: np.ndarray,
        point_diameters: np.ndarray,
        labels: list[str],
    ) -> None:
        self._image = image
        self._image_name = image_name
        self.cells = []
        for id, (y, x), r, label in zip(ids, points, point_diameters, labels):
            self.cells.append(Cell(id=id, x=float(x), y=float(y), diameter=float(r), label=label))

    def add_cells(self, added_cells: list[Cell]) -> None:
        existing_ids = {cell.id for cell in self.cells}
        for new_cell in added_cells:
            if new_cell.id in existing_ids:
                self.logger.error(f"Duplicate id found: {new_cell.id}, skipping")
                return
            self.cells.append(new_cell)

    def edit_cells(self, cells: list[Cell]) -> None:
        id_to_cell = {cell.id: cell for cell in cells}
        for i, existing_cell in enumerate(self.cells):
            if existing_cell.id in id_to_cell:
                updated = id_to_cell[existing_cell.id]
                existing_cell.x = updated.x
                existing_cell.y = updated.y
                existing_cell.diameter = updated.diameter
                existing_cell.label = updated.label

    def remove_cells(self, cells: list[Cell]) -> None:
        ids_to_remove = {cell.id for cell in cells}
        self.cells = [cell for cell in self.cells if cell.id not in ids_to_remove]
        self.logger.info(f"Removed {len(ids_to_remove)} cells. Remaining cells: {len(self.cells)}")

    def restore_previous_state(self) -> None:
        if self._previous_cell_state:
            self.cells = copy.deepcopy(self._previous_cell_state)
            self.logger.info("Restored previous annotation state.")
        else:
            self.logger.info("No previous state to restore.")

    def save_state(self) -> None:
        self._previous_cell_state = copy.deepcopy(self.cells)

    def clear_previous_state(self) -> None:
        self._previous_cell_state = []

    @property
    def cell_data(self) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
        ids = [cell.id for cell in self.cells]
        points = np.array([[cell.y, cell.x] for cell in self.cells])
        point_diameters = np.array([cell.diameter for cell in self.cells])
        labels = [cell.label for cell in self.cells]
        assert len(ids) == len(points) == len(point_diameters) == len(labels), "cell data is inconsistent"
        return ids, points, point_diameters, labels

    @property
    def full_data(self) -> AnnotationDTO:
        return AnnotationDTO(self.image, self.image_name, *self.cell_data)

    @property
    def image(self) -> np.ndarray:
        if self._image is None:
            raise RuntimeError("Image data not yet loaded")
        return self._image

    @property
    def image_name(self) -> str:
        if self._image_name is None:
            raise RuntimeError("Image name not yet set")
        return self._image_name
