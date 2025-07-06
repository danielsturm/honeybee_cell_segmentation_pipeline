import napari
from napari.layers import Image, Points, Labels
from napari.layers._data_protocols import LayerDataProtocol
import logging
import numpy as np
import pandas as pd
import uuid
from magicgui.widgets import ComboBox, Container, PushButton, Label, CheckBox
from annotation_tool.utils import restrict_brush_layer_tools, show_unlabeled_warning, patch_napari_points_highlight
from annotation_tool.annotation_model import AnnotationDTO, Cell
from typing import cast

from qtpy.QtCore import QObject, Signal  # type: ignore
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from qtpy.QtGui import QColor, QPalette


class AnnotationViewer(QObject):
    point_max_diameter = 100
    point_min_diameter = 1
    point_layer_name = "Cells"
    brush_layer_name = "BrushLayer"

    label_changed = Signal(str, list)
    point_data_changed = Signal(str, list)
    image_change = Signal(str)
    restore_prev_state = Signal()

    def __init__(
        self,
        label_map: dict[str, str],
        logger: logging.Logger,
    ) -> None:
        super().__init__()
        patch_napari_points_highlight()
        self.label_map = label_map
        self.logger = logger

        self._lock_labeled_mode = False

        self.viewer = napari.Viewer()
        self.lock_checkbox = self._create_lock_checkbox()
        self.label_menu = self._create_label_menu()
        self.nav_buttons = self._create_navigation_buttons()
        self.label_legend = self._create_label_legend()
        self.reset_button = self._create_reset_button()
        self._prevent_close_if_unlabeled()

    def register_layers(self, data: AnnotationDTO, button_label: str) -> None:
        # image layer
        self.viewer.add_image(data.image, name=data.image_name)

        # points layer
        points_layer = self.viewer.add_points(
            data.points,
            features=self._construct_features(data.labels, data.point_diameters, data.ids),
            face_color="cell_type",
            face_color_cycle=self.label_map,
            size=data.point_diameters,  # type: ignore
            name=self.point_layer_name,
        )
        points_layer.face_color_mode = "cycle"
        self.points_layer = points_layer
        self._set_init_point_params()
        self._update_feature_defaults_with_label()
        self._update_feature_defaults_with_uuid()
        self._update_feature_defaults_with_point_size()
        self._update_point_borders()
        self.update_button_label(button_label)

        # brush layer
        self.brush_layer = self.viewer.add_labels(
            np.zeros(data.image.shape[:2], dtype=np.uint8), name=self.brush_layer_name
        )
        self._set_custom_brush_limit()
        restrict_brush_layer_tools(self.viewer, self.brush_layer, self.points_layer)

        self._setup_key_bindings()

    def update_view(self, data: AnnotationDTO, button_label: str) -> None:
        try:
            """blocks temporary event emitting so _points_changed is not triggered
            when resetting or switching images"""
            self.points_layer.events.data.block()

            self.points_layer.selected_data = set()
            self.points_layer.refresh(force=True)

            # image layer
            self._remove_old_image_layer()
            self.viewer.add_image(data.image, name=data.image_name)

            # points layer
            self.points_layer.data = data.points
            self.points_layer.features = self._construct_features(data.labels, data.point_diameters, data.ids)
            self.points_layer.size = data.point_diameters
            self._update_point_borders()

            # brush layer
            new_data = np.zeros(data.image.shape[:2], dtype=np.uint8)
            self.brush_layer.data = cast(LayerDataProtocol, new_data)

            self._restore_layer_order()
            self.update_button_label(button_label)
            self._set_active_layer()
        finally:
            self.points_layer.events.data.unblock()

    def _set_active_layer(self) -> None:
        self.viewer.layers.selection.active = self.brush_layer

    def _restore_layer_order(self) -> None:
        layers = self.viewer.layers

        image_layer = next(layer for layer in layers if isinstance(layer, Image))
        points_layer = next(layer for layer in layers if isinstance(layer, Points))
        labels_layer = next(layer for layer in layers if isinstance(layer, Labels))

        layers.move(layers.index(image_layer), 0)
        layers.move(layers.index(points_layer), 1)
        layers.move(layers.index(labels_layer), len(layers) - 1)

    def _remove_old_image_layer(self) -> None:
        for layer in list(self.viewer.layers):
            if isinstance(layer, Image):
                self.viewer.layers.remove(layer)

    def print_layers(self):
        for layer in list(self.viewer.layers):
            print("layer type", type(layer))
            print("layer name", layer.name)

    def _create_label_legend(self):
        legend_widget = QWidget()
        layout = QVBoxLayout()
        for label, color in self.label_map.items():
            row = QWidget()
            row_layout = QHBoxLayout()
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            color_box = QLabel()
            color_box.setFixedSize(16, 16)
            color_box.setAutoFillBackground(True)
            opaque_color = color[:7]  # "#RRGGBB" -> Remove alpha for legend display
            qcolor = QColor(opaque_color)
            palette = color_box.palette()
            palette.setColor(QPalette.Window, qcolor)
            color_box.setPalette(palette)
            color_box.setStyleSheet(f"background-color: {opaque_color}; border: 1px solid #888;")

            text_label = QLabel(label)
            row_layout.addWidget(color_box)
            row_layout.addWidget(text_label)
            row.setLayout(row_layout)
            layout.addWidget(row)
        legend_widget.setLayout(layout)
        self.viewer.window.add_dock_widget(legend_widget, area="right", name="Label Legend")
        return legend_widget

    @property
    def label_categories(self) -> list[str]:
        return list(self.label_map.keys())

    def _construct_features(self, labels: list[str], point_diameters: np.ndarray, ids: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cell_type": pd.Categorical(labels, categories=self.label_categories),
                "diameter": point_diameters,
                "id": ids,
            }
        )

    def _set_init_point_params(self, init_point_size: int = 46, init_opacity: float = 0.75) -> None:
        self.points_layer.current_size = init_point_size
        self.points_layer.opacity = init_opacity

    def _setup_key_bindings(self) -> None:
        @self.viewer.bind_key("Ctrl+l")
        def toggle_lock_mode(viewer):
            self._lock_labeled_mode = not self._lock_labeled_mode
            self.lock_checkbox.value = self._lock_labeled_mode
            self.logger.info(f"Lock labeled points mode set to {self._lock_labeled_mode} with key")

        @self.viewer.bind_key("j")
        def apply_label_to_selected(viewer):
            selected_label = self.label_menu.value
            self._label_changed(selected_label)
            self.logger.info("Applied selected label via key bind")

        @self.viewer.bind_key("b")
        def activate_brush_layer(viewer):
            # Set brush layer as active
            self.viewer.layers.selection.active = self.brush_layer
            self.logger.info("Activated brush layer by key bind")

        @self.viewer.bind_key("c")
        def activate_points_layer(viewer):
            # Set points layer as active
            self.viewer.layers.selection.active = self.points_layer
            self.logger.info("Activated points layer by key bind")

        @self.viewer.bind_key("y")
        def points_opacity_to_zero(viewer):
            self.points_layer.opacity = 0.0
            self.logger.info("Set points layer opacity to 0.")

        @self.viewer.bind_key("x")
        def points_opacity_to_full(viewer):
            self.points_layer.opacity = 1.0
            self.logger.info("Set points layer opacity to 1.")

        @self.viewer.bind_key("Ctrl+i")
        def delete_inverse_selection(viewer):
            self._delete_inverse_point_selection()

    def _delete_inverse_point_selection(self) -> None:
        selected = set(self.points_layer.selected_data)
        if not selected:
            self.logger.info("No points selected, nothing to delete.")
            return
        all_indices = set(range(len(self.points_layer.data)))
        inverse_indices = all_indices - selected
        if not inverse_indices:
            self.logger.info("No unselected points to delete.")
            return
        self.points_layer.selected_data = inverse_indices
        self.points_layer.remove_selected()
        self.points_layer.selected_data = set(range(len(self.points_layer.data)))
        self.logger.info(f"Deleted {len(inverse_indices)} unselected points of {len(all_indices)} via Ctrl+i.")

    def _set_custom_brush_limit(self, max_size: int = 150) -> None:
        self.viewer.window._qt_viewer.controls.widgets[self.brush_layer].brushSizeSlider.setMaximum(max_size)
        self.brush_layer.brush_size = 70
        self.brush_layer.mode = "paint"

    def _on_brush_mask_change(self) -> None:
        if self.brush_layer.mode == "paint":
            mask_array = self.brush_layer.data
            selected = set()
            for i, (y, x) in enumerate(self.points_layer.data):
                if 0 <= int(y) < mask_array.shape[0] and 0 <= int(x) < mask_array.shape[1]:
                    if mask_array[int(y), int(x)] == 1:  # type: ignore
                        selected.add(i)

            self.points_layer.selected_data = selected  # type: ignore
            self.points_layer.refresh(force=True)

            self._update_point_borders()

            # === Prevent infinite recursion ===
            self.brush_layer.events.paint.disconnect(self._on_brush_mask_change)
            new_data = np.zeros_like(self.brush_layer.data)
            self.brush_layer.data = cast(LayerDataProtocol, new_data)
            self.brush_layer.events.paint.connect(self._on_brush_mask_change)

    def _on_active_layer_change(self, event):
        active = self.viewer.layers.selection.active
        if active == self.brush_layer:
            self.brush_layer.mode = "paint"
            self.brush_layer.brush_size = 70
        if active == self.points_layer:
            self.points_layer.mode = "select"

    def _focus_canvas_if_new_layer_selected(self, event=None):
        """This exists only to handle the 'bug' that a user could accidentially delete an
        entire layer. Can happen especially when selecting by brush, changing to points layer
        and hitting delete key. See also https://github.com/napari/napari/issues/7448"""
        # could be combined with _on_active_layer_change()
        active = self.viewer.layers.selection.active
        if active is None:
            return

        if isinstance(active, (Image, Points, Labels)):
            qt_viewer = self.viewer.window._qt_viewer
            qt_canvas_widget = qt_viewer.canvas.native
            if qt_canvas_widget is not None:
                qt_canvas_widget.setFocus()

    def _create_label_menu(self) -> ComboBox:
        label_menu = ComboBox(label="cell_type", choices=self.label_categories)
        label_widget = Container(widgets=[label_menu])
        self.viewer.window.add_dock_widget(label_widget, area="left")
        return label_menu

    def _update_label_menu(self, label_menu: ComboBox) -> None:
        new_label = str(self.points_layer.feature_defaults["cell_type"][0])
        if new_label != label_menu.value:
            """this is only to stop the ui from firing signal when
            only point is selected and label not changed. otherwise
            _label_changed() will be executed and spam signal"""
            with label_menu.changed.blocked():
                label_menu.value = new_label

    def _label_changed(self, selected_label: str) -> None:
        selected_points = list(self.points_layer.selected_data)

        if selected_points:
            self._update_points_label(selected_label, selected_points)

            updated_cells = self._build_cell_payload_from_indices(selected_points)
            for cell in updated_cells:
                cell.label = selected_label

            self.label_changed.emit(selected_label, updated_cells)

        self._update_feature_defaults_with_label(selected_label)

    def _update_points_label(self, selected_label: str, selected_points: list[int]):
        self.points_layer.features.loc[selected_points, "cell_type"] = selected_label
        self.points_layer.refresh_colors()

    def _update_point_borders(self) -> None:
        if len(self.points_layer.data) == 0:
            return

        selected = self.points_layer.selected_data
        border_colors = ["white" if i in selected else "black" for i in range(len(self.points_layer.data))]
        self.points_layer.border_color = border_colors

    def _points_changed(self, event) -> None:
        if event.action not in {"added", "adding", "changed", "removing", "removed"}:
            return
        match event.action:
            case "adding":
                """Napari seems to copy/block the default features in the
                'added' stage. So the uuid change is ineffective there"""
                self._update_feature_defaults_with_uuid()
                """In case no points are selected the points size changed
                is not triggered"""
                self._update_feature_defaults_with_point_size()
            case "added":
                # TODO: replace with _build_cell_payload_from_indices, pass [-1]
                new_point = self.points_layer.data[-1]
                feat = self.points_layer.features.iloc[-1]
                result = []
                result.append(
                    Cell(
                        feat["id"],
                        new_point[1],
                        new_point[0],
                        feat["diameter"],
                        feat["cell_type"],
                    )
                )
                self.point_data_changed.emit("added", result)
            case "changed":
                # TODO: replace with _build_cell_payload_from_indices
                result = []
                moved_indices = list(event.data_indices)
                moved_points = self.points_layer.data[moved_indices]
                moved_feats = self.points_layer.features.iloc[moved_indices]
                for i, point in enumerate(moved_points):
                    result.append(
                        Cell(
                            moved_feats["id"].iloc[i],
                            point[1],
                            point[0],
                            moved_feats["diameter"].iloc[i],
                            moved_feats["cell_type"].iloc[i],
                        )
                    )
                self.point_data_changed.emit("changed", result)
            case "removing":
                # TODO: replace with _build_cell_payload_from_indices,
                result = []
                to_be_removed_indices = list(event.data_indices)
                to_be_removed_feats = self.points_layer.features.iloc[to_be_removed_indices]
                for idx, row in to_be_removed_feats.iterrows():
                    result.append(Cell(row["id"], -1.0, -1.0, -1.0, "None"))
                self.point_data_changed.emit("removed", result)
                self.points_layer.selected_data = set()
                self.points_layer.refresh(force=True)
            case "removed":
                self.points_layer.selected_data = set()
                self.points_layer.refresh(force=True)
            case _:
                self.logger.warning(f"Unsupported event action: {event.action}")
                return

    def _set_feature_default(self, key: str, value) -> None:
        defaults = self.points_layer.feature_defaults
        defaults[key] = value
        self.points_layer.feature_defaults = defaults

    def _update_feature_defaults_with_label(self, label: str = "unlabeled") -> None:
        self._set_feature_default("cell_type", label)

    def _update_feature_defaults_with_uuid(self) -> None:
        self._set_feature_default("id", str(uuid.uuid4()))

    def _update_feature_defaults_with_point_size(self) -> None:
        self._set_feature_default("diameter", self.points_layer.current_size)

    def _build_cell_payload_from_indices(self, indices: list[int]) -> list[Cell]:
        result = []
        feats = self.points_layer.features.iloc[indices]
        data = self.points_layer.data[indices]
        for i, idx in enumerate(indices):
            result.append(
                Cell(
                    feats["id"].iloc[i],
                    data[i][1],
                    data[i][0],
                    feats["diameter"].iloc[i],
                    feats["cell_type"].iloc[i],
                )
            )
        return result

    def _on_scroll_change_point_size(self, _, event) -> None:
        if "Alt" not in event.modifiers:
            return

        delta = 1 if event.delta[0] > 0 else -1
        selected = list(self.points_layer.selected_data)
        if not selected:
            return

        for idx in selected:
            current_point_diameter = self.points_layer.features.at[idx, "diameter"]
            new_point_diameter = current_point_diameter + delta * 2
            new_point_diameter = min(
                self.point_max_diameter,
                max(self.point_min_diameter, new_point_diameter),
            )
            self.points_layer.features.at[idx, "diameter"] = new_point_diameter

        self.points_layer.size = self.points_layer.features["diameter"].values  # type: ignore

        self.points_layer.selected_data = set(selected)  # type: ignore

        self._update_feature_defaults_with_point_size()
        result = self._build_cell_payload_from_indices(selected)
        self.point_data_changed.emit("changed", result)

    def _on_point_size_changed(self, event):
        selected = list(self.points_layer.selected_data)
        if not selected:
            return

        current_sizes = self.points_layer.size
        is_scalar = np.isscalar(current_sizes)

        for idx in selected:
            new_point_diameter = current_sizes if is_scalar else current_sizes[idx]
            self.points_layer.features.at[idx, "diameter"] = new_point_diameter

        self._update_feature_defaults_with_point_size()
        result = self._build_cell_payload_from_indices(selected)
        self.point_data_changed.emit("changed", result)

    def _create_lock_checkbox(self) -> CheckBox:
        checkbox = CheckBox(label="Lock labeled points", value=False)
        self.viewer.window.add_dock_widget(checkbox, area="left")
        checkbox.changed.connect(self._on_lock_checkbox_changed)
        return checkbox

    def _on_lock_checkbox_changed(self, value: bool) -> None:
        self._lock_labeled_mode = value
        self.logger.info(f"Lock labeled points mode set to {value} with checkbox")

    def _filter_locked_selection(self, event=None):
        if not self._lock_labeled_mode:
            return
        features = self.points_layer.features
        to_keep = {i for i in self.points_layer.selected_data if features.iloc[i]["cell_type"] == "unlabeled"}
        if to_keep != self.points_layer.selected_data:
            self.logger.info("to_keep != selected_data")
            self.points_layer.selected_data = to_keep

    def _create_navigation_buttons(self) -> Container[PushButton | Label]:
        self.prev_button = PushButton(label="← Previous")
        self.nav_label = Label(label="")
        self.next_button = PushButton(label="Next →")
        button_container = Container(
            widgets=[self.prev_button, self.nav_label, self.next_button],
            layout="horizontal",
        )
        self.viewer.window.add_dock_widget(button_container, area="left")
        return button_container

    def update_button_label(self, label: str) -> None:
        self.nav_label.label = label

    def _create_reset_button(self):
        reset_button = PushButton(label="Reset")
        self.viewer.window.add_dock_widget(reset_button, area="right", name="Reset")
        return reset_button

    def _has_unlabeled_cells(self) -> int:
        # should live in the controller actually and call model.
        if self.points_layer is None:
            return 0
        count = (self.points_layer.features["cell_type"] == "unlabeled").sum()
        return int(count)

    def _on_change_image_clicked(self, direction: str) -> None:
        unlabeled_cells = self._has_unlabeled_cells()
        if unlabeled_cells:
            self.logger.warning(f"Changing image: {unlabeled_cells} are still unlabeled")
            show_unlabeled_warning(self.viewer, unlabeled_cells)
        self.image_change.emit(direction)

    def _prevent_close_if_unlabeled(self) -> None:
        qt_window = self.viewer.window._qt_window
        original_close_event = qt_window.closeEvent

        def custom_close_event(event):
            unlabeled_cells = self._has_unlabeled_cells()
            if unlabeled_cells:
                self.logger.warning(f"Closing viewer: {unlabeled_cells} are still unlabeled")
                show_unlabeled_warning(self.viewer, unlabeled_cells)
                original_close_event(event)
            else:
                original_close_event(event)

        qt_window.closeEvent = custom_close_event

    def connect_to_events(self) -> None:
        # self.points_layer.events.feature_defaults.connect(lambda event: self._update_label_menu(self.label_menu))
        self.points_layer.selected_data.events.items_changed.connect(self._filter_locked_selection)
        self.points_layer.selected_data.events.items_changed.connect(lambda e: self._update_point_borders())
        self.points_layer.events.data.connect(self._points_changed)

        self.label_menu.changed.connect(self._label_changed)

        self.points_layer.mouse_wheel_callbacks.append(self._on_scroll_change_point_size)
        self.points_layer.events.size.connect(self._on_point_size_changed)

        self.brush_layer.events.paint.connect(self._on_brush_mask_change)
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_change)
        self.viewer.layers.selection.events.active.connect(self._focus_canvas_if_new_layer_selected)

        self.prev_button.clicked.connect(lambda: self._on_change_image_clicked("prev"))
        self.next_button.clicked.connect(lambda: self._on_change_image_clicked("next"))
        self.reset_button.clicked.connect(self.restore_prev_state.emit)

    def run(self) -> None:
        napari.run()


"""
TODO: replace all complicated signatures to pass data with DTO objects
TODO: DTO should have serialization functions
TODO: Is the point size always in range?
TODO: key binding to delete points even when brush mask is activated
"""
