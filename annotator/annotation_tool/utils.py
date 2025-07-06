from qtpy.QtWidgets import QWidget, QLabel, QLayoutItem, QPushButton, QMessageBox
from qtpy.QtCore import QTimer
from napari import Viewer
from napari.layers import Points, Labels, Layer
from functools import partial
from pathlib import Path
import logging
import re
import sys


def setup_logger(log_dir: Path = Path("."), level=logging.DEBUG) -> logging.Logger:
    log_file = log_dir / "honeycomb_annotator.log"

    logger = logging.getLogger("honeycomb_annotation")
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_uncaught_exception

    logger.info("Honeycomb anntoator started")
    return logger


"""This is mostly a hacky workaround to disable functionality of the
napari ui to stop the user of doing things we don't want them to do.
This could not work in the future if things on the napari implementation
change.
See also here: https://forum.image.sc/t/disable-buttons-on-shapes-layer/55243/3"""


def restrict_brush_layer_tools(viewer: Viewer, brush_layer: Labels, points_layer: Points):
    hide_brush_controls = partial(
        hide_unwanted_layer_controls,
        allowed_labels={"opacity", "brush size", "label"},
    )

    hide_points_controls = partial(
        hide_unwanted_layer_controls,
        allowed_labels={"opacity", "point size"},
    )

    # Schedule it to run after the UI is fully built
    QTimer.singleShot(100, lambda: hide_unwanted_buttons(viewer, brush_layer))
    QTimer.singleShot(200, lambda: hide_brush_controls(viewer, brush_layer))
    QTimer.singleShot(250, lambda: hide_points_controls(viewer, points_layer))
    QTimer.singleShot(300, lambda: hide_layerlist_buttons(viewer))


def hide_unwanted_layer_controls(viewer: Viewer, layer: Layer, allowed_labels: set[str]):
    qt_controls = viewer.window._qt_viewer.controls
    layer_controls = qt_controls.widgets.get(layer)

    if layer_controls is None:
        logging.debug(f"Controls for layer '{layer.name}' not ready")
        return

    layout = layer_controls.layout()
    if not layout:
        logging.debug(f"No layout found in layer controls: {layer.name}")
        return

    for i in range(layout.count()):
        item = layout.itemAt(i)
        if not isinstance(item, QLayoutItem):
            continue

        widget = item.widget()
        if widget is None:
            continue

        if isinstance(widget, QLabel):
            label_text = widget.text().strip(":").lower()
            if label_text not in allowed_labels:
                logging.debug(f"Hiding row with label '{label_text}'")
                widget.hide()

                next_item = layout.itemAt(i + 1)
                if next_item:
                    next_widget = next_item.widget()
                    if next_widget:
                        next_widget.hide()


def hide_unwanted_buttons(viewer: Viewer, brush_layer: Labels):
    qt_controls = viewer.window._qt_viewer.controls
    brush_controls = qt_controls.widgets.get(brush_layer)

    if brush_controls is None:
        logging.debug("Brush controls not ready")
        return

    for child in brush_controls.findChildren(QWidget):
        if hasattr(child, "toolTip"):
            tip = str(child.toolTip()).lower()
            if any(
                mode in tip
                for mode in [
                    "fill",
                    "erase",
                    "erase label",
                    "bucket",
                    "polygon",
                    "pick",
                    "shuffle",
                ]
            ):
                logging.debug(f"Hiding button with tooltip: {tip}")
                child.hide()


def hide_layerlist_buttons(viewer: Viewer):
    buttons_widget = viewer.window._qt_viewer.layerButtons

    if buttons_widget is None:
        logging.debug("Layer buttons panel not found")
        return

    pattern = re.compile(r"^(new|delete).*layer")

    for btn in buttons_widget.findChildren(QPushButton):
        tooltip = btn.toolTip().strip().lower()

        if pattern.match(tooltip):
            logging.debug(f"Hiding button with tooltip: {tooltip}")
            btn.hide()


def show_unlabeled_warning(viewer: Viewer, unlabeled_cells_num: int):
    msg_box = QMessageBox(viewer.window._qt_window)
    msg_box.setIcon(QMessageBox.Warning)
    msg_box.setWindowTitle("Unlabeled Cells Remaining")
    msg_box.setText(
        f"There are still {unlabeled_cells_num} cells labeled as 'unlabeled'. \nCells that are not labeled are treated as background in mask."
    )
    msg_box.setStandardButtons(QMessageBox.Ok)
    msg_box.exec_()


def patch_napari_points_highlight():
    import napari.layers.points.points as napari_points

    original_set_highlight = napari_points.Points._set_highlight

    def safe_set_highlight(self, force=False):
        try:
            return original_set_highlight(self, force)
        except ValueError as e:
            if "None is not in list" in str(e):
                self._value = None
                self._highlight_index = []
                return self.refresh()
            raise
        except IndexError:
            print("this worked")
            self._highlight_index = []
            return self.refresh()

    napari_points.Points._set_highlight = safe_set_highlight
