import dearpygui.dearpygui as dpg
from pathlib import Path
from absl import logging
import traceback

from enum import Enum
from types import TracebackType
from typing import Optional, Type, NamedTuple, Tuple

class EDataset(str, Enum):
    gopro_blur = "kaggle goprodeblur"

#class EWindowShow

def _viewport_dims() -> Tuple[int, int, int, int]:
    vp_window_width = dpg.get_viewport_width()
    vp_window_height = dpg.get_viewport_height()
    vp_client_width = min(dpg.get_viewport_client_width(), vp_window_width)
    vp_client_height = min(dpg.get_viewport_client_height(), vp_window_height)
    return (vp_client_width, vp_client_height,vp_window_width, vp_window_height)

def _on_dataset_path(sender, app_data, user_data):
    logging.info("Current Path: %s", dpg.get_value(app_data))

def _on_dataset_selection(sender, app_data, user_data):
    logging.info("Dataset selected: %s", repr(sender))
    match str(sender):
        case EDataset.gopro_blur.value:
            logging.info("Dataset recognized")
        case _:
            raise ValueError("Unrecognized dataset selected. How?")


class DPGConsoleHandler(logging.PythonHandler):
    def __init__(self, console_tag: str, num_lines: int):
        super().__init__()
        self.console_tag = console_tag # window tag inside which add_text will be called
        self.num_lines = num_lines
        self.line_tags = []

    def emit(self, record):
        log_entry = self.format(record)
        self.line_tags.append(dpg.add_text(log_entry, parent=self.console_tag))
        if len(self.line_tags) > self.num_lines:
            to_remove = self.line_tags.pop(0)
            dpg.delete_item(to_remove)


class WindowState(NamedTuple):
    dataset_path: Path


class Window:
    _bootstrap_tag = "Bootstrap Window"
    _console_tag = "Console Window"
    # https://dearpygui.readthedocs.io/en/latest/documentation/viewport.html
    def __init__(self, title: str, *, width: int, height: int):
        # TODO get max monitor resolution?
        if not isinstance(title, str) or not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError("Incorrect parameters to constructor `Window` Constructor")

        self._state = WindowState(dataset_path=Path.cwd() / 'data')
        self.console_handler = DPGConsoleHandler(Window._console_tag, 10)
        self._manual_resize = False
        
        dpg.create_context()
        # TODO  `small_icon` and `large_icon`, `decorated=False`, maybe `disable_close`
        dpg.create_viewport(title=title, width=width, height=height, x_pos=0, y_pos=0, resizable=True, vsync=True, decorated=True, clear_color=[0.1, 0.1, 0.1, 1])
        dpg.setup_dearpygui()
        logging.info("created Window \"%s\", resolution: (%ux%u)", title, width, height)

        dpg.set_viewport_resize_callback(lambda sender, app_data, user_data: user_data._on_resize(), user_data=self)

        vp_client_width = dpg.get_viewport_client_width()
        vp_client_height = dpg.get_viewport_client_height()
        vp_window_height = dpg.get_viewport_height()
        console_position_winrel = int(vp_window_height * 0.7)
        console_height = int(vp_client_height * 0.3)

        # create all possible screens here (if too much, add methods to create parametrized windows)
        with dpg.window(tag=Window._bootstrap_tag):
            dpg.add_text("Hello World")
            with dpg.menu_bar(user_data=None, tag="Main Menu Bar"):
                with dpg.menu(label="Dataset"):
                    with dpg.file_dialog(directory_selector=True, show=False, id="dataset_path_file_dialog", width=600, height=400, callback=_on_dataset_path, cancel_callback=lambda: logging.info("Path selection canceled")):
                        pass
                    dpg.add_input_text(label="Path", default_value=str(self._state.dataset_path), callback=_on_dataset_path)
                    with dpg.menu(label="Datasets"):
                        dpg.add_button(label=EDataset.gopro_blur, tag=EDataset.gopro_blur, callback=_on_dataset_selection)
                        with dpg.tooltip(EDataset.gopro_blur): # tag of the parent
                            dpg.add_text("Dataset from `https://www.kaggle.com/datasets/rahulbhalley/gopro-deblur`")
        
            with dpg.window(label="Console", tag=Window._console_tag, horizontal_scrollbar=True, no_focus_on_appearing=True, pos=(0, console_position_winrel), height=console_height, width=vp_client_width, no_close=True, no_move=True, no_resize=True):
                dpg.add_text("Console Output:")
                logging.get_absl_logger().addHandler(self.console_handler)


        
    def run_render_loop(self) -> None:
        while dpg.is_dearpygui_running():
            # ...
            dpg.render_dearpygui_frame()

    def __enter__(self):
        dpg.show_viewport()
        # select bootstrap window for display
        dpg.set_primary_window(Window._bootstrap_tag, True)
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], tb: Optional[TracebackType]) -> bool:
        dpg.destroy_context()
        if exc_type is not None:
            traceback.print_exception(exc_type, value=exc_value, tb=tb)

        return False # false to throw the exception exc_value (if any), true to swallow it

    def _adjust_console(self, vp_client_width, vp_client_height):
        dpg.set_item_pos(Window._console_tag, [0, 0.7 * vp_client_height])
        dpg.set_item_width(Window._console_tag, vp_client_width)
        dpg.set_item_height(Window._console_tag, int(0.3 * vp_client_height))

    def _on_resize(self):
        vp_client_width, vp_client_height, vp_window_width, vp_window_height = _viewport_dims()
        self._adjust_console(vp_client_width, vp_client_height)

        # hack: on Windows, the sequence "Maximize" and "Click and Drag Titlebar" breaks everything
        if self._manual_resize:
            self._manual_resize = False
        else:
            self._manual_resize = True
            dpg.set_viewport_width(vp_window_width + 1)