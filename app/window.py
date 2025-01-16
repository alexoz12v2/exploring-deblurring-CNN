import faulthandler
import sys
from absl.app import parse_flags_with_usage
import dearpygui.dearpygui as dpg
from pathlib import Path
from absl import logging
from absl.logging import converter
from absl import command_name

# python standaard
from enum import Enum
from types import TracebackType
from typing import Optional, Type, NamedTuple, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import traceback

# os related
import platform
import os
import subprocess
import io
import shutil
from shutil import copyfileobj

# multiprocessing/multithreading
import multiprocessing as mp
import multiprocessing.managers
from multiprocessing.context import SpawnContext
from multiprocessing.managers import SyncManager
from multiprocessing import Process, Queue, Value, Manager
from threading import Thread

# must come before portalocker
import pywin32_patch  # type: ignore # noqa: F401
import portalocker as pl

import utils
from utils import AtomicBool
import data
import time


class EDataset(str, Enum):
    gopro_blur = "kaggle goprodeblur"
    blur_dataset = "blur dataset"


class EDatasetPath(Enum):
    folder = (0,)
    training_folder = (1,)
    test_folder = 2


class EProcessKeys(Enum):
    handler_tag = 0,
    handler_lines = 1,


class Extent2D(NamedTuple):
    """screen space extent"""

    width: int
    height: int


# TODO add shutdown requested
# todo add logging interface linked to a panel
def _worker_entry_point(queue: Queue, atomic_flag: AtomicBool):
    def setup_absl_logging_for_process():
        command_name.make_process_name_useful()
        parse_flags_with_usage([sys.argv[0]])
        logging.use_absl_handler()
        logging.set_verbosity(logging.INFO)
        if faulthandler:
            try:
                faulthandler.enable()
            except Exception:  # pylint: disable=broad-except
                pass

    setup_absl_logging_for_process()
    #sys.stdout = open("Y:\\worker_stdout.log", "a+")
    #sys.stderr = open("Y:\\worker_stderr.log", "a+")
    sys.excepthook = lambda exc_value, exc_type, tb: {
        logging.error(traceback.format_exception(exc_type, exc_value, tb))
    }

    logging.info("Process Worker started correctly")
    while not atomic_flag.should_close():
        while not atomic_flag.load():
            time.sleep(0.1)

        if not queue.empty():
            func, args, kwargs = queue.get()
            #logging.info("Executing function %s", func.__name__)
            print(f"Executing function {func.__name__}")
            func(*args, **kwargs)

        atomic_flag.reset()


def _viewport_dims() -> Tuple[Extent2D, Extent2D]:
    vp_window_width = dpg.get_viewport_width()
    vp_window_height = dpg.get_viewport_height()
    vp_client_width = min(dpg.get_viewport_client_width(), vp_window_width)
    vp_client_height = min(dpg.get_viewport_client_height(), vp_window_height)
    return (
        Extent2D(vp_client_width, vp_client_height),
        Extent2D(vp_window_width, vp_window_height),
    )


def _listen_for_logs(console_logger: logging.PythonHandler, msg_queue: Queue):
    while True:
        byte_str = msg_queue.get() # blocks here
        fn, lno, func, sinfo = logging.get_absl_logger().findCaller(False, 0)
        record = logging.get_absl_logger().makeRecord("worker", logging.INFO, fn, lno, str(byte_str), [])
        console_logger.emit(record)

class DPGConsoleHandler(logging.PythonHandler):
    _color_error = [200, 30, 0]
    _color_warning = [200, 135, 0]
    _color_trace = [128, 128, 128]
    _color_info = [200, 200, 200]

    # TODO init starts a thread which listens on a fifo queue between main process and worker process
    def __init__(self, console_tag: str, num_lines: int, msg_queue: Queue):
        super().__init__()
        self.console_tag = (
            console_tag  # window tag inside which add_text will be called
        )
        self.num_lines = num_lines
        self.line_tags = []

        self.io_thread = Thread(target=_listen_for_logs, args=(self, msg_queue))

    

    def _level_to_color(self, level):
        std_level = converter.standard_to_absl(level)
        if std_level >= logging.DEBUG:
            return DPGConsoleHandler._color_trace
        elif std_level >= logging.INFO:
            return DPGConsoleHandler._color_info
        elif std_level >= logging.WARNING:
            return DPGConsoleHandler._color_warning
        else:
            return DPGConsoleHandler._color_error

    def emit(self, record):
        log_entry = self.format(record)
        color = self._level_to_color(record.levelno)
        self.line_tags.append(
            dpg.add_text(log_entry, parent=self.console_tag, color=color)
        )
        if len(self.line_tags) > self.num_lines:
            to_remove = self.line_tags.pop(0)
            dpg.delete_item(to_remove)

    def flush(self):
        pass

    def write(self, message):
        dpg.add_text(
            message, parent=self.console_tag, color=DPGConsoleHandler._color_info
        )
        if len(self.line_tags) > self.num_lines:
            to_remove = self.line_tags.pop(0)
            dpg.delete_item(to_remove)


class Panel(ABC):
    def __init__(self, window, *, label: str, tag: str, **kwargs) -> None:
        self._children = []
        on_close_callback = (
            kwargs["on_close"]
            if "on_close" in kwargs
            else lambda sender, app_data, user_data: user_data.remove_window(self.tag)
        )
        with dpg.stage() as stage:
            self.tag: str = dpg.add_window(
                label=label,
                tag=tag,
                user_data=window,
                on_close=on_close_callback,
                **kwargs,
            )
        self.stage = stage

    def add_child(self, child):
        dpg.move_item(child.tag, parent=self.tag)

    def submit(self):
        dpg.unstage(self.stage)

    @abstractmethod
    def on_resize(self, vp_window: Extent2D, vp_client: Extent2D):
        pass


class Console(Panel):
    _console_tag = "Console Window:"
    _instance_count = 0

    def __init__(self, window, *, label: str):
        vp_client, vp_window = _viewport_dims()
        console_position_winrel = int(vp_window.height * 0.7)
        console_height = int(0.3 * vp_client.height)
        tag = Console._console_tag + str(Console._instance_count)
        Console._instance_count = Console._instance_count + 1
        super().__init__(
            window,
            label=label,
            tag=tag,
            horizontal_scrollbar=True,
            no_collapse=True,
            no_focus_on_appearing=True,
            pos=(0, console_position_winrel),
            height=console_height,
            width=vp_client.width,
            no_close=True,
            no_move=True,
            no_resize=True,
        )
        self.console_handler = DPGConsoleHandler(tag, 100)
        logging.get_absl_logger().addHandler(self.console_handler)

    def on_resize(self, vp_window, vp_client):
        dpg.set_item_pos(self.tag, [0, 0.7 * vp_client.height])
        dpg.set_item_width(self.tag, vp_client.width)
        dpg.set_item_height(self.tag, int(0.3 * vp_client.height))


@dataclass
class WindowState:
    dataset_path: Path
    dataset_path_eelected: bool


class Window:
    _bootstrap_tag = "Bootstrap Window"
    _console_tag = "Console Window"
    _dataset_path_tag = "Dataset_Path_ID"
    _lock_file_name = ".ecdnn_lock"

    # https://dearpygui.readthedocs.io/en/latest/documentation/viewport.html
    def __init__(self, title: str, *, ctx: SpawnContext, width: int, height: int):
        # TODO get max monitor resolution?
        if (
            not isinstance(title, str)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width <= 0
            or height <= 0
        ):
            raise ValueError("Incorrect parameters to constructor `Window` Constructor")

        self._state = WindowState(
            dataset_path=Path.cwd() / "data", dataset_path_eelected=False
        )
        self._manual_resize = False
        self._panels: dict[str, Panel] = {}

        dpg.create_context()
        # TODO  `small_icon` and `large_icon`, `decorated=False`, maybe `disable_close`
        dpg.create_viewport(
            title=title,
            width=width,
            height=height,
            x_pos=0,
            y_pos=0,
            resizable=True,
            vsync=True,
            decorated=True,
            clear_color=[0.1, 0.1, 0.1, 1],
        )
        dpg.setup_dearpygui()
        logging.info('created Window "%s", resolution: (%ux%u)', title, width, height)

        dpg.set_viewport_resize_callback(
            lambda sender, app_data, user_data: user_data._on_resize(), user_data=self
        )

        # create all possible screens here (if too much, add methods to create parametrized windows)
        with dpg.window(tag=Window._bootstrap_tag):
            dpg.add_text("Hello World")
            with dpg.menu_bar(user_data=None, tag="Main Menu Bar"):
                with dpg.menu(label="Dataset"):
                    with dpg.file_dialog(
                        directory_selector=True,
                        show=False,
                        tag="dataset_path_file_dialog",
                        width=600,
                        height=400,
                        callback=lambda sender,
                        app_data,
                        user_data: user_data._on_dataset_path(sender, app_data),
                        user_data=self,
                        cancel_callback=lambda: logging.info("Path selection canceled"),
                    ):
                        pass
                    dpg.add_button(
                        label="Choose Dataset Path...",
                        # callback=lambda: dpg.show_item("dataset_path_file_dialog"),
                        callback=lambda sender,
                        app_data,
                        user_data: user_data._ask_directory(),
                        user_data=self,
                    )
                    dpg.add_text(
                        label="Path",
                        default_value=str(self._state.dataset_path),
                        tag=Window._dataset_path_tag,
                    )
                    with dpg.menu(label="Datasets"):
                        dpg.add_button(
                            label=EDataset.gopro_blur,
                            tag=EDataset.gopro_blur,
                            callback=(
                                lambda sender,
                                app_data,
                                user_data: user_data._on_dataset_selection(
                                    sender, app_data
                                )
                            ),
                            user_data=self,
                        )
                        dpg.add_button(
                            label=EDataset.blur_dataset,
                            tag=EDataset.blur_dataset,
                            callback=(
                                lambda sender,
                                app_data,
                                user_data: user_data._on_dataset_selection(
                                    sender, app_data
                                )
                            ),
                            user_data=self,
                        )
                        dpg.add_button(
                            label="test",
                            tag="test",
                            callback=(
                                lambda sender,
                                app_data,
                                user_data: user_data._on_dataset_selection(
                                    sender, app_data
                                )
                            ),
                            user_data=self,
                        )
                        with dpg.tooltip(EDataset.gopro_blur):  # tag of the parent
                            dpg.add_text(
                                "Dataset from `https://www.kaggle.com/datasets/rahulbhalley/gopro-deblur`"
                            )
                        with dpg.tooltip(EDataset.blur_dataset):  # tag of the parent
                            dpg.add_text(
                                "Dataset from `https://www.kaggle.com/datasets/kwentar/blur-dataset"
                            )

        self.manager = SyncManager(ctx=ctx)
        self.manager.start()

        c_window = Console(self, label="Console")
        self.add_window(c_window)
        self.console_handler = c_window.console_handler  # TODO better

        # process related stuff
        self.downloading = AtomicBool(ctx, False)
        self.work_queue = ctx.Queue(maxsize=1)
        self.worker = ctx.Process(
            target=_worker_entry_point,
            args=(self.work_queue, self.downloading),
        )

    def add_window(self, panel: Panel) -> None:
        self._panels[panel.tag] = panel
        self._panels[panel.tag].submit()

    def remove_window(self, tag: str) -> None:
        if self._panels[tag] is not None:
            del self._panels[tag]
            dpg.delete_item(tag)

    def run_render_loop(self) -> None:
        while dpg.is_dearpygui_running():
            # ...
            dpg.render_dearpygui_frame()

    def __enter__(self):
        dpg.show_viewport()
        # select bootstrap window for display
        dpg.set_primary_window(Window._bootstrap_tag, True)
        self.worker.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> bool:
        if hasattr(self, "f_lock"):
            self._remove_file_lock()

        dpg.destroy_context()
        if exc_type is not None:
            traceback.print_exception(exc_type, value=exc_value, tb=tb)

        self.downloading.request_close()
        self.worker.join(timeout=0.1)
        if self.worker.is_alive():
            self.worker.kill()
        
        self.manager.shutdown()
        return (
            False  # false to throw the exception exc_value (if any), true to swallow it
        )

    def _on_resize(self):
        vp_client, vp_window = _viewport_dims()

        for tag, panel in self._panels.items():
            panel.on_resize(vp_window, vp_client)

        # hack: on Windows, the sequence "Maximize" and "Click and Drag Titlebar" breaks everything
        if self._manual_resize:
            self._manual_resize = False
        else:
            self._manual_resize = True
            dpg.set_viewport_width(vp_window.width + 1)

    def _on_dataset_path(self, sender, app_data):
        logging.info("Current Path: %s", dpg.get_value(app_data))

    def _on_dataset_selection(self, sender, app_data):
        if self.downloading.load():
            logging.error("Already occupied...")
            return

        if not self._state.dataset_path_eelected:
            logging.error("Before Choosing a Dataset, select the Download Path")
            return

        logging.info("Chosen dataset %s", str(sender))
        match str(sender):
            case EDataset.gopro_blur.value:
                self.work_queue.put(
                    (
                        data.datasets_make_available_gopro_deblur,
                        [
                            self._state.dataset_path,
                        ],
                        {},
                    )
                )
            case EDataset.blur_dataset.value:
                self.work_queue.put(
                    (
                        data.datasets_make_available_blur_dataset,
                        [
                            self._state.dataset_path,
                        ],
                        {},
                    )
                )
            case "test":
                self.work_queue.put((data.test_work, ["str"], {"key": "sdfds"}))
            case _:
                raise ValueError("Unrecognized dataset selected. How?")
        self.downloading.set()

    # Two methods: First therese python webview, the second is OS Specific scripting
    def _ask_directory(self):
        """If nothing is selected it uses home"""
        if self.downloading.load():
            logging.error("Cannot change directory while downloading a dataset")
            return

        path = None
        if platform.system() == "Windows":
            path = (
                str(self._state.dataset_path)
                if self._state.dataset_path.exists()
                else '"' + os.environ["USERPROFILE"] + '"'
            )
            command = io.BytesIO(
                b". $Profile\x0d\x0a"
                b'$MyFunctions = "Function Get-Folder(`$InitialDirectory) {'
                b"[void] [System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms');"
                b"`$FolderBrowserDialog = New-Object System.Windows.Forms.FolderBrowserDialog;"
                b"`$FolderBrowserDialog.RootFolder = 'MyComputer';"
                b"if (`$InitialDirectory) { `$FolderBrowserDialog.SelectedPath = `$InitialDirectory };"
                b"[void] `$FolderBrowserDialog.ShowDialog();"
                b"return `$FolderBrowserDialog.SelectedPath;"
                b'}"\x0d\x0a'
                b". { Invoke-Expression $MyFunctions };\x0d\x0a"
                b"Get-Folder " + path.encode("utf-8") + b"\x0d\x0a"
            )
            with subprocess.Popen(
                ["powershell", "-NoProfile"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            ) as ps:
                copyfileobj(command, ps.stdin)
                ps.stdin.close()  # fire command!
                lines = ps.stdout.readlines()
                path = Path(lines[-2].decode(encoding="utf-8").removesuffix("\r\n"))
        elif platform.system() == "Darwin":
            # TODO TEST how
            # https://developer.apple.com/library/archive/documentation/LanguagesUtilities/Conceptual/MacAutomationScriptingGuide/PromptforaFileorFolder.html
            path = (
                str(self._state.dataset_path)
                if self._state.dataset_path.exists()
                else os.environ["HOME"]
            )
            script = (
                'set chosenFolder to POSIX path of (choose folder with prompt "Select a folder:" default location POSIX file "{initial_directory}")\n'
                "return chosenFolder"
            ).format(initial_directory=path)

            try:
                result = subprocess.run(
                    ["osascript", "-e", script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                if result.returncode == 0:
                    path = Path(result.stdout.strip())
                else:
                    logging.error("Error during folder selection: %s", result.stderr)
                    path = Path(
                        os.environ["HOME"]
                    )  # Default to home directory on error
            except Exception as e:
                logging.error("Exception while running osascript: %s", str(e))
                path = Path(
                    os.environ["HOME"]
                )  # Default to home directory on exception
        else:  # assumes platform is an X11 linux
            path = (
                str(self._state.dataset_path)
                if self._state.dataset_path.exists()
                else '"' + os.environ["HOME"] + '"'
            )
            script = (
                "#!/bin/bash\n"
                "get_folder() {\n"
                '    local initial_directory="$1"\n'
                "    local chosen_folder\n"
                "    if command -v zenity &>/dev/null; then\n"
                '        chosen_folder=$(zenity --file-selection --directory --title="Select a Folder" --filename="$initial_directory")\n'
                "    elif command -v kdialog &>/dev/null; then\n"
                '        chosen_folder=$(kdialog --getexistingdirectory "$initial_directory")\n'
                "    else\n"
                "        echo \"Error: Neither 'zenity' nor 'kdialog' is installed. Please install one to use this script.\" >&2\n"
                "        return 1\n"
                "    fi\n"
                '    echo "$chosen_folder"\n'
                "}\n"
                "# Main script\n"
                'user_home="$HOME"  # Get the user\'s home directory\n'
                'echo "Initial directory: $user_home"\n'
                "# Call the folder chooser function\n"
                'selected_folder=$(get_folder "$user_home")\n'
                'if [ -n "$selected_folder" ]; then\n'
                '    echo "$selected_folder"\n'
                "else\n"
                '    echo "$user_home"\n'
                "fi\n"
            )
            script_path = Path.cwd() / "sh-dir.sh"
            if not script_path.exists():
                with script_path.open("w", encoding="utf-8", newline="\n") as f:
                    f.write(script)
            command = io.BytesIO(b"./sh-dir.sh " + path.encode(encoding="utf-8"))
            with subprocess.Popen(
                ["sh"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=Path.cwd()
            ) as sh:
                copyfileobj(command, sh.stdin)
                sh.stdin.close()  # fire command!
                # if stderr is not empty raise error softawre not installed
                if len(sh.stderr.read()) > 0:
                    raise ValueError(
                        "Error (probably neither `zenity` nor `kdialog` (kdtools) are installed)"
                    )
                # its the last one right? Test this
                lines = sh.stdout.readlines()
                path = Path(lines[-1].decode(encoding="utf-8").removesuffix("\n"))

        if path.exists():
            if hasattr(self, "f_lock") and self.f_lock is not None:
                self._remove_file_lock()

            # create lock file (delete old one if existing)
            p_lock = path / Window._lock_file_name
            self._state.dataset_path = path
            self._state.dataset_path_eelected = True

            self.f_lock = open(p_lock, "w")
            pl.lock(self.f_lock, pl.LockFlags.EXCLUSIVE)  # lock the file

            dpg.set_value(Window._dataset_path_tag, str(path))
        logging.info("Dataset path: %s", self._state.dataset_path)
        return path

    def _remove_file_lock(self):
        pl.unlock(self.f_lock)
        self.f_lock.close()
        (self._state.dataset_path / Window._lock_file_name).unlink()
        del self.f_lock
