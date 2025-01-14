import dearpygui.dearpygui as dpg
from pathlib import Path
from absl import logging
import traceback

from enum import Enum
from types import TracebackType
from typing import Optional, Type, NamedTuple, Tuple
from dataclasses import dataclass

import platform
import os
import subprocess
import io
from shutil import copyfileobj

class EDataset(str, Enum):
    gopro_blur = "kaggle goprodeblur"


# class EWindowShow


def _viewport_dims() -> Tuple[int, int, int, int]:
    vp_window_width = dpg.get_viewport_width()
    vp_window_height = dpg.get_viewport_height()
    vp_client_width = min(dpg.get_viewport_client_width(), vp_window_width)
    vp_client_height = min(dpg.get_viewport_client_height(), vp_window_height)
    return (vp_client_width, vp_client_height, vp_window_width, vp_window_height)


class DPGConsoleHandler(logging.PythonHandler):
    def __init__(self, console_tag: str, num_lines: int):
        super().__init__()
        self.console_tag = (
            console_tag  # window tag inside which add_text will be called
        )
        self.num_lines = num_lines
        self.line_tags = []

    def emit(self, record):
        log_entry = self.format(record)
        self.line_tags.append(dpg.add_text(log_entry, parent=self.console_tag))
        if len(self.line_tags) > self.num_lines:
            to_remove = self.line_tags.pop(0)
            dpg.delete_item(to_remove)


@dataclass
class WindowState:
    dataset_path: Path
    dataset_path_eelected: bool


class Window:
    _bootstrap_tag = "Bootstrap Window"
    _console_tag = "Console Window"

    # https://dearpygui.readthedocs.io/en/latest/documentation/viewport.html
    def __init__(self, title: str, *, width: int, height: int):
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
        self.console_handler = DPGConsoleHandler(Window._console_tag, 100)
        self._manual_resize = False

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
                    with dpg.file_dialog(
                        directory_selector=True,
                        show=False,
                        tag="dataset_path_file_dialog",
                        width=600,
                        height=400,
                        callback=lambda sender, app_data, user_data: user_data._on_dataset_path(sender, app_data),
                        user_data=self,
                        cancel_callback=lambda: logging.info("Path selection canceled"),
                    ):
                        pass
                    dpg.add_button(
                        label="Choose Dataset Path...",
                        # callback=lambda: dpg.show_item("dataset_path_file_dialog"),
                        callback=lambda sender, app_data, user_data: user_data._ask_directory(),
                        user_data=self
                    )
                    dpg.add_text(
                        label="Path",
                        default_value=str(self._state.dataset_path),
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
                        with dpg.tooltip(EDataset.gopro_blur):  # tag of the parent
                            dpg.add_text(
                                "Dataset from `https://www.kaggle.com/datasets/rahulbhalley/gopro-deblur`"
                            )

            with dpg.window(
                label="Console",
                tag=Window._console_tag,
                horizontal_scrollbar=True,
                no_collapse=True,
                no_focus_on_appearing=True,
                pos=(0, console_position_winrel),
                height=console_height,
                width=vp_client_width,
                no_close=True,
                no_move=True,
                no_resize=True,
            ):
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

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> bool:
        dpg.destroy_context()
        if exc_type is not None:
            traceback.print_exception(exc_type, value=exc_value, tb=tb)

        return (
            False  # false to throw the exception exc_value (if any), true to swallow it
        )

    def _adjust_console(self, vp_client_width, vp_client_height):
        dpg.set_item_pos(Window._console_tag, [0, 0.7 * vp_client_height])
        dpg.set_item_width(Window._console_tag, vp_client_width)
        dpg.set_item_height(Window._console_tag, int(0.3 * vp_client_height))

    def _on_resize(self):
        vp_client_width, vp_client_height, vp_window_width, vp_window_height = (
            _viewport_dims()
        )
        self._adjust_console(vp_client_width, vp_client_height)

        # hack: on Windows, the sequence "Maximize" and "Click and Drag Titlebar" breaks everything
        if self._manual_resize:
            self._manual_resize = False
        else:
            self._manual_resize = True
            dpg.set_viewport_width(vp_window_width + 1)

    def _on_dataset_path(self, sender, app_data):
        logging.info("Current Path: %s", dpg.get_value(app_data))

    def _on_dataset_selection(self, sender, app_data):
        logging.info("Dataset selected: %s", repr(sender))
        match str(sender):
            case EDataset.gopro_blur.value:
                logging.info("Dataset recognized")
            case _:
                raise ValueError("Unrecognized dataset selected. How?")
    
    def _ask_directory(self):
        """If nothing is selected it uses home"""
        path = None
        if platform.system() == "Windows":
            path = str(self._state.dataset_path) if self._state.dataset_path.exists() else '"' + os.environ["USERPROFILE"] + '"'
            command = io.BytesIO(
                b". $Profile\x0D\x0A"
                b"$MyFunctions = \"Function Get-Folder(`$InitialDirectory) {"
                b"[void] [System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms');"
                b"`$FolderBrowserDialog = New-Object System.Windows.Forms.FolderBrowserDialog;"
                b"`$FolderBrowserDialog.RootFolder = 'MyComputer';"
                b"if (`$InitialDirectory) { `$FolderBrowserDialog.SelectedPath = `$InitialDirectory };"
                b"[void] `$FolderBrowserDialog.ShowDialog();"
                b"return `$FolderBrowserDialog.SelectedPath;"
                b"}\"\x0D\x0A"
                b". { Invoke-Expression $MyFunctions };\x0D\x0A"
                b"Get-Folder " + path.encode("utf-8") + b"\x0D\x0A"
            )
            with subprocess.Popen(["powershell", "-NoProfile"], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as ps:
                copyfileobj(command, ps.stdin)
                ps.stdin.close() # fire command!
                lines = ps.stdout.readlines()
                path = Path(lines[-2].decode(encoding='utf-8').removesuffix("\r\n"))
        elif platform.system() == "Darwin":
            # TODO TEST how
            # https://developer.apple.com/library/archive/documentation/LanguagesUtilities/Conceptual/MacAutomationScriptingGuide/PromptforaFileorFolder.html
            path = str(self._state.dataset_path) if self._state.dataset_path.exists() else os.environ["HOME"]
            script = (
                'set chosenFolder to POSIX path of (choose folder with prompt "Select a folder:" default location POSIX file "{initial_directory}")\n'
                'return chosenFolder'
            ).format(initial_directory=path)
    
            try:
                result = subprocess.run(
                    ['osascript', '-e', script],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if result.returncode == 0:
                    path = Path(result.stdout.strip())
                else:
                    logging.error("Error during folder selection: %s", result.stderr)
                    path = Path(os.environ["HOME"])  # Default to home directory on error
            except Exception as e:
                logging.error("Exception while running osascript: %s", str(e))
                path = Path(os.environ["HOME"])  # Default to home directory on exception
        else: # assumes platform is an X11 linux
            path = str(self._state.dataset_path) if self._state.dataset_path.exists() else '"' + os.environ["HOME"] + '"'
            script = (
                "#!/bin/bash\n"
                "get_folder() {\n"
                "    local initial_directory=\"$1\"\n"
                "    local chosen_folder\n"
                "    if command -v zenity &>/dev/null; then\n"
                "        chosen_folder=$(zenity --file-selection --directory --title=\"Select a Folder\" --filename=\"$initial_directory\")\n"
                "    elif command -v kdialog &>/dev/null; then\n"
                "        chosen_folder=$(kdialog --getexistingdirectory \"$initial_directory\")\n"
                "    else\n"
                "        echo \"Error: Neither 'zenity' nor 'kdialog' is installed. Please install one to use this script.\" >&2\n"
                "        return 1\n"
                "    fi\n"
                "    echo \"$chosen_folder\"\n"
                "}\n"
                "# Main script\n"
                "user_home=\"$HOME\"  # Get the user's home directory\n"
                "echo \"Initial directory: $user_home\"\n"
                "# Call the folder chooser function\n"
                "selected_folder=$(get_folder \"$user_home\")\n"
                "if [ -n \"$selected_folder\" ]; then\n"
                "    echo \"$selected_folder\"\n"
                "else\n"
                "    echo \"$user_home\"\n"
                "fi\n"
            )
            script_path = Path.cwd() / 'sh-dir.sh'
            if not script_path.exists():
                with script_path.open('w', encoding='utf-8', newline='\n') as f:
                    f.write(script)
            command = io.BytesIO(b"./sh-dir.sh " + path.encode(encoding="utf-8"))
            with subprocess.Popen(['sh'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=Path.cwd()) as sh:
                copyfileobj(command, sh.stdin)
                sh.stdin.close() # fire command!
                # if stderr is not empty raise error softawre not installed
                if len(sh.stderr.read()) > 0:
                    raise ValueError("Error (probably neither `zenity` nor `kdialog` (kdtools) are installed)")
                # its the last one right? Test this
                lines = sh.stdout.readlines()
                path = Path(lines[-1].decode(encoding="utf-8").removesuffix("\n"))

        self._state.dataset_path = path
        self._state.dataset_path_eelected = True
        logging.info("Dataset path: %s", self._state.dataset_path)
        return path