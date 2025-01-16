from multiprocessing.context import SpawnContext
import os
from pathlib import Path

if __name__ == "__main__" and os.environ.get("BAZEL_FIX_DIR"):
    main_module_path = Path(__file__).parent.parent
    print(main_module_path)
    os.chdir(main_module_path)

import traceback
from types import TracebackType
import torch
from absl import app, flags, logging
from lib.layers.convir_layers import build_net, ConvIR
import kaggle

from multiprocessing import Manager
import multiprocessing as mp
import window
import sys

# command line arguments:
FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    name="window",
    default=None,
    help="Whether to run the app in window mode (meaning the command line tool already generated a mode3)",
)


def sayHello() -> None:
    logging.info(Path.cwd())
    logging.info(f"torch version: {torch.__version__}")
    if torch.cuda.is_available():
        logging.info("CUDA is available")
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logging.info(f"Current GPU: {torch.cuda.current_device()}")
        logging.info(
            f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}"
        )
    else:
        logging.info("CUDA is not available")


def kaggle_download_and_extract_zip(dataset_name: str, output_path: Path) -> None:
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        dataset_name, path=str(output_path), quiet=False, unzip=True
    )


def logging_exception_hook(
    exc_type: type[BaseException] | None,
    exc_value: BaseException | None,
    tb: TracebackType | None,
) -> None:
    # alternative: logging.exception
    # Log the exception with the full traceback
    logging.exception(
        "Exception %s [%s] \n\tTraceback:\n%s",
        exc_value,
        exc_type,
        traceback.format_tb(tb),
    )


def main(args: list[str]) -> None:
    # Set up the redirection
    ctx: SpawnContext = mp.get_context("spawn")
    sys.excepthook = logging_exception_hook
    if FLAGS.window:
        with window.Window("EDCNN", ctx=ctx, width=800, height=600) as w:
            w.run_render_loop()
    else:
        sayHello()
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        model: ConvIR = build_net().to(device)
        file_id = "1y_wQ5G5B65HS_mdIjxKYTcnRys_AGh5v"
        target_folder = "data"
        # google_drive_download_and_extract_zip(file_id, target_folder)
        kaggle_download_and_extract_zip(
            "rahulbhalley/gopro-deblur", Path.cwd() / target_folder
        )


if __name__ == "__main__":
    app.run(main)
