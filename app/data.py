import kaggle

from pathlib import Path
from typing import NamedTuple
import sys
import utils
from utils import AtomicBool


def datasets_make_available_gopro_deblur(datasets_path: Path, writeable, atomic_flag: AtomicBool) -> None:
    original_stdout = sys.stdout
    try:
        sys.stdout = writeable
        atomic_flag.set(True)
        kaggle.api.dataset_download_files(
            "rahulbhalley/gopro-deblur", path=datasets_path, quiet=False, unzip=True
        )
    finally:
        atomic_flag.set(False)
        sys.stdout = original_stdout
