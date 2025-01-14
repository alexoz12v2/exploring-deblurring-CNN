import kaggle

from pathlib import Path
from typing import NamedTuple


def datasets_make_available_gopro_deblur(datasets_path: Path) -> Path:
    ret = datasets_path / "gopro-deblur"
    if not ret.exists():
        kaggle.api.dataset_download_files("rahulbhalley/gopro-deblur")

    return ret
