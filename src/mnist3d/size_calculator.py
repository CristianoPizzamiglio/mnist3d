from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import List

DIR_PATH = Path("../../dataset")


def compute_dataset_sizes() -> List[int]:
    """
    Compute the dataset size in bytes.

    Returns
    -------
    List[int]

    """
    return [
        os.path.getsize(Path(DIR_PATH, filename)) for filename in os.listdir(DIR_PATH)
    ]


def compute_download_size() -> int:
    """
    Compute the download size in bytes.

    Returns
    -------
    int

    """
    archive_path = Path(DIR_PATH, "archive.zip")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_LZMA) as zipf:
        for filename in os.listdir(DIR_PATH):
            zipf.write(os.path.join(DIR_PATH, filename), arcname=filename)
    return os.path.getsize(archive_path)


if __name__ == "__main__":
    dataset_sizes = compute_dataset_sizes()
    dataset_size = sum(dataset_sizes)
    download_size = compute_download_size()
