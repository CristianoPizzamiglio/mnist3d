from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def import_dataset(
    dir_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Import dataset.

    Parameters
    ----------
    dir_path : Path

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

    """
    return (
        np.load(Path(rf"{dir_path}\train_point_clouds.npy")),
        np.load(Path(rf"{dir_path}\train_labels.npy")),
        np.load(Path(rf"{dir_path}\test_point_clouds.npy")),
        np.load(Path(rf"{dir_path}\test_labels.npy")),
    )


def export_dataset(
    train_point_clouds: np.ndarray,
    train_labels: np.ndarray,
    test_point_clouds: np.ndarray,
    test_labels: np.ndarray,
    dir_path: Path,
) -> None:
    """
    Export dataset as NumPy arrays.

    Parameters
    ----------
    train_point_clouds : np.ndarray
    train_labels : np.ndarray
    test_point_clouds : np.ndarray
    test_labels : np.ndarray
    dir_path : Path

    """
    np.save(Path(rf"{dir_path}\train_point_clouds"), train_point_clouds)
    np.save(Path(rf"{dir_path}\train_labels"), train_labels)
    np.save(Path(rf"{dir_path}\test_point_clouds"), test_point_clouds)
    np.save(Path(rf"{dir_path}\test_labels"), test_labels)
