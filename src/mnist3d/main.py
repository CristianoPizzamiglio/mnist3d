from __future__ import annotations

from dataclasses import dataclass, InitVar, field
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from io_ import export_dataset
from parameters import Parameters, import_parameters

np.random.seed(42)

IMAGE_SIZE = 28


def main(
    parameters: Parameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the original MNIST dataset and convert images to point clouds.

    Parameters
    ----------
    parameters : Parameters

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

    """
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    binary_intensities = compute_binary_intensities(
        train_images, parameters.pixel_intensity_threshold
    )
    point_count = compute_point_count(binary_intensities)

    train_point_clouds = convert_images_to_point_clouds(
        train_images,
        point_count,
        parameters.pixel_intensity_threshold,
        parameters.noise_standard_deviation,
    )
    test_point_clouds = convert_images_to_point_clouds(
        test_images,
        point_count,
        parameters.pixel_intensity_threshold,
        parameters.noise_standard_deviation,
    )
    return train_point_clouds, train_labels, test_point_clouds, test_labels


@dataclass
class ActivePixelStats:
    """
    Active pixel (i.e. intensity = 1) statistics.

    Parameters
    ----------
    binary_intensities : np.ndarray
        Binary pixel intensities (i.e. 0 or 1).

    Attributes
    ----------
    counts : np.ndarray
    first_quartile : int
    third_quartile : int
    median : int
    iqr : float
        Interquartile range.
    minimum : int
        Outliers excluded.
    maximum : int
        Outliers excluded.

    """

    binary_intensities: InitVar[np.ndarray]
    counts: np.ndarray = field(init=False)
    first_quartile: int = field(init=False)
    third_quartile: int = field(init=False)
    median: int = field(init=False)
    minimum: int = field(init=False)
    maximum: int = field(init=False)

    def __post_init__(self, binary_intensities: np.ndarray) -> None:
        self.counts = np.sum(binary_intensities, axis=1).astype(int)
        self.first_quartile = np.percentile(self.counts, 25).astype(int)
        self.third_quartile = np.percentile(self.counts, 75).astype(int)
        self.median = np.median(self.counts).astype(int)
        self.iqr = self.third_quartile - self.first_quartile
        iqr_factor = 1.5
        self.minimum = self.counts[
            self.counts >= self.first_quartile - iqr_factor * self.iqr
        ].min()
        self.maximum = self.counts[
            self.counts <= self.third_quartile + iqr_factor * self.iqr
        ].max()


def create_xy_grid(image_size: int) -> np.ndarray:
    """
    Create x-y grid.

    Parameters
    ----------
    image_size : int
        Pixel count (the image is squared).

    Returns
    -------
    np.ndarray

    """
    x = np.tile(np.linspace(0.0, 1.0, image_size), image_size)
    y = np.repeat(np.linspace(0.0, 1.0, image_size), image_size)
    return np.column_stack((x, y))


def convert_images_to_point_clouds(
    images: np.ndarray,
    point_count: int,
    pixel_intensity_threshold: int,
    noise_standard_deviation: float,
) -> np.ndarray:
    """
    Convert images to point clouds.

    Parameters
    ----------
    images : np.ndarray
    point_count : int
    pixel_intensity_threshold : int
    noise_standard_deviation : float

    Returns
    -------
    np.ndarray

    """
    binary_intensities = compute_binary_intensities(images, pixel_intensity_threshold)

    xy_grid = create_xy_grid(image_size=IMAGE_SIZE)
    xy_grids = np.tile(xy_grid, (images.shape[0], 1, 1))
    point_clouds = np.concatenate(
        (xy_grids, binary_intensities[:, :, np.newaxis]), axis=2
    )

    point_clouds_resized = np.array(
        [resize_point_cloud(point_cloud, point_count) for point_cloud in point_clouds]
    )
    point_clouds_resized_noisy = np.array(
        [
            add_noise(point_cloud, noise_standard_deviation)
            for point_cloud in point_clouds_resized
        ]
    )
    return point_clouds_resized_noisy.astype(np.float16)


def compute_binary_intensities(
    images: np.ndarray, pixel_intensity_threshold: int
) -> np.ndarray:
    """
    Compute binary pixel intensities (i.e. 0 or 1).

    Parameters
    ----------
    images : np.ndarray
    pixel_intensity_threshold : int

    Returns
    -------
    np.ndarray

    """
    images = (images > pixel_intensity_threshold).astype(int)
    return images.reshape(images.shape[0], images.shape[1] * images.shape[2])


def compute_point_count(binary_intensities: np.ndarray) -> int:
    """
    Compute the number of points as the maximum of the boxplot (excluding any outliers).

    Parameters
    ----------
    binary_intensities : np.ndarray

    Returns
    -------
    int

    """
    active_pixel_stats = ActivePixelStats(binary_intensities)
    return active_pixel_stats.maximum


def resize_point_cloud(point_cloud: np.ndarray, point_count: int) -> np.ndarray:
    """
    Resize point cloud to have `point_count` points.

    Parameters
    ----------
    point_cloud :  p.ndarray
    point_count : int

    Returns
    -------
    np.ndarray

    """
    point_cloud = point_cloud[point_cloud[:, 2] > 0]
    if len(point_cloud) < point_count:
        missing_count = point_count - len(point_cloud)
        indices = np.random.choice(len(point_cloud), missing_count)
        return np.concatenate((point_cloud, point_cloud[indices, :]), axis=0)
    elif len(point_cloud) > point_count:
        indices = np.random.choice(len(point_cloud), point_count)
        return point_cloud[indices, :]
    else:
        return point_cloud


def add_noise(point_cloud: np.ndarray, standard_deviation: float) -> np.ndarray:
    """
    Add gaussian noise.

    Parameters
    ----------
    point_cloud : np.ndarray
    standard_deviation : float

    Returns
    -------
    np.ndarray

    """
    point_cloud[:, 2] = point_cloud[:, 2] - 1.0
    noise = np.random.normal(0.0, standard_deviation, point_cloud.shape)
    return point_cloud + noise


if __name__ == "__main__":
    parameters_ = import_parameters()
    train_point_clouds, train_labels, test_point_clouds, test_labels = main(parameters_)
    export_dataset(
        train_point_clouds,
        train_labels,
        test_point_clouds,
        test_labels,
        dir_path=Path("../../dataset"),
    )
