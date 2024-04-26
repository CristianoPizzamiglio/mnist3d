from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from io_ import import_dataset
from main import compute_binary_intensities, ActivePixelStats
from parameters import Parameters, import_parameters


def main(parameters: Parameters) -> None:
    """
    Plot image pixel intensity distributions, active pixel count boxplot, and images and
    corresponding point clouds.

    Parameters
    ----------
    parameters : Parameters

    """
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    plot_image_pixel_intensity_distributions(
        train_images, train_labels, title="Train Images"
    )
    plot_image_pixel_intensity_distributions(
        test_images, test_labels, title="Test Images"
    )

    binary_intensities = compute_binary_intensities(
        train_images, parameters.pixel_intensity_threshold
    )
    active_pixel_stats = ActivePixelStats(binary_intensities)
    plot_active_pixel_count_boxplot(active_pixel_stats)

    train_point_clouds, train_labels, test_point_clouds, test_labels = import_dataset(
        dir_path=Path("../../dataset")
    )
    label_count = 10
    label_to_indices = {
        index: np.where(train_labels == index)[0] for index in range(label_count)
    }
    indices = [np.random.choice(indices) for indices in label_to_indices.values()]
    for label, index in enumerate(indices):
        plot_point_cloud_image(train_point_clouds[index], train_images[index], label)


def plot_point_cloud_image(
    point_cloud: np.ndarray, image: np.ndarray, label: int
) -> None:
    """
    Plot point cloud and corresponding image.

    Parameters
    ----------
    point_cloud : np.ndarray
    image : np.ndarray
    label : int

    """
    figure = plt.figure(figsize=(12, 6))

    axis_point_cloud = figure.add_subplot(121, projection="3d")
    axis_point_cloud.scatter(
        point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=40
    )
    axis_point_cloud.set_xlim(0.0, 1.0)
    axis_point_cloud.set_ylim(0.0, 1.0)
    axis_point_cloud.set_zlim(-0.1, 0.1)
    axis_point_cloud.view_init(elev=-90, azim=-85)

    axis_image = figure.add_subplot(122)
    axis_image.imshow(image, cmap="gray")
    axis_image.set_xticks([])
    axis_image.set_yticks([])

    figure.suptitle(f"Label: {label}")
    plt.tight_layout()
    plt.show()


def plot_image_pixel_intensity_distributions(
    images: np.ndarray, labels: np.ndarray, title: str
) -> None:
    """
    Plot image pixel intensity distributions.

    Parameters
    ----------
    images : np.ndarray
    labels : np.ndarray
    title : str

    """
    label_count = 10
    figure, axis = plt.subplots(2, 5, figsize=(12, 5))
    for label in range(label_count):
        indices = np.where(labels == label)[0]
        intensities = images[indices].flatten()
        i = 0 if label < label_count // 2 else 1
        j = label if label < label_count // 2 else (label - label_count // 2)
        axis[i, j].hist(intensities, bins=40)
        axis[i, j].set_title(label)
        axis[i, j].set_xticks(np.arange(0, 256, 85))
        axis[i, j].get_yaxis().set_visible(False)

    figure.suptitle(f"{title} - Pixel Intensity Distributions")
    plt.tight_layout()
    plt.show()


def plot_active_pixel_count_boxplot(active_pixel_stats: ActivePixelStats) -> None:
    """
    Plot active pixel counts boxplot.

    Parameters
    ----------
    active_pixel_stats : ActivePixelStats

    """
    figure, axis = plt.subplots(figsize=(5, 6))
    plt.boxplot(active_pixel_stats.counts)

    median = np.median(active_pixel_stats.median)
    minimum = np.min(active_pixel_stats.minimum)
    maximum = np.max(active_pixel_stats.maximum)

    axis.annotate(f"Median: {median}", xy=(1, median), xytext=(1.1, median))
    axis.annotate(f"Minimum: {minimum}", xy=(1, minimum), xytext=(1.1, minimum))
    axis.annotate(f"Maximum: {maximum}", xy=(1, maximum), xytext=(1.1, maximum))

    plt.xlabel("Intensity = 1")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    parameters_ = import_parameters()
    main(parameters_)
