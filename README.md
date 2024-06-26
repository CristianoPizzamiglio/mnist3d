---
annotations_creators:
- expert-generated
language_creators:
- found
language:
- en
license:
- mit
multilinguality:
- monolingual
size_categories:
- 10K<n<100K
source_datasets:
- extended|mnist
task_categories:
- other
pretty_name: MNIST3D
dataset_info:
  features:
  - name: point_cloud
    dtype: float16
  - name: label
    dtype:
      class_label:
        names:
          '0': '0'
          '1': '1'
          '2': '2'
          '3': '3'
          '4': '4'
          '5': '5'
          '6': '6'
          '7': '7'
          '8': '8'
          '9': '9'
  config_name: mnist
  splits:
  - name: train
    num_bytes: 69540256
    num_examples: 60000
  - name: test
    num_bytes: 11590256
    num_examples: 10000
  download_size: 68593760
  dataset_size: 81130512
---

![7](docs/7.png)

# Dataset Card for MNIST3D

## Dataset Description

### Dataset Summary

The MNIST3D dataset consists of 70,000 point clouds of handwritten digits generated 
by converting the images from the original [MNIST](https://huggingface.co/datasets/mnist) dataset.
Each point cloud has 193 points.

### Languages

English

## Dataset Structure

### Data Splits

The data is split into training and test set. The original data split of the MNIST 
dataset is preserved.

## Dataset Creation

### Curation Rationale

The MNIST3D dataset serves as an accessible entry point for those interested in 
applying machine learning models to 3D point cloud classification tasks while spending 
minimal efforts on preprocessing and formatting.

### Methods

1. The MNIST dataset is loaded from the `keras.datasets` module.
2. The pixel intensity distribution of the MNIST images is analyzed. As shown in the 
   plot below, intensities are clustered toward 0 and 255. A threshold equal to 128 
   is chosen.

![hist](docs/test_image_pixel_intensity_distribution_0.png)

3. Images are binarized. Intensities greater than the threshold are converted to 1, 
   the remaining ones to 0.
4. Images are converted to point clouds. Pixels with non-zero intensity are 
   considered as points where the intensity acts as the z-coordinate. The `x` and 
   `y` coordinates are normalized in the 0-1 range. The `z` coordinate is converted 
   to 0.
5. The total number of points is set to 193, which is the maximum (outliers excluded)
   of the distribution of non-zero intensities in the binarized images (see boxplot 
   below).

![boxplot](docs/non_zero_intensity_distribution_boxplot.png)

6. Gaussian noise with mean set to zero and standard deviation equal to 0.01 is 
   added to the three dimensions.

The dataset is generated by running `main.py` and written to disk as NumPy files 
(`npy` format). Plots are drawn by running `plotter.py`, while `size_calculator.py` 
allows to compute the dataset size.

## Additional Information

### Dataset Curators

Cristiano Pizzamiglio

### Licensing Information

MIT Licence

### Acknowledgment 

[Mariona Carós](https://datascienceub.medium.com/pointnet-implementation-explained-visually-c7e300139698)
