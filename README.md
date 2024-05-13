# Turion Space assessment

This repository contains experiments and implementations for uncertainty quantification in deep neural networks applied to object detection tasks. The goal is to leverage dropout layers and multiple stochastic forward passes to measure prediction uncertainty.

## Table of Contents

- [Turion Space assessment](#turion-space-assessment)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
- [Satellite Dataset](#satellite-dataset)
  - [Dataset Details](#dataset-details)
  - [Bounding Boxes](#bounding-boxes)
  - [Additional Resources](#additional-resources)
  - [Acknowledgements](#acknowledgements)

## Introduction

Uncertainty quantification is crucial in critical applications like space object detection. This project aims to implement techniques to measure uncertainty in object detection models using dropout layers.

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/uncertainty-quantification-object-detection.git
cd uncertainty-quantification-object-detection
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the experiments:

```bash
python experiments.py
```

# Satellite Dataset

This repository contains a satellite dataset for object detection and segmentation using both synthetic and real satellite images. The dataset includes 3116 images, masks with size 1280x720, and bounding boxes of both synthetic and real satellite images. Each satellite is segmented into at most 3 parts, including body, solar panel, and antenna, represented by three colors: green, red, and blue.

![Example sample](figure1.png)

### Dataset Details

- Images with index 0-1002 have fine masks, while images from index 1003-3116 have coarse masks.
- The dataset is divided into two parts:
  - Training data includes 403 fine masks from index 0-402 and 2114 coarse masks from index 1003-3116.
  - The validation dataset includes 600 images with fine masks indexed from 403 to 1002.

### Bounding Boxes

The file `all_bbox.txt` includes bounding boxes of all satellites inside the dataset based on segmentation masks. It's in the form of a dictionary with the index of images as the key. Each bounding box has the format [max_x, max_y, min_x, min_y].

### Additional Resources

- Link to the paper: [A Spacecraft Dataset for Detection, Segmentation and Parts Recognition](https://arxiv.org/abs/2106.08186)
- Link to the dataset: [https://github.com/Yurushia1998/SatelliteDataset](https://github.com/Yurushia1998/SatelliteDataset)

## Acknowledgements

This project is inspired by the work of [Yarin Gal](http://www.cs.ox.ac.uk/people/yarin.gal/website/index.html)
and [A review of uncertainty quantification in deep learning: Techniques, applications and challenges](https://www.sciencedirect.com/science/article/pii/S1566253521001081).

For the Turion Space assessment, I have implemented the following experiments:

- [ ] Dropout layers in object detection models
- [ ] Monte Carlo dropout
- [ ] Ensemble methods
- [ ] Bayesian neural networks
- [ ] Calibration metrics
- [ ] Evaluation on SpaceNet dataset
- [ ] Report
