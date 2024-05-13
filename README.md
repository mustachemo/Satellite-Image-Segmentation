# Turion Space assessment

This repository contains experiments and implementations for uncertainty quantification in deep neural networks applied to object detection tasks. The goal is to leverage dropout layers and multiple stochastic forward passes to measure prediction uncertainty.

## Table of Contents

- [Turion Space assessment](#turion-space-assessment)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Datasets](#datasets)
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

## Datasets

The experiments are conducted on the [SpaceNet](https://spacenetchallenge.github.io/) dataset.

## Acknowledgements

This project is inspired by the work of [Yarin Gal](http://www.cs.ox.ac.uk/people/yarin.gal/website/index.html)
and [A review of uncertainty quantification in deep learning: Techniques, applications and challenges](https://www.sciencedirect.com/science/article/pii/S1566253521001081).

For the Turion Space assessment, I have implemented the following experiments:

- [x] Dropout layers in object detection models
- [ ] Monte Carlo dropout
- [ ] Ensemble methods
- [ ] Bayesian neural networks
- [ ] Calibration metrics
- [ ] Evaluation on SpaceNet dataset
- [ ] Report
