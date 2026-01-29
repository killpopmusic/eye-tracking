# Eye Tracking Computing Environment

## About
The project provides a modular environment that enables various tests regarding eye tracking algorithms utilizing facial landmark coordinates. Different approaches to gaze prediction can be found in dedicated branches.

## Features
- Testing different approaches to the eye tracking task: gaze point regression and gaze ROI classification;
- Complex database analysis tools in Jupyter notebooks;
- Support for multiple datasets;
- Easy switching of neural network architectures along with hyperparameters;
- Training and performance evaluation integrated with Weights & Biases;
- Local saving of summarized test results and visualizations.

## Requirements
- Ubuntu (minimum 22.04)
- NVIDIA GPU with CUDA 12.x
- Python 3.10+

## Setup

```
curl -sSL https://install.python-poetry.org | python -
```
To install dependencies and create the environment: 

```
poetry install
```

Then run it using: 

```
poetry shell
```

To run the application:

```
poetry run python3  src/main.py
```

