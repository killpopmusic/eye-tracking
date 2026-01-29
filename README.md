# Eye tracking computing environment

### About
The projet provides a modular environment, which enables various test regarding eye tracking algorithm utlizing facial landmarks' coordinates. Different approaches to the gaze prediction can be found in dedicated branches.

### Features 
- Test of different appraoches of eye tracking task: regression of the gaze point, classification to the gaze ROI;
- Complex database analysis tools in jupyter notebook;
- Usage of different databases;
- Simple change of used NN architectures along with hiperparameters;
- Training and performance evaluation integrated with Weights and Biases;
- Local saving of summarized test results and visualizations

### Requirements
- Ubuntu (minimum 22.04)
- NVIDIA GPU with CUDA 12.x
- Python 3.10+

## Setup 
The project uses poetry dependency managment, to install:
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

