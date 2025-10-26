# Formal Verification of Autonomous Car Control
This repository contains work for the *Dependable and Deployable AI for Robotics* course. Below is a short description on how to navigate the contents.

## Lab 1 - Adversarial Training
The `lab1` folder contains the following `.ipynb` scripts:
- `cars-manual-dataset-generate` - defines a `Car` class and a PID controller and collects simulation data.
- `train-nn-model` - trains a neural network on the data collected from the previous script to control an ego car.
- `run-nn-model` - runs both the neural network and PID controller in parallel to compare simulation events and outcomes.
- `adversarial-training` & `Baoyan_Adv.ipynb` - experimenting with PGD attacks on the collected dataset and adversarial training on the neural network.

## Custom Gymnasium Simulation Environment
*Work in progress!*

The `rl-sim-training` folder contains a custom Gymnasium environment adapted from S. Teuber's "Linear Cruise Control in Relative Coordinates" implementation for VerSAILLE & Mosaic [(link)](https://github.com/samysweb/VerSAILLE/blob/kikit/technical/docker/contents/libs/acc.py).

The environment is implemented inside the folder `custom-car-env`, and consists of a single straight lane with a leader car, which accelerates, brakes or idles at random, and an ego car, for the purpose of controlling it to avoid collision.

There are three related `.py` files:
- `sim_test` - runs the custom simulation environment with preset deterministic ego car policies (either solely brake, accelerate, idle, or random). Video files of each policy are saved in a `video` folder.
- `dqn-training` - trains a DQN RL agent to control the ego car. Model checkpoints are saved in a `model` folder.
- `data-collection` - uses a selected trained DQN model to run and collect simulation data.

## Installation
To install the necessary dependencies for this environment, run the following commands:

```{shell}
conda env create -f environment.yml
cd rl-sim-training/custom-car-env
pip install -e .
```
