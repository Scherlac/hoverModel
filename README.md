# Hovercraft Simulation

A simple reinforcement learning environment simulation for a hovercraft using Open3D for visualization.

## Installation

This project uses modern Python packaging with `pyproject.toml`.

### Prerequisites
- Python >= 3.13

### Install Dependencies
```bash
pip install -e .
```

This will install the package in editable mode along with its dependencies:
- open3d
- numpy

## Usage

Run the simulation:
```bash
python main.py
```

Run the demonstration tests:
```bash
python test_demo.py
```

The test script demonstrates:
- Hovering behavior with random forces
- Forward movement and boundary bouncing
- Rotational control
- Combined physics simulation

Each test opens an Open3D visualization window showing the hovercraft's behavior.

## Features

- 3D visualization with Open3D
- Hovercraft physics simulation with Gaussian forces
- Rectangular fence boundary with bouncing effects
- Configurable control forces for reinforcement learning