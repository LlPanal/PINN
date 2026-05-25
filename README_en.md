*Read this in [Català](README.md).*

# PINN - Physics-Informed Neural Networks

This repository contains the code used for my Bachelor's Thesis (TFG) on Physics-Informed Neural Networks and their applications.

## Types of Problems

* **Forward Problems:** Finding an approximation of the real solution given the initial and boundary conditions, and a set of collocation points that must satisfy the physical laws (ODEs or PDEs).
* **Inverse Problems:** In this project, this has only been studied for the Meinhardt equations with the goal of observing Turing instability. However, the process is similar for other equations or systems of equations.

## Installation

To install the necessary dependencies, run the following command:

```
pip install -r requirements.txt
```

## Execution

To run any of the ODEs/PDEs, use the following command structure:

```
python problem_type/script.py
```

For example:

```
python directe/burguers.py
```

> **Note:** At the beginning of each file, there are a series of parameters you can tweak to "play" with the equations and observe how the results change.