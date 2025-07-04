# Histogram Filter for Robot Localization

This project implements a histogram filter, a discrete grid-based Bayesian filter, for robot localization.

## 🚀 Core Components

*   **Histogram Filter Implementation**: The core logic for updating the robot's belief about its position based on motion and sensor observations.
*   **Motion Model**: Defines how the robot moves and the uncertainty associated with its movement.
*   **Sensor Model**: Describes the probability of observing a certain sensor reading given the robot's true position.
*   **Visualization**: Utilities for visualizing the belief distribution on the grid map.

## 🛠️ Usage

To use this project, you can define a grid map and an initial belief distribution. Then, you can apply the histogram filter iteratively with robot actions and sensor observations to update the robot's estimated position.
