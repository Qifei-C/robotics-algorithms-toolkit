# IMU Attitude Estimation

This project implements an Unscented Kalman Filter (UKF) for attitude estimation from IMU (Inertial Measurement Unit) data.

## 🚀 Core Components

*   **Unscented Kalman Filter**: The core UKF implementation for estimating roll, pitch, and yaw angles.
*   **Quaternion-based Representation**: Uses quaternions to represent orientation, avoiding issues like gimbal lock.
*   **Sensor Calibration**: Includes parameters for calibrating accelerometer and gyroscope data.
*   **Visualization**: Utilities for plotting the estimated attitude.

## 🛠️ Usage

To run the project, you can use the `estimate_rot` function to process IMU data and obtain the estimated roll, pitch, and yaw angles. The results can then be visualized using the provided plotting functions.
