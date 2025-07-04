# Robotics Algorithms Toolkit

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-orange.svg)](https://numpy.org/)
[![OpenAI Gym](https://img.shields.io/badge/OpenAI%20Gym-0.21+-green.svg)](https://gym.openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive collection of fundamental robotics algorithms including SLAM, localization, control, and reinforcement learning. Features implementations of particle filters, Kalman filters, PPO, and probabilistic robotics methods with simulation environments.

## 🎯 Features

- **SLAM Algorithms**: Particle Filter SLAM with mapping
- **Localization**: Histogram filters and probabilistic methods
- **State Estimation**: Kalman and particle filters
- **Reinforcement Learning**: PPO for bipedal walking
- **Probabilistic Models**: HMM and Baum-Welch algorithm
- **Sensor Fusion**: IMU attitude estimation

## 🚀 Quick Start

```python
from src.particle_filter_slam import ParticleFilterSLAM
from src.ppo_bipedal import PPOAgent

# SLAM Example
slam = ParticleFilterSLAM(num_particles=1000)
slam.load_sensor_data('data/lidar_odometry.txt')
trajectory, map_estimate = slam.run_slam()

# Reinforcement Learning Example  
agent = PPOAgent(env_name='BipedalWalker-v3')
agent.train(total_timesteps=1000000)
agent.save_model('models/bipedal_walker_ppo.zip')
```

## 🤖 Algorithm Collection

### 1. Particle Filter SLAM
- **Location**: `src/ParticleFilter-SLAM/`
- **Features**: Simultaneous localization and mapping
- **Applications**: Mobile robot navigation, autonomous vehicles

### 2. PPO Bipedal Walker
- **Location**: `src/PPO-BipedalWalker/` 
- **Features**: Reinforcement learning for locomotion
- **Applications**: Humanoid robots, dynamic walking

### 3. Histogram Filter Localization
- **Location**: `src/HistogramFilter/`
- **Features**: Probabilistic localization in discrete grids
- **Applications**: Indoor robot navigation

### 4. HMM Baum-Welch Algorithm
- **Location**: `src/HMM-BaumWelch/`
- **Features**: Hidden Markov Model parameter estimation
- **Applications**: Behavior recognition, sensor modeling

### 5. IMU Attitude Estimation
- **Location**: `src/IMU-AttitudeEstimation/`
- **Features**: Quaternion-based orientation tracking
- **Applications**: Drones, spacecraft, mobile robots

## 📁 Project Structure

```
robotics-algorithms-toolkit/
├── src/                           # Source algorithms
│   ├── ParticleFilter-SLAM/       # SLAM implementation
│   ├── PPO-BipedalWalker/         # RL locomotion
│   ├── HistogramFilter/           # Localization
│   ├── HMM-BaumWelch/            # Probabilistic models
│   └── IMU-AttitudeEstimation/    # Sensor fusion
├── examples/                      # Usage examples
├── simulations/                   # Simulation environments
├── tests/                         # Test suite
├── docs/                          # Documentation
└── README.md                     # This file
```

## 🔧 Algorithm Details

### Particle Filter SLAM
```python
# Initialize SLAM system
slam = ParticleFilterSLAM(
    num_particles=1000,
    map_resolution=0.1,
    motion_noise=0.1,
    sensor_noise=0.5
)

# Process sensor measurements
for timestep in range(num_timesteps):
    slam.predict(odometry[timestep])
    slam.update(lidar_scan[timestep])
    slam.resample()
```

### PPO Training
```python
# Configure PPO hyperparameters
ppo_config = {
    'learning_rate': 3e-4,
    'clip_range': 0.2,
    'n_steps': 2048,
    'batch_size': 64,
    'n_epochs': 10
}

agent = PPOAgent(env_name='BipedalWalker-v3', **ppo_config)
agent.train(total_timesteps=1000000)
```

### Histogram Filter
```python
# Discrete localization
hfilter = HistogramFilter(
    grid_size=(50, 50),
    cell_size=0.2,
    motion_uncertainty=0.1
)

belief = hfilter.localize(sensor_measurements, motion_commands)
```

## 📊 Performance Benchmarks

Performance metrics will vary based on your specific environment, hardware configuration, and algorithm parameters. Each algorithm is designed to achieve competitive results when properly configured for your robotics application.

## 🎮 Simulation Environments

### Custom Environments
- 2D grid world for localization
- Landmark-based SLAM environment
- IMU simulation with noise models

### OpenAI Gym Integration
- BipedalWalker-v3 for locomotion
- Custom robotics environments
- Sensor noise simulation

## 🔬 Research Applications

- **Academic Research**: Probabilistic robotics education
- **Industry**: Autonomous navigation systems
- **Competitions**: Robotics challenges and benchmarks
- **Prototyping**: Algorithm validation before hardware deployment

## 🛠 Installation & Setup

```bash
# Clone repository
git clone <repository-url>
cd robotics-algorithms-toolkit

# Install dependencies
pip install -r requirements.txt

# Install additional RL dependencies
pip install stable-baselines3[extra]

# Run example
python examples/slam_demo.py
```

## 📈 Educational Value

Perfect for:
- Robotics engineering courses
- Research in probabilistic robotics
- Algorithm comparison studies
- Rapid prototyping of robot systems

## 🤝 Contributing

We welcome contributions of new algorithms, improvements, and bug fixes!

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

🤖 **Fundamental Algorithms for Intelligent Robots**