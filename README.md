# 🤖 SCALABLE MULTI-ROBOT NAVIGATION USING TD3-PER WITH REWARD SHAPING AND RECURSIVE CURRICULUM LEARNING

Supplementary material for the paper **SCALABLE MULTI-ROBOT NAVIGATION USING TD3-PER WITH REWARD SHAPING AND RECURSIVE CURRICULUM LEARNING**.

This project trains TurtleBot3 robots to navigate without relying on pre-built maps, using the Twin-Delayed Deep Deterministic Policy Gradient (TD3) algorithm as the base learner. On top of TD3, it adds Prioritized Experience Replay (PER) for more efficient sampling and recursive curriculum learning to scale the policy across multiple robots. Robots rely solely on LiDAR for sensing — no SLAM or mapping is involved.

It's built within the ROS framework, with Gazebo handling simulation and PyTorch handling the DRL model.

## Table of Contents

- [Citation](#citation)
- [Getting Started](#getting-started)
  - [Assumed Background Knowledge](#assumed-background-knowledge)
  - [Prerequisite](#prerequisite)
  - [Installation](#installation)
    - [Clone the Repository](#clone-the-repository)
- [Simulation](#simulation)
  - [Training](#training)
  - [Testing](#testing)

## 📄 Citation

**TBD** — citation will be added here once the paper is officially accepted/published.

## 🚀 Getting Started

### 🧠 Assumed Background Knowledge

This guide assumes you're comfortable with:

- **Linux / Ubuntu command line** — basic navigation and running shell commands.
- **ROS fundamentals** — nodes, topics, and launch files.
- **Gazebo simulation basics** — launching worlds and spawning robot models.
- **TurtleBot3** — familiarity with the simulation stack and multi-robot spawning.
- **Python** — the RL agent is implemented as Python ROS nodes.
- **Reinforcement learning concepts**

If any of these are unfamiliar, the [ROS Tutorials](https://wiki.ros.org/ROS/Tutorials) is good starting points before diving into this package.

### ⚙️ Prerequisite

This package assumes you already have a working **ROS (Noetic)** + **Gazebo** + **TurtleBot3 simulation** setup. If you haven't set that up yet, refer to the [ROS Noetic installation guide](https://wiki.ros.org/noetic/Installation/Ubuntu) and the [TurtleBot3 e-Manual](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/).

Python dependencies are installed as part of the steps below.

### 📦 Installation

#### 📥 Clone the Repository

Clone this repository into the `src` folder of your catkin workspace:
```bash
cd ~/catkin_ws/src
git clone https://github.com/rfan121/multirobot-nav-td3-recl.git
```

**Install dependencies:**
```bash
pip install numpy torch tensorboard squaternion
```

**Install ROS dependencies:**
```bash
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -y
```

**Build the workspace and source it:**
```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

## 🎮 Simulation

### 🏋️ Training

```bash
cd multirobot-nav-td3-recl/td3_training
python3 train.py
```

### 🧪 Testing

```bash
cd multirobot-nav-td3-recl/td3_training/env1   # environment 1
python3 test2_env1.py   # 2 robots
python3 test4_env1.py   # 4 robots
python3 test8_env1.py   # 8 robots
```

```bash
cd multirobot-nav-td3-recl/td3_training/env2   # environment 2
python3 test2_env2.py   # 2 robots
python3 test4_env2.py   # 4 robots
python3 test8_env2.py   # 8 robots
```

```bash
cd multirobot-nav-td3-recl/td3_training/env3   # environment 3
python3 test2_env3.py   # 2 robots
python3 test4_env3.py   # 4 robots
python3 test8_env3.py   # 8 robots
```

---
