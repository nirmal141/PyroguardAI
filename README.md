# PyroGuard AI - Wildfire Suppression Drone System

A comprehensive Python-based wildfire simulation with both rule-based and reinforcement learning firefighting drone agents. The system features:

- **3D wildfire visualization** with realistic fire spread mechanics
- **Multiple drone types**: Simple rule-based and Deep Q-Network (DQN) RL agents
- **Interactive controls** for manual intervention
- **Real-time environmental factors** (wind, terrain, weather)
- **Training infrastructure** for RL agent development

## Project Structure

```
PyroguardAI/
├── drones/
│   ├── simple/                 # Rule-based drone implementation
│   │   └── firefighter_drone.py
│   └── rl/                     # Reinforcement Learning drone
│       ├── rl_agent.py         # DQN agent with Dueling architecture
│       └── rl_environment.py   # Gymnasium-compatible wrapper
├── environment/
│   └── wildfire_env.py         # Core wildfire simulation
├── demos/
│   ├── demo_simple.py          # Simple drone demo
│   └── demo_integrated.py      # RL-integrated demo
├── training/
│   └── train_rl_drone.py      # RL training script
└── run_*.py                   # Easy-to-use runner scripts
```

## Quick Start

### Prerequisites
```bash
pip install pygame numpy torch gymnasium matplotlib
```

### Run Demos
```bash
# Simple rule-based drone
python run_simple_demo.py

# RL-integrated demo (falls back to simple if no model)
python run_rl_demo.py

# Train RL agent
python run_training.py
```

## Drone Types

### Simple Rule-Based Drone
- Autonomous patrol patterns
- Fire detection and prioritization
- Water resource management
- Emergency recharging behavior

### RL Drone (Deep Q-Network)
- **Architecture**: Dueling DQN with CNN for spatial processing
- **Experience Replay**: Prioritized experience replay buffer
- **Multi-objective Rewards**: Fire suppression, efficiency, safety
- **Training**: Supports both training from scratch and fine-tuning

## Controls
- **Arrow Keys**: Manual drone control
- **Space**: Toggle water suppression
- **R**: Reset simulation
- **Q**: Quit

## Features

### Fire Dynamics
- Heat-based fire spread with wind influence
- Multiple fire intensity levels
- Terrain-dependent burn rates
- Smoke and heat visualization

### RL Training System
- **Deep Q-Network**: Dueling architecture for value decomposition
- **CNN Processing**: Spatial fire pattern recognition
- **Experience Replay**: Prioritized sampling for efficient learning
- **Multi-objective Rewards**: Balances suppression, efficiency, and safety
- **Training Metrics**: Comprehensive logging and visualization

### Environment
- Procedurally generated terrain
- Dynamic weather conditions
- Real-time 3D rendering
- Gymnasium-compatible for RL training

## Configuration

### Simple Demo
```python
env = WildfireEnvironment(
    grid_size=25,           # World size
    enable_drones=True,     # AI drone control
    fire_spread_rate=0.1,   # Fire intensity
    wind_strength=0.3       # Wind effect
)
```

### RL Training
```python
# Train new agent
python run_training.py --episodes 5000 --save-freq 500

# Resume training
python run_training.py --episodes 2000 --model-path models/checkpoint_1000.pth
```

## RL Agent Architecture

The RL drone uses a sophisticated Deep Q-Network with:

- **Dueling DQN**: Separates state value and action advantage estimation
- **CNN Backbone**: Processes spatial fire patterns and terrain
- **Prioritized Replay**: Focuses learning on important experiences
- **Multi-head Output**: Handles discrete movement and water actions
- **Reward Engineering**: Balances multiple objectives (suppression, efficiency, safety)

## Training Results

The RL agent learns to:
- Identify fire hotspots and prioritize suppression
- Optimize flight paths for fuel efficiency
- Balance aggressive suppression with safety
- Coordinate water usage and refilling strategies

## Performance

- **Simple Drone**: ~60 FPS on modest hardware
- **RL Training**: GPU recommended for faster training
- **RL Inference**: Real-time performance on CPU

## Future Enhancements

- Multi-agent coordination
- Hierarchical reinforcement learning
- Real fire data integration
- Advanced weather modeling
- Distributed training support