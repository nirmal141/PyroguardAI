# Multi-Agent Drone Firefighting System

This system extends the basic wildfire simulation with multi-agent reinforcement learning for coordinated drone firefighting operations.

## 🚁 Key Features

### Drone Agents
- **Battery Management**: Drones consume battery with movement and operations
- **Water Capacity**: Limited water supply for extinguishing fires
- **Temperature Sensors**: Monitor fire intensity and detect nearby fires
- **Autonomous Operations**: RL-trained agents can coordinate firefighting

### Multi-Agent Environment
- **Coordinated Actions**: Multiple drones can work together
- **Real-time Visualization**: See drones, fires, and status in real-time
- **Trigger System**: Manual activation of drone firefighting mode
- **Performance Tracking**: Monitor fires extinguished, battery levels, etc.

## 🎮 Controls

### Interactive Simulation
- **SPACE**: Toggle pause/resume
- **T**: Toggle drone firefighting mode (trigger button)
- **R**: Recharge all drones at base station
- **A**: Toggle auto mode (auto-activate when fires appear)
- **ESC**: Exit simulation

### Drone Actions
Each drone can perform 7 actions:
1. **Move Up/Down/Left/Right**: Navigate the grid
2. **Stay**: Remain in current position
3. **Extinguish**: Use water to put out fires
4. **Scan Temperature**: Update temperature readings

## 🚀 Usage

### Quick Demo
```bash
python demo_multi_agent.py
```

### Interactive Simulation
```bash
python scripts/run_multi_agent_fire_sim.py
```

### RL Training
```bash
# Train new agents
python scripts/train_drone_rl.py

# Test trained agents
python scripts/train_drone_rl.py test
```

## 🧠 RL Training

The system uses **PPO (Proximal Policy Optimization)** from stable-baselines3 for training:

### Environment Wrapper
- Converts multi-agent problem to single-agent format
- Flattens observations for RL compatibility
- Handles action space mapping

### Training Features
- **Vectorized Environments**: Parallel training for efficiency
- **Evaluation Callbacks**: Monitor training progress
- **Model Checkpointing**: Save best performing models
- **Tensorboard Logging**: Visualize training metrics

### Reward Structure
- **+10.0**: Successfully extinguish a fire
- **+5.0**: Fire reduction in environment
- **-1.0**: Attempt to extinguish non-burning cell
- **-0.1**: Low battery warning
- **-0.5**: Low water warning
- **-5.0**: Battery depletion penalty

## 🔧 Configuration

### Environment Parameters
```python
env = MultiAgentFireEnv(
    width=16,                    # Grid width
    height=16,                   # Grid height
    p_spread=0.4,               # Fire spread probability
    p_burnout=0.15,             # Fire burnout probability
    max_steps=1000,             # Maximum simulation steps
    ignitions=(2, 4),           # Number of initial fires
    num_drones=3,               # Number of drone agents
    seed=42                     # Random seed
)
```

### Drone Parameters
```python
drone = DroneAgent(
    drone_id=0,
    start_x=5, start_y=5,
    max_battery=1.0,            # Maximum battery level
    max_water=1.0,              # Maximum water capacity
    battery_drain_rate=0.005,   # Battery consumed per action
    water_usage_rate=0.15,      # Water used per extinguishing
    temperature_threshold=0.7    # Fire detection threshold
)
```

## 📊 Monitoring

### Drone Status
- **Position**: Current grid coordinates
- **Battery**: Energy level (0.0 to 1.0)
- **Water**: Water tank level (0.0 to 1.0)
- **Temperature**: Current temperature reading
- **Activity**: Operational status

### Performance Metrics
- **Fires Extinguished**: Total fires put out
- **Distance Traveled**: Movement efficiency
- **Battery Efficiency**: Actions per battery unit
- **Water Efficiency**: Fires per water unit

## 🎯 Future Enhancements

### Planned Features
1. **Formation Flying**: Coordinated drone formations
2. **Communication**: Inter-drone information sharing
3. **Path Planning**: Optimal route calculation
4. **Weather Effects**: Wind and weather impact
5. **Base Station**: Automated recharging and refilling

### Advanced RL
1. **Multi-Agent RL**: Independent learning agents
2. **Hierarchical RL**: High-level coordination
3. **Transfer Learning**: Pre-trained models
4. **Curriculum Learning**: Progressive difficulty

## 🛠️ Technical Details

### Architecture
```
MultiAgentFireEnv
├── FireSimEnv (base fire simulation)
├── DroneAgent[] (individual drone agents)
├── Action Space (7 actions per drone)
├── Observation Space (drone state + local grid)
└── Reward Function (firefighting performance)
```

### Dependencies
- **gymnasium**: RL environment framework
- **stable-baselines3**: RL algorithms
- **pygame**: Visualization
- **numpy**: Numerical computations
- **torch**: Deep learning backend

## 📈 Training Tips

1. **Start Small**: Begin with 2 drones and small grids
2. **Monitor Metrics**: Watch battery and water usage
3. **Adjust Rewards**: Tune reward structure for desired behavior
4. **Use Callbacks**: Monitor training progress
5. **Save Models**: Keep best performing models

## 🐛 Troubleshooting

### Common Issues
- **Import Errors**: Ensure all dependencies are installed
- **Pygame Issues**: Update pygame to latest version
- **Memory Issues**: Reduce number of parallel environments
- **Training Slow**: Use GPU acceleration if available

### Performance Optimization
- **Vectorized Environments**: Use multiple parallel environments
- **Batch Processing**: Process multiple observations together
- **Model Checkpointing**: Save progress regularly
- **Early Stopping**: Stop training when performance plateaus
