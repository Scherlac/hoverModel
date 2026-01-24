# Hovercraft Simulation

A simple reinforcement learning environment simulation for a hovercraft using Open3D for 3D visualization.

## Current Status

✅ **Fully Implemented and Working:**
- Complete hovercraft physics simulation with Gaussian forces
- Rectangular fence boundary with bouncing mechanics
- Mass-based Newtonian physics (F=ma, torque responses)
- Friction proportional to elevation
- Controlled forward thrust and rotational torque
- 3D visualization with Open3D
- Test suite demonstrating all physics behaviors

⚠️ **Requirements:**
- Python >= 3.12 (for Open3D compatibility)
- Open3D installed via `pip install -e .`

## Installation

This project uses modern Python packaging with `pyproject.toml`.

### Prerequisites
- Python >= 3.12

### Install Dependencies
```bash
pip install -e .
```

This installs numpy and Open3D for the physics simulation and 3D visualization.

## Usage

Run the simulation with 3D visualization:
```bash
python main.py
```

This will open an Open3D 3D visualization window showing the hovercraft moving within the fenced training area.

### Testing Physics
Run the demonstration tests to verify physics:
```bash
python test_demo.py
```

This runs tests for:
- Hovering behavior with random forces
- Forward movement and boundary bouncing
- Rotational control
- Combined physics simulation

### Creating Demo Video
Run the video demonstration (requires matplotlib and ffmpeg):
```bash
python video_demo.py
```

This creates `hovercraft_demo.mkv` or `hovercraft_demo.gif` showing the hovercraft movement and height changes over time.
```

This will open an Open3D visualization window showing the hovercraft moving within the fenced training area.

## Future Steps

### Immediate (High Priority)
1. **Add Reinforcement Learning Interface**
   - Implement Gymnasium-compatible environment
   - Add reward functions for RL training
   - Define observation and action spaces

2. **Enhance Visualization**
   - Add hovercraft orientation arrows
   - Improve 3D rendering performance
   - Add camera controls

### Medium Term
3. **Enhanced Physics**
   - Add more realistic aerodynamic effects
   - Implement collision detection with obstacles
   - Add wind disturbances

4. **Performance Optimization**
   - Optimize visualization rendering
   - Add parallel simulation capabilities

### Long Term
5. **Advanced Features**
   - Multi-agent scenarios
   - Terrain interaction
   - Sensor simulation (lidar, camera)

## Contributing

The physics simulation is complete and tested. Contributions welcome for:
- Alternative visualization implementations
- RL integration
- Additional physics features
- Performance improvements

## License

Add your license here