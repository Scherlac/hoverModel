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
- Video generation from 3D visualization (MP4 format)
- All features tested and validated

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

## Project Structure

- `main.py` - Core hovercraft environment with physics simulation and Open3D visualization
- `demo.py` - Combined testing and video generation functionality
- `pyproject.toml` - Project configuration and dependencies
- `README.md` - This documentation file

## Usage

Run the simulation with 3D visualization:
```bash
python main.py
```

This will open an Open3D 3D visualization window showing the hovercraft moving within the fenced training area.

### Testing Physics
Run the physics tests:
```bash
python demo.py
```

This runs tests for hovering, movement, and rotation behaviors.

### Creating Demo Video
Create a demonstration video:
```bash
python demo.py video
```

This captures frames from the 3D Open3D visualization and creates `hovercraft_demo.mp4` in the project directory.
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

MIT License