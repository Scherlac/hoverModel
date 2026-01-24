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

**Note:** On Windows command line, GUI windows may not be visible. The visualization works correctly for video capture (see below) and in Python IDEs/Jupyter notebooks.

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

## Architecture

The codebase follows SOLID principles with a composable architecture:

### Components
- **`physics.py`**: Physics engine with abstract `PhysicsEngine` base class
- **`visualization.py`**: Visualization backends with abstract `Visualizer` base class  
- **`environment.py`**: Main environment orchestrating physics and visualization
- **`demo.py`**: Demonstration and testing utilities

### Design Benefits
- **High Cohesion**: Each component has a single, well-defined responsibility
- **Low Coupling**: Components communicate through abstractions, not concrete implementations
- **Testability**: Physics can be tested independently of visualization
- **Extensibility**: Easy to add new physics models or visualization backends
- **Performance**: Vectorized physics calculations using NumPy tensors
- **Composition**: Components can be mixed and matched via dependency injection

### Physics Implementation
The physics engine uses **vectorized NumPy operations** for efficient computation:

- **Position/Velocity**: 3D vectors for spatial coordinates
- **Forces**: Vector calculations for Newtonian mechanics
- **Boundaries**: Vectorized collision detection and response
- **Integration**: Efficient tensor operations for state updates

This provides better performance and cleaner code compared to scalar operations.

### Usage Examples
```python
# Default configuration
env = HovercraftEnv()

# Custom physics with null visualization (for testing)
from physics import HovercraftPhysics
from visualization import NullVisualizer
physics = HovercraftPhysics({'mass': 2.0})
env = HovercraftEnv(physics_engine=physics, visualizer=NullVisualizer({}))

# Vector gravity (e.g., simulating wind effects)
wind_physics = HovercraftPhysics({
    'gravity': [0.5, 0.0, -9.81],  # [x, y, z] gravity vector
    'bounds': [[-10, 10], [-10, 10], [0, 15]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
})
env = HovercraftEnv(physics_engine=wind_physics)

# Compact bounds configuration
compact_physics = HovercraftPhysics({
    'bounds': [[-5, 5], [-5, 5], [0, 10]]  # Clean array format
})
```

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