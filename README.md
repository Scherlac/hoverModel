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
- **Modular demonstration system** with interchangeable control sources and output handlers
- **Comprehensive CLI interface** for easy demonstrations and testing
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
- `demo.py` - Combined testing and video generation functionality using modular architecture
- `control_sources.py` - Control signal generators for different movement patterns
- `demo_outputs.py` - Output handlers for various demonstration modes
- `physics.py` - Physics engine with abstract `PhysicsEngine` base class
- `visualization.py` - Visualization backends with abstract `Visualizer` base class
- `environment.py` - Main environment orchestrating physics and visualization
- `pyproject.toml` - Project configuration and dependencies
- `README.md` - This documentation file

## Usage

### Quick Start
Run all physics tests to see the hovercraft behaviors:
```bash
python demo.py
```

### Individual Demonstrations
Test specific movement patterns:
```bash
python demo.py hover              # Test hovering (no control inputs)
python demo.py linear             # Test forward movement
python demo.py rotate             # Test rotational movement
python demo.py sinusoid           # Test combined sinusoidal motion
python demo.py chaotic            # Test boundary bouncing behavior
```

### Video Creation
Create demonstration videos:
```bash
python demo.py video sinusoid     # Create sinusoidal movement video
python demo.py video chaotic      # Create boundary bouncing video
python demo.py video linear --steps 100 --fps 30 --output my_demo.mp4
```

### Advanced Options
```bash
python demo.py --help             # Show all available commands and options
```

**Options:**
- `--steps STEPS` - Number of simulation steps (default: 50 for tests, 200 for video)
- `--fps FPS` - Video frame rate (default: 25)
- `--output FILE` - Output video filename (default: hovercraft_demo.mp4)

### 3D Visualization
Run the interactive 3D visualization:
```bash
python main.py
```

This will open an Open3D 3D visualization window showing the hovercraft moving within the fenced training area.

**Note:** On Windows command line, GUI windows may not be visible. The visualization works correctly for video capture and in Python IDEs/Jupyter notebooks.

### Command Line Interface (CLI)
The demo system features a comprehensive CLI for easy experimentation:

**Available Control Types:**
- `hover` - Hovering (zero control inputs)
- `linear` - Constant forward thrust
- `rotate` - Pure rotational movement
- `sinusoid` - Combined sinusoidal motion
- `chaotic` - High-amplitude boundary testing

**Common Commands:**
```bash
# Run demonstrations
python demo.py hover              # Quick hover test
python demo.py linear --steps 100 # Extended linear test

# Create videos
python demo.py video sinusoid     # Standard video
python demo.py video chaotic      # Boundary bouncing video
python demo.py video linear --fps 30 --output custom.mp4

# Get help
python demo.py --help
```

## Architecture

The codebase follows SOLID principles with a highly composable, modular architecture:

### Components
- **`control_sources.py`**: Control signal generators with abstract `ControlSource` base class
  - `HoveringControl` - Zero control signals for stability testing
  - `LinearMovementControl` - Constant forward thrust
  - `RotationalControl` - Pure rotational movement
  - `SinusoidalControl` - Combined sinusoidal motion
  - `ChaoticControl` - Boundary testing with high-amplitude signals
  - `ControlSourceFactory` - Factory pattern for creating control sources
- **`demo_outputs.py`**: Output handlers with abstract `DemoOutput` base class
  - `NullOutput` - Silent testing mode
  - `LoggingOutput` - Console position reporting
  - `VideoOutput` - MP4 video generation with Open3D visualization
  - `BouncingVideoOutput` - Specialized boundary collision videos
  - `DemoRunner` - Composition layer combining control sources with outputs
- **`physics.py`**: Physics engine with abstract `PhysicsEngine` base class
- **`visualization.py`**: Visualization backends with abstract `Visualizer` base class
- **`environment.py`**: Main environment orchestrating physics and visualization
- **`demo.py`**: Demonstration and testing utilities using the modular system

### Design Benefits
- **High Cohesion**: Each component has a single, well-defined responsibility
- **Low Coupling**: Components communicate through abstractions, not concrete implementations
- **Strategy Pattern**: Interchangeable control sources and output handlers
- **Factory Pattern**: Clean object creation for control sources
- **Composition over Inheritance**: Flexible combination of control and output strategies
- **Testability**: Physics, control, and output can be tested independently
- **Extensibility**: Easy to add new movement patterns or output formats
- **Performance**: Vectorized physics calculations using NumPy tensors
- **Modularity**: Clean separation between control generation and output handling

### Physics Implementation
The physics engine uses **vectorized NumPy operations** for efficient computation:

- **Position/Velocity**: 3D vectors for spatial coordinates
- **Forces**: Vector calculations for Newtonian mechanics
- **Boundaries**: Vectorized collision detection and response
- **Integration**: Efficient tensor operations for state updates

This provides better performance and cleaner code compared to scalar operations.

### Modular Demo System
The demonstration system uses composition to combine control sources with output handlers, accessible through both programmatic API and CLI:

**Programmatic Usage:**
```python
from control_sources import ControlSourceFactory
from demo_outputs import DemoRunner

# Create control source
control = ControlSourceFactory.create_sinusoidal()

# Create demo runner
runner = DemoRunner()

# Run physics test with logging
runner.run_test(control, steps=50)

# Create video demonstration
runner.create_video(control, "demo.mp4", steps=200, fps=25)

# Create boundary bouncing video
runner.create_bouncing_video(control, "bounce.mp4", steps=300)
```

**CLI Usage:**
```bash
# Equivalent CLI commands
python demo.py sinusoid --steps 50
python demo.py video sinusoid --steps 200 --fps 25 --output demo.mp4
python demo.py video chaotic --steps 300 --output bounce.mp4
```

This architecture allows testing any control strategy with any output format without code duplication.

### Usage Examples

**CLI Commands:**
```bash
# Quick demonstrations
python demo.py hover              # Test stability
python demo.py linear             # Test forward movement
python demo.py chaotic            # Test boundary interactions

# Video creation
python demo.py video sinusoid     # Create smooth motion video
python demo.py video chaotic      # Create dynamic bouncing video
```

**Programmatic API:**
```python
# Default configuration
env = HovercraftEnv()

# Custom physics with null visualization (for testing)
from physics import HovercraftPhysics
from visualization import NullVisualizer
physics = HovercraftPhysics({'mass': 2.0})
env = HovercraftEnv(physics_engine=physics, visualizer=NullVisualizer({}))

# Modular demo system - combine any control source with any output
from control_sources import ControlSourceFactory
from demo_outputs import DemoRunner

runner = DemoRunner()

# Test different movement patterns
hovering = ControlSourceFactory.create_hovering()
runner.run_test(hovering, steps=50)

linear = ControlSourceFactory.create_linear(forward_force=1.0)
runner.run_test(linear, steps=50)

chaotic = ControlSourceFactory.create_chaotic()
runner.create_bouncing_video(chaotic, "boundary_test.mp4", steps=300)

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

The physics simulation and modular demonstration system are complete and tested. Contributions welcome for:
- Additional control source implementations (new movement patterns)
- Alternative visualization backends
- Enhanced CLI features and commands
- RL integration and reward function design
- Additional physics features and environmental effects
- Performance improvements and optimizations
- Multi-agent scenarios and advanced features

## License

MIT License