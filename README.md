# HoverModel Simulation

A reinforcement learning environment simulation for a hovercraft with multiple physics backends (Open3D + NumPy or Genesis) and 3D visualization capabilities.

## Current Status

✅ **Fully Implemented and Working:**
- Complete hovercraft physics simulation with Gaussian forces
- Rectangular fence boundary with bouncing mechanics
- Mass-based Newtonian physics (F=ma, torque responses)
- Friction proportional to elevation
- Controlled forward thrust and rotational torque
- 3D visualization with Open3D (live and video modes)
- Video generation from 3D visualization (MP4 format) with automatic cleanup
- **Modular demonstration system** with interchangeable control sources and output handlers
- **Comprehensive CLI interface** for easy demonstrations and testing
- **Separated architecture** with clear file-level separation between outputs and demos
- **Clean state representation** with vector-based `BodyState` class (r/v vectors, theta/omega scalars)
- **Vectorized physics** using natural mathematical objects instead of array slicing
- All features tested and validated
- **Automatic frame cleanup** after video generation to keep workspace clean

⚠️ **Requirements:**
- Python >= 3.12 (for Open3D compatibility)
- Open3D installed via `pip install -e .`
- **Optional**: Genesis physics engine (`pip install genesis-world`) for advanced physics simulation

## Installation

This project uses modern Python packaging with `pyproject.toml`.

### Prerequisites
- Python >= 3.12

### Install Dependencies
```bash
pip install -e .
```

This installs numpy, Open3D for the physics simulation and 3D visualization, and Click for the CLI interface.

### Optional: Genesis Backend
For advanced physics simulation with collision detection and complex rigid body dynamics:
```bash
pip install genesis-world
```

The Genesis backend provides more sophisticated physics capabilities including automatic collision detection, complex mesh support, and potential GPU acceleration.

## Project Structure

- `main.py` - Legacy entry point for basic visualization
- `demo.py` - CLI interface for running demonstrations with modular control and output systems
- `components.py` - Abstract base classes and interfaces for physics engines, bodies, environments, and visualization
- `default_backend.py` - Default physics backend using Open3D + NumPy with Newtonian physics, bodies, and visualization
- `genesis_backend.py` - Alternative physics backend using Genesis engine with advanced physics simulation
- `state.py` - Clean vector-based state representation with `BodyState` class
- `control_sources.py` - Control signal generators for different movement patterns
- `simulation_outputs.py` - Output handlers for various demonstration modes (logging, video, live visualization)
- `demo_runner.py` - Demo orchestration and configuration logic
- `pyproject.toml` - Project configuration and dependencies
- `README.md` - This documentation file

## Usage

### Quick Start
Run physics tests with console output:
```bash
python demo.py run --control linear:force=1.0 --output console --steps 50
```

### Backend Selection
Choose between physics backends:
```bash
# Default Open3D + NumPy backend (faster, simpler physics)
python demo.py run --control hovering --output console --backend default

# Genesis backend (advanced physics with collision detection)
python demo.py run --control hovering --output console --backend genesis
```

### Individual Demonstrations
Test specific movement patterns:
```bash
# Linear movement with console output
python demo.py run --control linear:force=2.0 --output console --steps 100

# Rotational movement
python demo.py run --control rotational:torque=0.5 --output console --steps 100

# Sinusoidal combined motion
python demo.py run --control sinusoidal --output console --steps 100

# Chaotic boundary testing
python demo.py run --control chaotic --output console --steps 150
```

### Video Creation
Create demonstration videos (frames automatically cleaned up after generation):
```bash
# Basic linear movement video
python demo.py run --control linear:force=5.0 --output video:filename=linear_demo.mp4:fps=10 --steps 100

# Bouncing boundary demo (start from center for real movement)
python demo.py run --control linear:force=20.0 --output video:filename=bouncing_demo.mp4:fps=10 --start-x=0.0 --start-y=0.0 --start-z=5.0 --steps 100

# Rotational video with custom settings
python demo.py run --control rotational:torque=1.0 --output video:filename=rotation.mp4:fps=15 --steps 150
```

### Live 3D Visualization
Run interactive 3D visualization:
```bash
# Live visualization of linear movement
python demo.py run --control linear:force=3.0 --output live --steps 100

# Live bouncing demo
python demo.py run --control linear:force=15.0 --output live --start-x=0.0 --start-y=0.0 --start-z=5.0 --steps 200
```

### Advanced Options
```bash
# Custom starting position
python demo.py run --control linear:force=10.0 --output video:filename=custom_start.mp4:fps=10 --start-x=2.0 --start-y=1.0 --start-z=3.0 --steps 100
```

# Multiple controls (not yet implemented - single control per run)
# Custom video settings
python demo.py run --control chaotic --output video:filename=chaos.mp4:fps=20 --steps 200

# Get help
python demo.py --help
python demo.py run --help
```

**Control Options:**
- `linear:force=<float>` - Constant forward thrust
- `rotational:torque=<float>` - Pure rotational movement
- `sinusoidal` - Combined sinusoidal motion
- `chaotic` - Boundary testing with high-amplitude signals
- `hovering` - Zero control inputs for stability testing

**Output Options:**
- `console` - Text-based logging of position, velocity, and events
- `live` - Interactive 3D Open3D visualization window
- `video:filename=<name.mp4>:fps=<int>` - MP4 video generation with automatic cleanup

### Command Line Interface (CLI)
The demo system features a comprehensive Click-based CLI for easy experimentation:

**Available Commands:**
- `run` - Execute simulation with specified control and output

**Backend Options:**
- `--backend default` - Use Open3D + NumPy physics (default, faster)
- `--backend genesis` - Use Genesis physics engine (advanced features)

**Control Types:**
- `linear:force=<float>` - Constant forward thrust
- `rotational:torque=<float>` - Pure rotational movement  
- `sinusoidal` - Combined sinusoidal motion
- `chaotic` - Boundary testing with high-amplitude signals
- `hovering` - Zero control inputs for stability testing

**Output Types:**
- `console` - Text-based logging output
- `live` - Interactive 3D Open3D visualization
- `video:filename=<file.mp4>:fps=<int>` - MP4 video generation

**Common Commands:**
```bash
# Basic demonstrations
python demo.py run --control linear:force=1.0 --output console --steps 50
python demo.py run --control chaotic --output console --steps 100

# Backend selection
python demo.py run --control hovering --output console --backend default
python demo.py run --control hovering --output console --backend genesis

# Video creation with automatic cleanup
python demo.py run --control linear:force=10.0 --output video:filename=demo.mp4:fps=10 --steps 100
python demo.py run --control chaotic --output video:filename=bounce.mp4:fps=15 --steps 150

# Live 3D visualization
python demo.py run --control sinusoidal --output live --steps 100

# Custom starting positions for boundary testing
python demo.py run --control linear:force=20.0,steps=100 --output video:filename=bouncing.mp4:fps=10 --start-x=0.0 --start-y=0.0 --start-z=5.0

# Get help
python demo.py --help
python demo.py run --help
```
```

## Architecture

The codebase follows SOLID principles with a highly composable, modular architecture:

### Components
- **`components.py`**: Abstract base classes and interfaces
  - `Body` - Abstract base class for physical bodies with mass, shape, and dynamics
  - `Environment` - Abstract base class for simulation environments
  - `PhysicsEngine` - Abstract base class for physics implementations
  - `Visualizer` & `VisualizationOutput` - Abstract visualization interfaces
  - `SimulationOutput` - Abstract base class for output handlers
- **`default_backend.py`**: Default physics backend implementation
  - `NewtonianPhysics` - General Newtonian physics using Open3D + NumPy
  - `DefaultBody` - Concrete hovercraft implementation with lifting force characteristics
  - `DefaultBodyEnv` - Environment with Open3D visualization
  - `Open3DVisualizer` & `Open3DVisualizationOutput` - Open3D-based visualization
  - Fast, lightweight physics simulation
- **`genesis_backend.py`**: Advanced physics backend using Genesis engine
  - `GenesisPhysics` - Genesis-based physics with collision detection
  - `GenesisRigidBody` - Rigid body with complex mesh support
  - `GenesisBodyEnv` - Environment with advanced physics capabilities
  - `GenesisVisualizer` & `GenesisVisualizationOutput` - Genesis-based visualization
- **`state.py`**: State representation and management
  - `BodyState` - Clean vector-based state with r/v vectors and theta/omega scalars
  - Single source of truth for state format and operations
  - Provides semantic vector accessors and validation
- **`control_sources.py`**: Control signal generators
  - `ControlSourceFactory` - Factory pattern for creating control sources
  - `LinearMovementControl` - Constant forward thrust
  - `RotationalControl` - Pure rotational movement
  - `SinusoidalControl` - Combined sinusoidal motion
  - `ChaoticControl` - Boundary testing with high-amplitude signals
- **`simulation_outputs.py`**: Output handlers with abstract `SimulationOutput` base class
  - `LoggingSimulationOutput` - Console position reporting
  - `LiveVisualizationOutput` - Interactive 3D Open3D display
  - `VideoSimulationOutput` - MP4 video generation with automatic frame cleanup
- **`demo_runner.py`**: Demo orchestration and configuration logic
  - `DemoRunner` - High-level demo management and execution
- **`demo.py`**: CLI interface using the modular system

### Design Benefits
- **High Cohesion**: Each component has a single, well-defined responsibility
- **Low Coupling**: Components communicate through abstractions, not concrete implementations
- **Body-Physics Separation**: Physical properties separated from physics calculations
- **Multi-Body Support**: Architecture supports multiple interacting bodies
- **Extensibility**: Easy to add new body types (cars, drones, etc.) without changing physics
- **Single State Representation**: `BodyState` class owns all state operations
- **State Ownership**: Environment owns bodies, physics operates on them
- **Strategy Pattern**: Interchangeable control sources and output handlers
- **Factory Pattern**: Clean object creation for control sources
- **Composition over Inheritance**: Flexible combination of control and output strategies
- **Testability**: Physics, control, and output can be tested independently
- **Extensibility**: Easy to add new movement patterns or output formats
- **Performance**: Vectorized physics calculations using NumPy tensors
- **Modularity**: Clean separation between control generation and output handling

### Physics Implementation
The physics engine uses **body-based Newtonian physics** with vectorized NumPy operations:

**Body-Physics Separation:**
- Bodies define their physical properties (mass, forces, bounds)
- Physics engine handles numerical integration and constraints
- Multi-layer physics: single-body dynamics + body-environment interactions

**Vectorized Operations:**
- **Position/Velocity**: 3D vectors for spatial coordinates
- **Forces**: Vector calculations for Newtonian mechanics (F = ma)
- **Torques**: Angular dynamics (τ = Iα)
- **Boundaries**: Vectorized collision detection and response
- **Integration**: Efficient tensor operations for state updates

**Multi-Body Support:**
- `step_multiple()` for simultaneous body updates
- Potential for body-body interactions (future extension)
- Environment manages body collections

This provides better performance and cleaner code compared to scalar operations, with clear separation between "what a body is" and "how physics works".

### Multiple Physics Backends

The project supports two physics backends with different capabilities:

**Default Backend (Open3D + NumPy):**
- **Performance**: Fast, lightweight simulation
- **Physics**: Newtonian physics with custom force models
- **Visualization**: Open3D-based 3D rendering
- **Features**: Boundary collision, friction, custom force fields
- **Use Case**: Quick prototyping, RL training, simple demonstrations

**Genesis Backend:**
- **Performance**: Advanced physics engine with GPU acceleration potential
- **Physics**: Realistic rigid body dynamics with automatic collision detection
- **Visualization**: Built-in OpenGL renderer with real-time performance
- **Features**: Complex mesh support, multi-body collisions, quaternion rotations
- **Use Case**: Advanced simulations, complex geometries, realistic physics

**Backend Selection:**
```python
# Default backend
from default_backend import DefaultBodyEnv
env = DefaultBodyEnv()

# Genesis backend
from genesis_backend import GenesisBodyEnv
env = GenesisBodyEnv()
```

### Modular Demo System
The demonstration system uses composition to combine control sources with output handlers, accessible through both programmatic API and CLI:

**Programmatic Usage:**
```python
from control_sources import ControlSourceFactory
from simulation_outputs import LoggingSimulationOutput, LiveVisualizationOutput, VideoSimulationOutput
from default_backend import DefaultBodyEnv

# Create environment
env = DefaultBodyEnv()

# Create control source
control = ControlSourceFactory.create_linear(force=1.0)

# Create output handler
output = LoggingSimulationOutput(env)

# Run simulation
env.run_simulation(control, steps=50)

# For video with automatic cleanup
video_output = VideoSimulationOutput(env, "demo.mp4", fps=10)
env.run_simulation(control, steps=100)
```

**CLI Usage:**
```bash
# Equivalent CLI commands
python demo.py run --control linear:force=1.0 --output console --steps 50
python demo.py run --control linear:force=1.0 --output video:filename=demo.mp4:fps=10 --steps 100
```

This architecture allows testing any control strategy with any output format without code duplication.

### State Management System

The `BodyState` class provides a clean, vector-based state representation associated with physical bodies:

**State-Body Relationship:**
- `Body` objects own their physical properties (mass, forces, shape)
- `BodyState` represents the current kinematic state (position, velocity, orientation)
- Environment manages collections of bodies and their states
- Physics engine operates on bodies to produce new states

**State Properties:**
```python
r      # 3D position vector [x, y, z]
v      # 3D velocity vector [vx, vy, vz]
theta  # Orientation angle (scalar)
omega  # Angular velocity (scalar)
```

**Key Features:**
- **Automatic Frame Cleanup**: Video generation automatically removes temporary frame files after successful encoding, keeping the workspace clean
- **Flexible Starting Positions**: Custom initial positions for testing boundary interactions and bouncing behavior
- **Real-time Event Logging**: Collision events and physics state are logged during simulation
- **High-Performance Encoding**: FFmpeg integration for efficient MP4 video creation

**Usage:**
```python
from body import DefaultBody
from state import BodyState
import numpy as np

# Create a hovercraft body
hovercraft = DefaultBody(mass=1.0, lift_force_mean=10.0)

# Access its state
state = hovercraft.get_state()
position = state.r      # [x, y, z]
velocity = state.v      # [vx, vy, vz]

# Update state
new_state = BodyState(r=np.array([1, 2, 3]), v=np.array([0.1, 0.2, 0.3]))
hovercraft.set_state(new_state)
```

### Recent Architectural Improvements

**Body-Physics Separation:**
- **`Body` Class**: Abstract base class for physical bodies with mass, shape, and force calculations
- **`DefaultBody` Class**: Concrete implementation with hovercraft-specific properties
- **Physics Engine**: Now works with any `Body` object, not hovercraft-specific
- **Multi-Body Support**: Architecture supports multiple interacting bodies
- **Force Delegation**: Bodies define their own force calculations via `get_forces()`

**State Representation Refactoring:**
- **`BodyState` Class**: Clean vector-based representation associated with bodies
- **Vector Properties**: Natural mathematical objects (`r`, `v` vectors, `theta`, `omega` scalars)
- **Physics Integration**: Physics engine uses body properties instead of hardcoded values
- **Simplified Code**: Eliminated separate indexing for position/velocity components
- **Maintained Compatibility**: `__array__()` method preserves backward compatibility

**File-Level Separation:**
- **`components.py`**: Abstract interfaces and base classes
- **`default_backend.py`**: Concrete implementations for default physics backend
- **`genesis_backend.py`**: Concrete implementations for Genesis physics backend
- **`simulation_outputs.py`**: Pure output handling (logging, video generation)
- **`demo_runner.py`**: Demo orchestration and configuration logic
- Clear separation: interfaces vs implementations, output vs orchestration

**Semantic Improvements:**
- Renamed classes for clarity (`DemoOutput` → `SimulationOutput`)
- Bouncing behavior as demo configuration, not output type
- Factory patterns for clean object creation

**Benefits:**
- Single Responsibility Principle: Each file has one clear purpose
- Better maintainability: Output logic separate from demo logic
- Cleaner dependencies: Only essential packages included
- Physical Intuition: State represents actual physical quantities
- Simplified Physics: Vector operations instead of array manipulation
- **Extensibility**: Easy to add new body types (cars, airplanes, etc.) without changing physics

### Latest Improvements (2026)
**CLI Interface Refinement:**
- **Unified Command Structure**: Single `run` command with `--control` and `--output` options
- **Flexible Control Specification**: Control types specified as `type:param=value:param=value` format
- **Modular Output System**: Console, live visualization, and video outputs with consistent interface
- **Custom Starting Positions**: `--start-x`, `--start-y`, `--start-z` options for boundary testing

**Video Generation Enhancements:**
- **Automatic Frame Cleanup**: Temporary frame directories are automatically removed after successful video encoding
- **Improved Error Handling**: Failed video generation preserves frames for debugging
- **Workspace Management**: Clean workspace with no leftover temporary files
- **Performance**: Efficient FFmpeg integration with optimized encoding settings

**Bouncing Demo Capabilities:**
- **Real Movement Visualization**: Start from boundary center (z=5.0) to demonstrate actual collision and bounce
- **Linear Movement**: Strong force application (force=20.0) for visible acceleration and boundary impact
- **Collision Detection**: Event logging for boundary collisions with position tracking
- **Video Documentation**: MP4 output showing complete bouncing physics cycle

### Usage Examples

**CLI Commands:**
```bash
# Quick demonstrations
python demo.py run --control hovering --output console --steps 50    # Test stability
python demo.py run --control linear:force=2.0 --output console --steps 100  # Test forward movement
python demo.py run --control chaotic --output console --steps 150    # Test boundary interactions

# Video creation with automatic frame cleanup
python demo.py run --control sinusoidal --output video:filename=smooth.mp4:fps=10 --steps 100
python demo.py run --control chaotic --output video:filename=bounce.mp4:fps=15 --steps 200

# Live 3D visualization
python demo.py run --control linear:force=5.0 --output live --steps 100
```

**Programmatic API:**
```python
# Default configuration
env = DefaultBodyEnv()

# Custom physics with null visualization (for testing)
from physics import NewtonianPhysics
from visualization import NullVisualizer
physics = NewtonianPhysics({'mass': 2.0})
env = DefaultBodyEnv(physics_engine=physics, visualizer=NullVisualizer({}))

# Body-based configuration
from body import DefaultBody
hovercraft = DefaultBody(
    mass=1.5,
    lift_force_mean=12.0,
    friction_coefficient=0.05
)
env = DefaultBodyEnv(bodies=[hovercraft])

# State management with vector properties
from state import BodyState
import numpy as np

# Access body state
state = env.body.get_state()
position = state.r      # [x, y, z]
velocity = state.v      # [vx, vy, vz]

# Update body state
new_state = BodyState(r=np.array([1, 2, 3]), v=np.array([0.1, 0.2, 0.3]))
env.body.set_state(new_state)

# Modular demo system - combine any control source with any output
from control_sources import ControlSourceFactory
from simulation_outputs import VideoSimulationOutput

control = ControlSourceFactory.create_linear(force=10.0)
video_output = VideoSimulationOutput(env, "boundary_test.mp4", fps=10)
env.run_simulation(control, steps=100)

# Vector gravity (e.g., simulating wind effects)
from default_backend import NewtonianPhysics, DefaultBodyEnv
wind_physics = NewtonianPhysics({
    'gravity': [0.5, 0.0, -9.81],  # [x, y, z] gravity vector
    'bounds': [[-10, 10], [-10, 10], [0, 15]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
})
env = DefaultBodyEnv(physics_engine=wind_physics)

# Compact bounds configuration
compact_physics = NewtonianPhysics({
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
   - Add more realistic aerodynamic effects to default backend
   - Implement collision detection with obstacles (Genesis backend already supports this)
   - Add wind disturbances and environmental effects
   - Leverage Genesis backend for complex multi-body simulations
   - Compare and optimize physics accuracy between backends

4. **Performance Optimization**
   - Optimize visualization rendering for both backends
   - Add parallel simulation capabilities
   - GPU acceleration for Genesis backend
   - Benchmark and profile different backend performance

### Long Term
5. **Advanced Features**
   - Multi-agent scenarios
   - Terrain interaction
   - Sensor simulation (lidar, camera)

## Additional Resources

### Software Engineering Principles

**SOLID Principles:**
- [SOLID Principles - Robert C. Martin](https://blog.cleancoder.com/uncle-bob/2020/10/18/Solid-Relevance.html)
- [Design Principles and Design Patterns - Robert C. Martin](http://www.objectmentor.com/resources/articles/Principles_and_Patterns.pdf)

**Clean Architecture:**
- [Clean Architecture: A Craftsman's Guide to Software Structure - Robert C. Martin](https://www.amazon.com/Clean-Architecture-Craftsmans-Software-Structure/dp/0134494164)
- [The Clean Architecture - Robert C. Martin (Original Article)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)

**Design Patterns:**
- [Design Patterns: Elements of Reusable Object-Oriented Software - Gang of Four](https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612)
- [Head First Design Patterns - Eric Freeman & Elisabeth Robson](https://www.amazon.com/Head-First-Design-Patterns-Freeman/dp/0596007124)

### Technical Documentation

**Libraries Used:**
- [Open3D Documentation](https://www.open3d.org/docs/) - 3D visualization and processing
- [NumPy Documentation](https://numpy.org/doc/) - Vectorized mathematical operations
- [Click Documentation](https://click.palletsprojects.com/) - Command-line interface framework
- [FFmpeg Documentation](https://ffmpeg.org/documentation.html) - Video encoding

**Python Resources:**
- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [Python Typing Documentation](https://docs.python.org/3/library/typing.html)

### Physics and Simulation

**Physics Simulation:**
- [Game Physics Engine Development - Ian Millington](https://www.amazon.com/Game-Physics-Engine-Development-Commercial-Grade/dp/0123819768)
- [Real-Time Collision Detection - Christer Ericson](https://www.amazon.com/Real-Time-Collision-Detection-Interactive-Technology/dp/1558607323)

**Numerical Integration:**
- [Numerical Recipes: The Art of Scientific Computing - Press et al.](https://www.amazon.com/Numerical-Recipes-3rd-Scientific-Computing/dp/0521880688)
- [Runge-Kutta Methods - Wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)

### Reinforcement Learning

**Gymnasium (Formerly OpenAI Gym):**
- [Gymnasium Documentation](https://gymnasium.farama.org/) - RL environment standard
- [Reinforcement Learning: An Introduction - Sutton & Barto](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)

**RL Environments:**
- [Farama Foundation Environments](https://farama.org/) - Collection of RL environments
- [PettingZoo](https://pettingzoo.farama.org/) - Multi-agent RL environments

### Related Projects

**Similar Simulation Environments:**
- [PyBullet](https://pybullet.org/) - Physics simulation for robotics and RL
- [MuJoCo](https://mujoco.org/) - Advanced physics engine
- [Box2D](https://box2d.org/) - 2D physics engine
- [Bullet Physics](https://pybullet.org/) - 3D physics simulation

**Python Simulation Frameworks:**
- [Pygame](https://www.pygame.org/) - 2D game development
- [Pyglet](https://pyglet.org/) - Multimedia library
- [Panda3D](https://www.panda3d.org/) - 3D game engine

### Development Tools

**Version Control:**
- [Git Documentation](https://git-scm.com/doc)
- [Conventional Commits](https://conventionalcommits.org/)

**Python Development:**
- [Black - Code Formatter](https://black.readthedocs.io/)
- [isort - Import Sorter](https://pycqa.github.io/isort/)
- [mypy - Type Checker](https://mypy.readthedocs.io/)
- [pytest - Testing Framework](https://docs.pytest.org/)

### Academic Papers

**Relevant Research:**
- [Deep Reinforcement Learning - Mnih et al. (Nature, 2015)](https://www.nature.com/articles/nature14236)
- [Proximal Policy Optimization - Schulman et al. (2017)](https://arxiv.org/abs/1707.06347)
- [Soft Actor-Critic - Haarnoja et al. (2018)](https://arxiv.org/abs/1801.01290)

## Contributing

The physics simulation and modular demonstration system are complete and tested. The codebase follows clean architecture principles with separated concerns. Contributions welcome for:
- Additional control source implementations (new movement patterns)
- Alternative physics backends (extending the multi-backend architecture)
- Enhanced Genesis backend features and optimizations
- Alternative visualization backends
- Enhanced CLI features and commands
- RL integration and reward function design
- Additional physics features and environmental effects
- Performance improvements and optimizations
- Multi-agent scenarios and advanced features

**Architecture Notes:**
- Physics backends belong in separate modules (e.g., `genesis_backend.py`, `bullet_backend.py`)
- Output handlers belong in `simulation_outputs.py`
- Demo orchestration logic belongs in `demo_runner.py`
- Keep file-level separation of concerns when adding new features
- New backends should implement the abstract interfaces in `components.py`

## License

MIT License