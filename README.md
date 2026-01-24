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
- **Separated architecture** with clear file-level separation between outputs and demos
- **Clean state representation** with vector-based `BodyState` class (r/v vectors, theta/omega scalars)
- **Vectorized physics** using natural mathematical objects instead of array slicing
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

This installs numpy, Open3D for the physics simulation and 3D visualization, and Click for the CLI interface.

## Project Structure

- `main.py` - Core hovercraft environment with physics simulation and Open3D visualization
- `demo.py` - Combined testing and video generation functionality using modular architecture
- `control_sources.py` - Control signal generators for different movement patterns
- `simulation_outputs.py` - Output handlers for various demonstration modes (logging, video, etc.)
- `demo_runner.py` - Demo orchestration and configuration logic
- `physics.py` - Physics engine with abstract `PhysicsEngine` base class
- `visualization.py` - Visualization backends with abstract `Visualizer` base class
- `environment.py` - Main environment orchestrating physics and visualization
- `body.py` - Physical body representations with `Body` and `Hovercraft` classes
- `state.py` - Clean vector-based state representation with `BodyState` class
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
python demo.py video hover        # Create hovering video
python demo.py video linear       # Create linear movement video
python demo.py video rotate       # Create rotational video
python demo.py video sinusoid     # Create sinusoidal video
python demo.py video chaotic      # Create boundary bouncing video
```

### Advanced Options
```bash
python demo.py --help             # Show all available commands and options
python demo.py hover --steps 100  # Custom number of steps
python demo.py linear --force 1.5 # Custom force for linear movement
python demo.py video linear --output my_demo.mp4 --fps 30 --steps 150
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
The demo system features a comprehensive Click-based CLI for easy experimentation:

**Available Commands:**
- `hover` - Hovering (zero control inputs)
- `linear` - Constant forward thrust
- `rotate` - Pure rotational movement
- `sinusoid` - Combined sinusoidal motion
- `chaotic` - High-amplitude boundary testing

**Video Subcommands:**
- `video hover` - Create hovering video
- `video linear` - Create linear movement video
- `video rotate` - Create rotational video
- `video sinusoid` - Create sinusoidal video
- `video chaotic` - Create boundary bouncing video

**Common Commands:**
```bash
# Run demonstrations
python demo.py hover              # Quick hover test
python demo.py linear --force 1.0 # Linear with custom force
python demo.py chaotic --steps 100 # Extended chaotic test

# Create videos
python demo.py video sinusoid     # Standard video
python demo.py video chaotic      # Boundary bouncing video
python demo.py video linear --output custom.mp4 --fps 30

# Get help
python demo.py --help
python demo.py video --help
```

# Get help
python demo.py --help
```

## Architecture

The codebase follows SOLID principles with a highly composable, modular architecture:

### Components
- **`body.py`**: Physical body representations
  - `Body` - Abstract base class for physical bodies with mass, shape, and dynamics
  - `Hovercraft` - Concrete hovercraft implementation with lifting force and control characteristics
  - Clean separation of physical properties from state and physics calculations
- **`state.py`**: State representation and management
  - `BodyState` - Clean vector-based state with r/v vectors and theta/omega scalars
  - Single source of truth for state format and operations
  - Provides semantic vector accessors and validation
- **`physics.py`**: Physics engine with abstract `PhysicsEngine` base class
  - `NewtonianPhysics` - General Newtonian physics for any body type
  - Multi-body physics support with interaction handling
  - Vectorized calculations for performance
  - `HoveringControl` - Zero control signals for stability testing
  - `LinearMovementControl` - Constant forward thrust
  - `RotationalControl` - Pure rotational movement
  - `SinusoidalControl` - Combined sinusoidal motion
  - `ChaoticControl` - Boundary testing with high-amplitude signals
  - `ControlSourceFactory` - Factory pattern for creating control sources
- **`simulation_outputs.py`**: Output handlers with abstract `SimulationOutput` base class
  - `NullSimulationOutput` - Silent testing mode
  - `LoggingSimulationOutput` - Console position reporting
  - `VideoSimulationOutput` - MP4 video generation with Open3D visualization
- **`demo_runner.py`**: Demo orchestration and configuration
  - `DemoRunner` - Main demo orchestrator combining control sources with outputs
  - `BouncingVideoDemo` - Specialized configuration for boundary collision videos
- **`physics.py`**: Physics engine with abstract `PhysicsEngine` base class
- **`visualization.py`**: Visualization backends with abstract `Visualizer` base class
- **`environment.py`**: Main environment orchestrating physics and visualization
- **`demo.py`**: Demonstration and testing utilities using the modular system

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

### Modular Demo System
The demonstration system uses composition to combine control sources with output handlers, accessible through both programmatic API and CLI:

**Programmatic Usage:**
```python
from control_sources import ControlSourceFactory
from demo_runner import DemoRunner

# Create control source
control = ControlSourceFactory.create_sinusoidal()

# Create demo runner
runner = DemoRunner()

# Run physics test with logging
runner.run_test(control, steps=50)

# Create video demonstration
runner.create_video(control, "demo.mp4", steps=200, fps=25)

# Create boundary bouncing video (uses bouncing=True parameter)
chaotic_control = ControlSourceFactory.create_chaotic()
runner.create_video(chaotic_control, "bounce.mp4", steps=300, bouncing=True)
```

**CLI Usage:**
```bash
# Equivalent CLI commands
python demo.py sinusoid --steps 50
python demo.py video sinusoid --steps 200 --fps 25 --output demo.mp4
python demo.py video chaotic --steps 300 --output bounce.mp4
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
- **Vector Representation**: Natural mathematical objects (vectors, scalars) instead of flat arrays
- **Physical Intuition**: State represents actual kinematic quantities
- **Body Association**: States are tied to specific body instances
- **Encapsulation**: State operations are centralized in one class
- **Validation**: Automatic validation of state values
- **Persistence**: Load/save state to/from JSON files
- **Backward Compatibility**: `__array__()` method for existing visualization code
- **Type Safety**: Clear interfaces for state operations

**Usage:**
```python
from body import Hovercraft
from state import BodyState
import numpy as np

# Create a hovercraft body
hovercraft = Hovercraft(mass=1.0, lift_force_mean=10.0)

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
- **`Hovercraft` Class**: Concrete implementation with hovercraft-specific properties
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
- **`body.py`**: Physical body representations and properties
- **`simulation_outputs.py`**: Pure output handling (logging, video generation)
- **`demo_runner.py`**: Demo orchestration and configuration logic
- Clear separation: "what to output" vs "how to configure demos"

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
from physics import NewtonianPhysics
from visualization import NullVisualizer
physics = NewtonianPhysics({'mass': 2.0})
env = HovercraftEnv(physics_engine=physics, visualizer=NullVisualizer({}))

# Body-based configuration
from body import Hovercraft
hovercraft = Hovercraft(
    mass=1.5,
    lift_force_mean=12.0,
    friction_coefficient=0.05
)
env = HovercraftEnv(bodies=[hovercraft])

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
from demo_runner import DemoRunner

runner = DemoRunner()

# Test different movement patterns
hovering = ControlSourceFactory.create_hovering()
runner.run_test(hovering, steps=50)

linear = ControlSourceFactory.create_linear(forward_force=1.0)
runner.run_test(linear, steps=50)

chaotic = ControlSourceFactory.create_chaotic()
runner.create_video(chaotic, "boundary_test.mp4", steps=300, bouncing=True)

# Vector gravity (e.g., simulating wind effects)
wind_physics = NewtonianPhysics({
    'gravity': [0.5, 0.0, -9.81],  # [x, y, z] gravity vector
    'bounds': [[-10, 10], [-10, 10], [0, 15]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
})
env = HovercraftEnv(physics_engine=wind_physics)

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
- Alternative visualization backends
- Enhanced CLI features and commands
- RL integration and reward function design
- Additional physics features and environmental effects
- Performance improvements and optimizations
- Multi-agent scenarios and advanced features

**Architecture Notes:**
- Output handlers belong in `simulation_outputs.py`
- Demo orchestration logic belongs in `demo_runner.py`
- Keep file-level separation of concerns when adding new features

## License

MIT License