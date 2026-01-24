# Software Engineering Considerations

This document captures the software engineering principles, architectural decisions, and design patterns implemented in the hovercraft simulation project.

## Table of Contents

1. [SOLID Principles](#solid-principles)
2. [Clean Architecture](#clean-architecture)
3. [Design Patterns](#design-patterns)
4. [Cohesion and Coupling](#cohesion-and-coupling)
5. [State Management](#state-management)
6. [Performance Considerations](#performance-considerations)
7. [Testing and Maintainability](#testing-and-maintainability)
8. [API Design](#api-design)
9. [File Organization](#file-organization)
10. [Evolution and Refactoring](#evolution-and-refactoring)

## SOLID Principles

### Single Responsibility Principle (SRP)
Each class has one reason to change and one primary responsibility:

- **`Body`**: Physical properties and force calculations
- **`Hovercraft`**: Hovercraft-specific properties and behaviors
- **`BodyState`**: Kinematic state representation and operations
- **`HovercraftEnv`**: Orchestrates physics and visualization of bodies
- **`NewtonianPhysics`**: Numerical physics integration for any body type
- **`ControlSource` subclasses**: Generate specific control patterns
- **`SimulationOutput` subclasses**: Handle specific output formats

### Open/Closed Principle (OCP)
Classes are open for extension but closed for modification:

- **Strategy Pattern**: New control sources and outputs can be added without modifying existing code
- **Abstract Base Classes**: `PhysicsEngine`, `Visualizer`, `ControlSource`, `SimulationOutput`
- **Factory Pattern**: `ControlSourceFactory` enables adding new control types

### Liskov Substitution Principle (LSP)
Subclasses can be substituted for their base classes:

- All `ControlSource` implementations can be used interchangeably
- All `SimulationOutput` implementations work with the same interface
- `BodyState` maintains compatibility with existing code via `__array__()` method

### Interface Segregation Principle (ISP)
Clients depend only on methods they actually use:

- Separate interfaces for physics, visualization, control, and output
- Environment depends on abstractions, not concrete implementations
- Modular composition allows mixing and matching components

### Dependency Inversion Principle (DIP)
High-level modules don't depend on low-level modules:

- Environment depends on `PhysicsEngine` and `Visualizer` abstractions
- Demo runner depends on `ControlSource` and `SimulationOutput` abstractions
- Dependency injection enables testability and flexibility

## Clean Architecture

### Component Organization

**Core Business Logic:**
- `body.py`: Physical body definitions and properties (innermost layer)
- `physics.py`: Physics simulation algorithms
- `state.py`: State representation and data structures

**Application Logic:**
- `environment.py`: Orchestrates bodies and physics
- `demo_runner.py`: Orchestrates demonstrations

**Interface Adapters:**
- `control_sources.py`: Input adapters (control signal generation)
- `simulation_outputs.py`: Output adapters (result handling)
- `visualization.py`: UI adapters (3D rendering)

**External Interfaces:**
- Open3D, NumPy, FFmpeg (frameworks and drivers)

### Dependency Flow
```
Demo Runner → Environment → Bodies + Physics + State
Demo Runner → Outputs → Visualization
Demo Runner → Control Sources → Environment
```

All dependencies point inward toward the core business logic. Bodies define their own properties, physics operates on them, and the environment orchestrates their interactions.

## Design Patterns

### Strategy Pattern
Interchangeable algorithms for control and output:

```python
# Control strategies
control = ControlSourceFactory.create_hovering()  # Zero input strategy
control = ControlSourceFactory.create_linear()    # Constant force strategy
control = ControlSourceFactory.create_chaotic()   # Boundary testing strategy

# Output strategies
output = LoggingSimulationOutput(env)      # Console logging
output = VideoSimulationOutput(env)        # MP4 video generation
output = NullSimulationOutput(env)         # Silent operation
```

### Factory Pattern
Clean object creation for complex hierarchies:

```python
# Control source factory
hovering = ControlSourceFactory.create_hovering()
linear = ControlSourceFactory.create_linear(force=1.5)
chaotic = ControlSourceFactory.create_chaotic()

# Environment factory methods
env = HovercraftEnv()  # Uses defaults
env = HovercraftEnv(physics_engine=custom_physics, visualizer=null_vis)
```

### Composition over Inheritance
Flexible component combination:

```python
# Compose any control source with any output
demo = DemoRunner()
demo.run_test(hovering_control, steps=50)        # Test with logging
demo.create_video(chaotic_control, "test.mp4")   # Video with boundary testing
```

### Dependency Injection
Loose coupling through constructor injection:

```python
# Inject physics engine
physics = HovercraftPhysics({'mass': 2.0})
env = HovercraftEnv(physics_engine=physics)

# Inject visualizer
visualizer = NullVisualizer(bounds)
env = HovercraftEnv(visualizer=visualizer)
```

## Cohesion and Coupling

### High Cohesion Examples

**Environment (`environment.py`):**
- Single responsibility: orchestrate physics + visualization
- Related methods: `step()`, `reset()`, `render()`, `close()`
- State ownership: manages state lifecycle
- 150 lines, focused scope

**State (`state.py`):**
- Single responsibility: state representation + operations
- Related methods: `save()`, `load()`, `to_dict()`, `from_dict()`
- Encapsulation: all state logic in one place
- 55 lines, complete state management

**Control Sources (`control_sources.py`):**
- Single responsibility: input signal generation
- Related classes: all implement `ControlSource` interface
- Factory pattern: clean creation of control strategies

### Low Coupling Examples

**Abstract Interfaces:**
- `PhysicsEngine`, `Visualizer`, `ControlSource`, `SimulationOutput`
- Environment depends on abstractions, not implementations
- Enables testing with mocks and easy component swapping

**Dependency Injection:**
- Components receive dependencies via constructor
- No hard-coded dependencies or global state
- Testable in isolation

**Composition:**
- Components composed at runtime
- No inheritance hierarchies tying components together
- Flexible combinations: any control + any output

## State Management

### Single Source of Truth
- `BodyState` class owns all kinematic state representation
- States are associated with specific `Body` instances
- Environment manages collections of bodies and their states
- Physics engine operates on bodies to produce new states

### Vector-Based Representation
```python
class BodyState:
    def __init__(self, r=None, v=None, theta=0.0, omega=0.0):
        self.r = r if r is not None else np.zeros(3)  # position vector
        self.v = v if v is not None else np.zeros(3)  # velocity vector
        self.theta = theta                            # orientation scalar
        self.omega = omega                            # angular velocity scalar
```

**Benefits:**
- Physical intuition: represents actual kinematic quantities
- Type safety: clear vector/scalar distinctions
- Performance: vectorized NumPy operations
- Maintainability: no array indexing magic numbers

### Body-State Relationship
- `Body` objects encapsulate physical properties (mass, forces, shape)
- `BodyState` represents current kinematic state (position, velocity, orientation)
- Clean separation: "what a body is" vs "current state of motion"
- Environment manages body lifecycle and state updates

### Backward Compatibility
- `__array__()` method maintains compatibility with existing visualization code
- Environment provides delegation properties for legacy access
- Gradual migration path for legacy code

## Performance Considerations

### Vectorized Physics
- NumPy array operations instead of scalar loops
- Efficient tensor computations for state updates
- Single vectorized boundary collision detection

### Memory Management
- Lazy initialization of visualizers
- Efficient state representation (8 floats total)
- Minimal object creation in simulation loops

### Computational Efficiency
- Physics calculations: O(1) per time step
- Visualization: Optional, can be disabled for testing
- Video generation: Frame capture only when needed

## Testing and Maintainability

### Testability Design
- Dependency injection enables mock testing
- Abstract interfaces allow for test doubles
- Pure functions where possible (physics calculations)
- Isolated component testing

### Code Organization
- Clear file-level separation of concerns
- Consistent naming conventions
- Comprehensive docstrings and type hints
- Modular imports, no circular dependencies

### Error Handling
- Graceful degradation (Open3D → Null visualizer)
- Informative error messages
- Resource cleanup in `close()` methods

## API Design

### Fluent Interfaces
```python
# Method chaining for configuration
physics = HovercraftPhysics({
    'mass': 2.0,
    'bounds': [[-10, 10], [-10, 10], [0, 15]]
})

# Builder pattern for complex objects
state = BodyState(
    r=np.array([1, 2, 3]),
    v=np.array([0.1, 0.2, 0.3]),
    theta=0.5,
    omega=0.1
)
```

### Consistent Naming
- `step()`: Advance simulation by one time step
- `reset()`: Return to initial state
- `render()`: Update visualization
- `close()`: Clean up resources

### Type Safety
- Full type hints throughout codebase
- NumPy array type annotations
- Abstract base class contracts

## File Organization

### File-Level Cohesion
- **`main.py`**: Entry point and basic visualization
- **`demo.py`**: CLI interface and demo orchestration
- **`environment.py`**: Core environment orchestration
- **`physics.py`**: Physics simulation logic
- **`state.py`**: State representation and persistence
- **`control_sources.py`**: Input generation strategies
- **`simulation_outputs.py`**: Output handling strategies
- **`demo_runner.py`**: Demo configuration and execution
- **`visualization.py`**: 3D rendering backends

### Import Structure
- Clear dependency hierarchy
- No circular imports
- Minimal import statements per file
- Relative imports within package

## Evolution and Refactoring

### Architectural Improvements Made

**Body-Physics Separation (Latest):**
- **`Body` Abstract Class**: Base class for physical bodies with mass, shape, and force calculations
- **`Hovercraft` Concrete Class**: Specific body type with lifting force and control characteristics
- **Physics Engine Refactoring**: `NewtonianPhysics` now works with any `Body` object
- **Multi-Body Support**: Environment can contain multiple interacting bodies
- **Force Delegation**: Bodies define their own `get_forces()` methods
- **Clean Separation**: "What a body is" vs "How physics works"

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

### Design Principles Applied

**Single Responsibility Principle (SRP):**
- `Body`: Physical properties and force calculations
- `BodyState`: Kinematic state representation
- `PhysicsEngine`: Numerical integration and constraints
- `Environment`: Body management and orchestration

**Open/Closed Principle (OCP):**
- New body types can be added without changing physics engine
- New control sources/outputs without changing core logic
- Abstract base classes enable extension through implementation

**Interface Segregation Principle (ISP):**
- `Body.get_forces()`: Bodies define their own physics
- `PhysicsEngine.step()`: Physics operates on body abstractions
- Clean interfaces prevent coupling between layers

**Dependency Inversion Principle (DIP):**
- High-level environment depends on body/physics abstractions
- Concrete implementations can be swapped without changing interface
- Testable through dependency injection

**YAGNI (You Aren't Gonna Need It):**
- Simple interfaces over complex abstractions
- Minimal viable architecture for current requirements

**DRY (Don't Repeat Yourself):**
- Single `Body` class instead of duplicate state logic
- Factory patterns eliminate repetitive object creation
- Abstract base classes reduce code duplication

**KISS (Keep It Simple, Stupid):**
- Vector-based state instead of complex class hierarchies
- Composition over inheritance for flexibility
- Clear, focused responsibilities per component

---

This document serves as a reference for the architectural decisions and software engineering principles applied throughout the hovercraft simulation project. These patterns ensure maintainable, testable, and extensible code that follows industry best practices.</content>
<parameter name="filePath">c:\01_dev\hoverModel\SOFTWARE_ENGINEERING.md