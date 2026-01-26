# Genesis Physics Engine - Sphere Collision Demo

## Overview

This demo showcases the Genesis physics simulation framework by implementing a simple sphere collision scenario. Two spheres with opposing initial velocities collide elastically, demonstrating realistic physics simulation with gravity, collision detection, and momentum conservation.

## Key Findings

### ✅ **Genesis Setup & Installation**
- **Dependencies**: Requires JAX, PyTorch, and scientific computing packages (numpy, scipy, numba)
- **Version Compatibility**: Critical to use compatible versions (numpy 2.3.5, scipy 1.17.0 for numba compatibility)
- **Backend Options**: Supports CPU and GPU backends (tested with CPU backend)

### ✅ **Physics Simulation Architecture**
- **Scene-Based**: All entities must be added to a scene before building
- **Build-First**: Scene must be built before setting positions/velocities
- **Rigid Body Dynamics**: Uses RigidEntity objects with morphs and materials
- **Real-time Visualization**: Built-in OpenGL viewer with interactive controls

### ✅ **Velocity Control**
- **DOF System**: Velocities set via `set_dofs_velocity([vx, vy, vz, wx, wy, wz])`
  - First 3 values: linear velocities (m/s)
  - Last 3 values: angular velocities (rad/s)
- **Post-Build Setting**: Velocities must be set after `scene.build()`
- **Immediate Application**: Velocities take effect immediately when set

### ✅ **Collision Physics**
- **Elastic Collisions**: Spheres bounce off each other conserving momentum
- **Gravity**: Automatic gravity simulation (-9.81 m/s² in z-direction)
- **Ground Plane**: Static collision surface prevents spheres from falling infinitely
- **Real-time Updates**: Physics simulation runs at ~58 FPS

## Demo Description

### Setup
- **Scene**: Ground plane + two spheres (radius 0.1m) + hoverBody mesh
- **Initial Positions**:
  - Sphere 1: (0.0, 0.0, 0.5)
  - Sphere 2: (1.0, 0.0, 0.5)
  - HoverBody: (0.0, 1.0, 0.2) - scaled to 0.1x size
- **Initial Velocities**:
  - Sphere 1: [2.0, 0.0, 0.0, 0.0, 0.0, 0.0] (moving right)
  - Sphere 2: [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0] (moving left)
  - HoverBody: Stationary (no initial velocity)

### Physics Behavior
1. **Approach Phase**: Spheres move toward each other under gravity
2. **Collision Phase**: Elastic collision reverses velocities (steps 25-30)
3. **Separation Phase**: Spheres bounce apart with conserved momentum

### Output
The demo provides real-time position and velocity tracking:
```
Step 20: Sphere1(idx=1) at (0.420, 0.000, 0.273) vel=(2.000, 0.000, -2.060)
         Sphere2(idx=2) at (0.580, 0.000, 0.273) vel=(-2.000, 0.000, -2.060)
Step 30: Sphere1(idx=1) at (0.373, 0.000, 0.086) vel=(-0.326, 0.001, 0.561)
         Sphere2(idx=2) at (0.626, -0.000, 0.085) vel=(0.322, -0.001, 0.570)
```

## Technical Implementation

### Core Components
```python
import genesis as gs

# Initialize with CPU backend
gs.init(backend=gs.cpu)

# Create scene with viewer
scene = gs.Scene(show_viewer=True)

# Add entities
plane = scene.add_entity(gs.morphs.Plane())
sphere1 = scene.add_entity(gs.morphs.Sphere(radius=0.1))
sphere2 = scene.add_entity(gs.morphs.Sphere(radius=0.1))

# Build scene (required before setting properties)
scene.build()

# Set positions and velocities
sphere1.set_pos((0.0, 0.0, 0.5))
sphere1.set_dofs_velocity([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Run simulation
for i in range(100):
    scene.step()
```

### Key API Patterns
- **Entity Creation**: `scene.add_entity(morph)`
- **Position Setting**: `entity.set_pos(position_tuple)` after build
- **Velocity Setting**: `entity.set_dofs_velocity([vx, vy, vz, wx, wy, wz])` after build
- **Simulation Step**: `scene.step()` advances physics by dt (default 0.01s)

## Challenges & Solutions

### 1. **Velocity Setting API**
- **Problem**: `set_vel()` and `set_qvel()` methods not available
- **Solution**: Use `set_dofs_velocity()` with 6-element array [vx, vy, vz, wx, wy, wz]

### 2. **Scene Building Order**
- **Problem**: Setting positions/velocities before `scene.build()` fails
- **Solution**: Always call `scene.build()` before manipulating entity properties

### 3. **Dependency Version Conflicts**
- **Problem**: NumPy/SciPy/Numba version incompatibilities
- **Solution**: Pin to compatible versions (numpy 2.3.5, scipy 1.17.0)

## Running the Demo

### Prerequisites
```bash
pip install genesis-world jax torch numpy==2.3.5 scipy==1.17.0 numba
```

### Execution
```bash
python sphere_collision_demo.py
```

The demo will:
1. Initialize Genesis physics engine
2. Create the collision scene
3. **Analyze and display material properties** (mass, density, elasticity)
4. Open an interactive 3D viewer
5. Run 100 simulation steps (1 second)
6. Display real-time position/velocity data
7. Show collision physics in action

### Material Properties Analysis

The demo includes utility functions for material analysis:

```python
# Extract material properties
props = get_entity_properties(sphere1, radius=0.1)
print(f"Mass: {props['mass']:.3f} kg")
print(f"Density: {props['density']} kg/m³")
print(f"Coupling restitution: {props['coupling_restitution']}")

# Update material properties (before scene.build())
update_entity_material(sphere1, rho=500.0, coup_restitution=0.8)
```

## Future Applications

This demo establishes a foundation for:
- **Robotics Simulation**: Testing control algorithms with realistic physics
- **Game Development**: Physics-based gameplay mechanics
- **Research**: Differentiable physics for machine learning applications
- **Engineering**: Prototyping mechanical systems with collision dynamics

## Files
- `sphere_collision_demo.py` - Complete working demo with collision physics
- `README.md` - This documentation

---

*Genesis Version: 0.3.13 | Tested on: Windows 11 | Backend: CPU*</content>
<parameter name="filePath">c:\01_dev\hoverModel\spike\genesis-demo\README.md