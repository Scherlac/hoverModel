# Genesis Physics Engine - Sphere Collision Demo

## Overview

This demo showcases the Genesis physics simulation framework by implementing a multi-entity collision scenario. Two spheres with opposing initial velocities collide elastically while a complex hoverBody mesh falls under gravity, demonstrating realistic physics simulation with gravity, collision detection, momentum conservation, and support for both primitive shapes and imported 3D models.

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

### ✅ **Angular Momentum & Rotational Dynamics**
- **Initial Rotation**: Support for Euler angle initialization (`euler=(30.0, 0.0, 0.0)`)
- **Angular Velocity**: Full 3D rotational motion via DOF velocities [wx, wy, wz]
- **Quaternion Tracking**: Real-time orientation monitoring with `get_quat()`
- **Complex Rotations**: Combined linear and angular motion for realistic dynamics

### ✅ **Mesh Loading & Complex Geometry**
- **OBJ File Support**: Load complex 3D models using `gs.morphs.Mesh(file="path/to/model.obj")`
- **Automatic Processing**: Meshes are decimated, convexified, and optimized for physics
- **Scaling & Positioning**: Support for scale, position, and orientation transforms
- **Material Consistency**: Imported meshes use same Rigid material system as primitives

## Demo Description

### Setup
- **Scene**: Ground plane + two spheres (radius 0.1m) + hoverBody mesh
- **Initial Positions**:
  - Sphere 1: (0.0, 0.0, 0.5)
  - Sphere 2: (1.0, 0.0, 0.5)
- **HoverBody**: (0.0, 1.0, 0.2) - scaled to 0.1x size, rotated 30° on x-axis
- **Initial Velocities**:
  - Sphere 1: [2.0, 0.0, 0.0, 0.0, 0.0, 0.0] (moving right)
  - Sphere 2: [-2.0, 0.0, 0.0, 0.0, 0.0, 0.0] (moving left)
  - HoverBody: [0.0, 0.0, 0.0, 0.0, 0.0, 2.0] (spinning around z-axis)

### Physics Behavior
1. **Approach Phase**: Spheres move toward each other under gravity
2. **Collision Phase**: Elastic collision reverses velocities (steps 25-30)
3. **Separation Phase**: Spheres bounce apart with conserved momentum

### Output
The demo provides real-time position and velocity tracking for all entities:
```
Material Properties Analysis:
Sphere1 properties: {'material_type': 'Rigid', 'density': 200.0, 'mass': 0.838 kg, ...}
Sphere2 properties: {'material_type': 'Rigid', 'density': 200.0, 'mass': 0.838 kg, ...}
HoverBody properties: {'material_type': 'Rigid', 'density': 200.0, ...}

Step 20: Sphere1(idx=1) at (0.420, 0.000, 0.273) vel=(2.000, 0.000, -2.060)
         Sphere2(idx=2) at (0.580, 0.000, 0.273) vel=(-2.000, 0.000, -2.060)
         HoverBody(idx=3) at (-0.011, 1.024, 0.027) vel=(-0.160, 0.318, -1.088, -7.285, 4.458, -0.362)
         HoverBody orientation: quat=(0.979, 0.048, 0.071, 0.183)
Step 30: Sphere1(idx=1) at (0.373, 0.000, 0.086) vel=(-0.326, 0.001, 0.561)
         Sphere2(idx=2) at (0.626, -0.000, 0.085) vel=(0.322, -0.001, 0.570)
         HoverBody(idx=3) at (-0.015, 1.028, 0.017) vel=(-0.093, -0.012, 0.150, 1.282, 1.965, 0.472)
         HoverBody orientation: quat=(0.978, 0.049, -0.042, 0.198)
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

# Add mesh from OBJ file with initial rotation
hoverbody = scene.add_entity(
    gs.morphs.Mesh(
        file="assets/hoverBody_main.obj",
        scale=0.1,
        pos=(0.0, 1.0, 0.2),
        euler=(30.0, 0.0, 0.0)  # 30° rotation around x-axis
    )
)

# Build scene (required before setting properties)
scene.build()

# Set positions and velocities
sphere1.set_pos((0.0, 0.0, 0.5))
sphere1.set_dofs_velocity([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Set angular velocity for rotational motion
hoverbody.set_dofs_velocity([0.0, 0.0, 0.0, 0.0, 0.0, 2.0])  # Spin around z-axis

# Analyze material properties and track orientation
props = get_entity_properties(sphere1, radius=0.1)
quat = hoverbody.get_quat()  # Get current orientation

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
2. Create the collision scene with spheres and imported mesh
3. **Analyze and display material properties for all entities** (mass, density, elasticity)
4. Open an interactive 3D viewer
5. Run 100 simulation steps (1 second)
6. Display real-time position/velocity data for all three entities
7. Show collision physics in action with complex geometry

### Material Properties Analysis

The demo includes utility functions for material analysis of all entity types:

```python
# Extract material properties for spheres (with mass calculation)
sphere_props = get_entity_properties(sphere1, radius=0.1)
print(f"Sphere mass: {sphere_props['mass']:.3f} kg")
print(f"Density: {sphere_props['density']} kg/m³")

# Extract material properties for meshes (no mass calculation)
mesh_props = get_entity_properties(hoverbody)
print(f"Mesh density: {mesh_props['density']} kg/m³")
print(f"Coupling restitution: {mesh_props['coupling_restitution']}")

# Update material properties (must be done before scene.build())
update_entity_material(sphere1, rho=500.0, coup_restitution=0.8)
```

**Material Analysis Output:**
- **Spheres**: Mass = 0.838 kg, Density = 200.0 kg/m³, Volume = 0.00419 m³
- **Meshes**: Same Rigid material properties, mass calculation requires known volume
- **Elasticity**: Coefficient of restitution ≈ 0.16 (actual collision behavior)

## Future Applications

This demo establishes a foundation for:
- **Robotics Simulation**: Testing control algorithms with realistic physics and complex geometries
- **Game Development**: Physics-based gameplay with imported 3D assets
- **Research**: Differentiable physics for machine learning with real-world models
- **Engineering**: Prototyping mechanical systems with collision dynamics and CAD imports
- **Mixed Reality**: Combining primitive shapes with complex imported meshes

## Files
- `sphere_collision_demo.py` - Complete working demo with collision physics
- `README.md` - This documentation

---

*Genesis Version: 0.3.13 | Tested on: Windows 11 | Backend: CPU*</content>
<parameter name="filePath">c:\01_dev\hoverModel\spike\genesis-demo\README.md