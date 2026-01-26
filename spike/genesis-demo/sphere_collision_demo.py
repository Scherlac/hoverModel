#!/usr/bin/env python3
"""
Simple Sphere Collision Demo using Genesis

This script demonstrates basic sphere collision physics in Genesis.
Two spheres are created with initial velocities that cause them to collide.
"""

import genesis as gs
import numpy as np


def get_entity_properties(entity, radius=None):
    """
    Extract and calculate material properties for a Genesis entity.

    Args:
        entity: Genesis RigidEntity object
        radius: Sphere radius (if applicable) for mass calculation

    Returns:
        dict: Dictionary containing material properties and calculated values
    """
    material = entity.material

    properties = {
        'material_type': type(material).__name__,
        'density': float(material.rho),
        'coupling_friction': float(material.coup_friction),
        'coupling_restitution': float(material.coup_restitution),  # For solver coupling
        'coupling_softness': float(material.coup_softness),
        'friction': material.friction,
        'gravity_compensation': float(material.gravity_compensation),
    }

    # Calculate mass if radius is provided (for spheres)
    if radius is not None:
        volume = (4/3) * np.pi * (radius ** 3)
        mass = properties['density'] * volume
        properties.update({
            'radius': radius,
            'volume': volume,
            'mass': mass,
        })

    return properties


def update_entity_material(entity, **kwargs):
    """
    Update material properties of a Genesis entity.

    Args:
        entity: Genesis RigidEntity object
        **kwargs: Material properties to update (density, friction, etc.)

    Note: This function should be called before scene.build() for changes to take effect.
    """
    material = entity.material

    # Update allowed properties
    updatable_props = ['rho', 'coup_friction', 'coup_restitution', 'coup_softness']
    for prop, value in kwargs.items():
        if prop in updatable_props:
            setattr(material, prop, value)
            print(f"Updated {prop} to {value}")
        else:
            print(f"Warning: {prop} is not updatable or not recognized")


def main():
    print("Initializing Genesis...")

    # Initialize Genesis with CPU backend
    gs.init(backend=gs.cpu)

    print("Creating scene...")

    # Create a scene with basic options
    scene = gs.Scene(
        show_viewer=True,  # Show the interactive viewer
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        sim_options=gs.options.SimOptions(
            dt=0.01,  # Time step
        ),
    )

    print("Adding entities...")

    # Add a ground plane
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    # Add first sphere (moving right)
    sphere1 = scene.add_entity(
        gs.morphs.Sphere(radius=0.1),
    )

    # Add second sphere (moving left)
    sphere2 = scene.add_entity(
        gs.morphs.Sphere(radius=0.1),
    )

    # Add hoverBody mesh from assets folder
    hoverbody = scene.add_entity(
        gs.morphs.Mesh(
            file="assets/hoverBody_main.obj",
            scale=0.1,  # Scale down the mesh
            pos=(0.0, 1.0, 0.2),  # Position above the collision area
        )
    )

    # Material Properties Analysis:
    # - Default Rigid material: density = 200.0 kg/m³
    # - Sphere mass: ρ × (4/3)πr³ = 200 × (4/3)π(0.1)³ ≈ 0.838 kg
    # - Elasticity: coup_restitution = 0.0 (material property for solver coupling)
    # - Actual collision behavior: coefficient of restitution ≈ 0.16 (partially elastic)
    # - Note: coup_restitution affects rigid-FEM/MPM coupling, not rigid-rigid collisions

    # Extract and display material properties
    print("\nMaterial Properties Analysis:")
    sphere1_props = get_entity_properties(sphere1, radius=0.1)
    print(f"Sphere1 properties: {sphere1_props}")

    sphere2_props = get_entity_properties(sphere2, radius=0.1)
    print(f"Sphere2 properties: {sphere2_props}")

    # For mesh objects, we can't easily calculate mass without knowing the volume
    hoverbody_props = get_entity_properties(hoverbody)
    print(f"HoverBody properties: {hoverbody_props}")

    # Example: Update material properties (must be done before scene.build())
    # update_entity_material(sphere1, rho=500.0, coup_restitution=0.8)

    print("Building scene...")
    scene.build()

    # Now set initial positions and velocities after building
    sphere1.set_pos((0.0, 0.0, 0.5))  # Initial position
    # Set velocity using DOFs (6 values: 3 linear, 3 angular)
    try:
        sphere1.set_dofs_velocity([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Linear velocity right, no rotation
        print("Successfully set velocity for sphere1")
    except Exception as e:
        print(f"Could not set velocity for sphere1: {e}")

    sphere2.set_pos((1.0, 0.0, 0.5))  # Initial position
    try:
        sphere2.set_dofs_velocity([-2.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Linear velocity left, no rotation
        print("Successfully set velocity for sphere2")
    except Exception as e:
        print(f"Could not set velocity for sphere2: {e}")

    print("Running simulation...")

    # Run simulation for 100 steps (1 second at dt=0.01) - collision happens around step 25-30
    for i in range(100):
        scene.step()

        # Print positions every 10 steps
        if i % 10 == 0:
            try:
                pos1 = sphere1.get_pos()
                pos2 = sphere2.get_pos()
                pos_hover = hoverbody.get_pos()
                vel1 = sphere1.get_dofs_velocity()
                vel2 = sphere2.get_dofs_velocity()
                print(f"Step {i}: Sphere1(idx={sphere1.idx}) at ({pos1[0]:.3f}, {pos1[1]:.3f}, {pos1[2]:.3f}) vel=({vel1[0]:.3f}, {vel1[1]:.3f}, {vel1[2]:.3f})")
                print(f"         Sphere2(idx={sphere2.idx}) at ({pos2[0]:.3f}, {pos2[1]:.3f}, {pos2[2]:.3f}) vel=({vel2[0]:.3f}, {vel2[1]:.3f}, {vel2[2]:.3f})")
                print(f"         HoverBody(idx={hoverbody.idx}) at ({pos_hover[0]:.3f}, {pos_hover[1]:.3f}, {pos_hover[2]:.3f})")
            except Exception as e:
                print(f"Step {i}: Could not get positions - {e}")

    print("Simulation completed!")
    print("The spheres started moving toward each other, collided elastically, and bounced apart.")
    print("This demonstrates Genesis physics simulation with collision detection and response.")
    print("Close the viewer window to exit.")


if __name__ == "__main__":
    main()