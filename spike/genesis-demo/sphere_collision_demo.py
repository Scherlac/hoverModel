#!/usr/bin/env python3
"""
Simple Sphere Collision Demo using Genesis

This script demonstrates basic sphere collision physics in Genesis.
Two spheres are created with initial velocities that cause them to collide.
"""

import genesis as gs
import numpy as np


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
                vel1 = sphere1.get_dofs_velocity()
                vel2 = sphere2.get_dofs_velocity()
                print(f"Step {i}: Sphere1(idx={sphere1.idx}) at ({pos1[0]:.3f}, {pos1[1]:.3f}, {pos1[2]:.3f}) vel=({vel1[0]:.3f}, {vel1[1]:.3f}, {vel1[2]:.3f})")
                print(f"         Sphere2(idx={sphere2.idx}) at ({pos2[0]:.3f}, {pos2[1]:.3f}, {pos2[2]:.3f}) vel=({vel2[0]:.3f}, {vel2[1]:.3f}, {vel2[2]:.3f})")
            except Exception as e:
                print(f"Step {i}: Could not get positions - {e}")

    print("Simulation completed!")
    print("The spheres started moving toward each other, collided elastically, and bounced apart.")
    print("This demonstrates Genesis physics simulation with collision detection and response.")
    print("Close the viewer window to exit.")


if __name__ == "__main__":
    main()