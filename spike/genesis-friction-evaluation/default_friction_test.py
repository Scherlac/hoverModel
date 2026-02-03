#!/usr/bin/env python3
"""
Default Backend Friction Test

This script tests friction behavior in the default physics backend
to establish expected behavior for comparison.
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from default_backend import DefaultBodyEnv


def test_default_friction():
    """Test friction in default backend."""

    print("Default Backend Friction Test")
    print("=" * 30)

    # Test different friction coefficients
    friction_coeffs = [0.1, 0.5, 1.0, 2.0]

    results = {}

    for friction in friction_coeffs:
        print(f"\nTesting friction coefficient: {friction}")

        # Create environment
        env = DefaultBodyEnv()

        # Get the body
        body = env.bodies[0]

        # Set friction
        body.set_friction(friction)

        # Set initial conditions
        initial_velocity = np.array([2.0, 0.0, 0.0])
        body.set_velocity(initial_velocity)
        body.state.r = np.array([0.0, 0.0, 1.0])

        initial_pos = body.state.r.copy()

        # Run simulation
        env.run_simulation_with_controls([], steps=30)

        final_pos = body.state.r
        final_vel = body.state.v

        distance_traveled = final_pos[0] - initial_pos[0]
        velocity_reduction = initial_velocity[0] - final_vel[0]

        results[friction] = {
            'distance_traveled': distance_traveled,
            'velocity_reduction': velocity_reduction,
            'final_velocity_x': final_vel[0],
            'initial_velocity_x': initial_velocity[0]
        }

        print(".3f")
        print(".3f")
        print(".3f")

    # Summary
    print("\n" + "=" * 50)
    print("DEFAULT BACKEND FRICTION SUMMARY")
    print("=" * 50)

    distances = [data['distance_traveled'] for data in results.values()]
    min_dist = min(distances)
    max_dist = max(distances)
    distance_range = max_dist - min_dist

    if distance_range > 0.01:
        print("✓ Friction is working in default backend!")
        print(".3f")
        for friction, data in sorted(results.items()):
            print(".3f")
    else:
        print("✗ Friction may not be working in default backend either")

    return results


if __name__ == "__main__":
    test_default_friction()