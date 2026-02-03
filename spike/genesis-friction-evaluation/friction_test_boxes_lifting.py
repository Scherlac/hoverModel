#!/usr/bin/env python3
"""
Genesis Friction Evaluation Test - BOXES LIFTING FORCE VARIATION
===============================================================

This test evaluates how lifting forces proportional to mass affect friction behavior.
We test boxes with different masses and apply lifting forces of 20%, 40%, and 60%
of each box's mass, while keeping friction coefficient constant.

Test Design:
- Fixed box dimensions: 0.2 x 0.05 x 0.2 (same as density test)
- Fixed friction coefficient: 0.5
- Variable densities: 0.5, 1.0, 2.0, 4.0 kg/m³ (masses: 0.001, 0.002, 0.004, 0.008 kg)
- Variable lifting forces: 20%, 40%, 60% of each box's mass
- Same initial conditions for all tests
"""

import numpy as np
import json
import os
from datetime import datetime
import genesis as gs

#box dimensions (fixed for all)
width, depth, height = 0.2, 0.2, 0.05
gravity = 9.81

def create_test_scene_with_lifting_forces(densities, friction_coeff=0.5):
    """Create a scene with boxes having different masses and proportional lifting forces."""

    # Initialize Genesis
    gs.init(backend=gs.cpu)

    # Scene setup
    scene = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=0.01))

    # Ground plane
    ground = scene.add_entity(gs.morphs.Plane())

    boxes = []
    box_info = []

    # Box dimensions (same as density test)
    volume = width * height * depth

    density_count = len(densities)
    # Create boxes with different densities and lifting forces
    for i, density in enumerate(densities):
        mass = volume * density
        gravity_force = mass * gravity

        # Create lifting forces: 20%, 40%, 60% of mass converted to Newtons using gravity
        lifting_forces = [0.1 * i * gravity_force for i in [1,2,3,4,5,6]]  # 10%,20%,30%,40%,50%,60%
        lift_count = len(lifting_forces)

        for j, lift_force in enumerate(lifting_forces):
            # Position boxes spread along y only
            y_pos = ( i * lift_count + j ) * 1.2 * width  # Spread along y axis only

            # Create box entity
            box = scene.add_entity(
                gs.morphs.Box(
                    size=(width, depth, height),
                    pos=(0, y_pos, height/2 + 0.2),
                ),
                material=gs.materials.Rigid(
                    rho=density,
                    friction=friction_coeff
                )
            )

            boxes.append(box)
            box_info.append({
                'density': density,
                'mass': mass,
                'gravity_force': gravity_force,
                'lift_force': lift_force,
                'lift_percentage': (lift_force / gravity_force) * 100,
                'initial_pos': [0, y_pos, height/2 + 0.01],
                'entity_idx': len(boxes) - 1
            })

            print(f"Created box with density {density:.1f}, mass {mass:.4f} kg, lift force {lift_force:.4f} N ({(lift_force/gravity_force)*100:.0f}% of gravity force)")

    # Build the scene before setting velocities/forces
    scene.build()

    # Now set velocities and forces on built entities
    for box, info in zip(boxes, box_info):
        # Set initial velocity (forward motion)
        # Genesis uses DOF velocity [vx, vy, vz, wx, wy, wz]
        dofs_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        box.set_dofs_velocity(dofs_velocity)

        # Apply constant upward lifting force
        # Genesis uses DOF forces [fx, fy, fz, tx, ty, tz]
        lift_force = info['lift_force']
        force_vector = np.array([0.0, 0.0, lift_force, 0.0, 0.0, 0.0])
        box.control_dofs_force(force_vector)

    return scene, boxes, box_info

def run_simulation(scene, boxes, box_info, steps=30, settling_steps=20):
    """Run the simulation with settling phase and collect data."""

    # Data collection
    results = []

    for i, (box, info) in enumerate(zip(boxes, box_info)):
        result = {
            'box_id': i,
            'density': info['density'],
            'mass': info['mass'],
            'lift_force': info['lift_force'],
            'lift_percentage': info['lift_percentage'],
            'positions': [],
            'velocities': [],
            'forces': []
        }
        results.append(result)

    print(f"Starting settling phase for {settling_steps} steps...")

    # Settling phase - let boxes reach equilibrium vertical position
    for step in range(settling_steps):
        # Reapply forces to each box (forces need to be maintained each step)
        for box, info in zip(boxes, box_info):
            lift_force = info['lift_force']
            force_vector = np.array([0.0, 0.0, lift_force, 0.0, 0.0, 0.0])
            box.control_dofs_force(force_vector)

        scene.step()

    print(f"Settling phase complete. Starting velocity application and measurement...")

    # Record settled positions (before applying velocity)
    for i, (box, result) in enumerate(zip(boxes, results)):
        pos = box.get_pos()
        vel = box.get_dofs_velocity()[:3]  # Get only translational velocity
        force = box.get_dofs_force()[:3] if hasattr(box, 'get_dofs_force') else [0, 0, result['lift_force']]

        result['positions'].append(pos)
        result['velocities'].append(vel)
        result['forces'].append(force)

        # Store the settled vertical position
        result['settled_vertical_pos'] = pos[2]


    # Measurement phase - boxes are now sliding with applied velocity
    for step in range(steps):
        # Reapply forces to each box (forces need to be maintained each step)
        for box, info in zip(boxes, box_info):

            if step < 20:
                # Apply horizontal velocity (keep vertical velocity from settling)
                # Maintain only the lifting force after initial velocity application
                lift_force = info['lift_force']
                acceleration = 1.4 * info['gravity_force']
                force_vector = np.array([acceleration, 0.0, lift_force, 0.0, 0.0, 0.0])
                box.control_dofs_force(force_vector)
            else:
                # Maintain only the lifting force after initial velocity application
                lift_force = info['lift_force']
                force_vector = np.array([0.0, 0.0, lift_force, 0.0, 0.0, 0.0])
                box.control_dofs_force(force_vector)

            

        scene.step()

        # Collect data for each box
        for i, (box, result) in enumerate(zip(boxes, results)):
            pos = box.get_pos()
            vel = box.get_dofs_velocity()[:3]  # Get only translational velocity
            force = box.get_dofs_force()[:3] if hasattr(box, 'get_dofs_force') else [0, 0, result['lift_force']]

            result['positions'].append(pos)
            result['velocities'].append(vel)
            result['forces'].append(force)

    return results

def analyze_results(results):
    """Analyze the simulation results."""

    print("\nANALYSIS RESULTS:")
    print("=" * 60)

    # Group by density (mass)
    density_groups = {}
    for result in results:
        density = result['density']
        if density not in density_groups:
            density_groups[density] = []
        density_groups[density].append(result)

    # Analyze each density group
    for density, group_results in density_groups.items():
        mass = group_results[0]['mass']
        print(f"\nDensity {density:.1f} kg/m³ (Mass {mass:.4f} kg):")

        for result in sorted(group_results, key=lambda x: x['lift_percentage']):
            lift_pct = result['lift_percentage']
            final_pos = np.array(result['positions'][-1])
            final_vel = np.array(result['velocities'][-1])
            initial_pos = np.array(result['positions'][0])
            initial_vel = np.array(result['velocities'][0])

            distance_x = final_pos[0] - initial_pos[0]
            vel_reduction_x = initial_vel[0] - final_vel[0]

            print(f"  Lift {lift_pct:.0f}%: Distance={distance_x:.3f}, Final Vel x={final_vel[0]:.3f}, Vel Reduction={vel_reduction_x:.3f}")

def save_results(results, box_info, timestamp, simulation_steps, settling_steps):
    """Save results to JSON file."""

    # Create results directory
    results_dir = f"results_boxes_lifting_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Save detailed results
    results_file = os.path.join(results_dir, "lifting_force_test_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'test_info': {
                'description': 'Box lifting force variation test with settling phase',
                'box_dimensions': [width, depth, height],
                'friction_coefficient': 0.5,
                'densities_tested': [0.5, 1.0, 2.0, 4.0],
                'lifting_force_percentages': [20, 40, 60],
                'simulation_steps': simulation_steps,
                'settling_steps': settling_steps,
                'timestamp': timestamp
            },
            'box_info': box_info,
            'results': [{
                'box_id': r['box_id'],
                'density': r['density'],
                'mass': r['mass'],
                'lift_force': r['lift_force'],
                'lift_percentage': r['lift_percentage'],
                'settled_vertical_pos': float(r.get('settled_vertical_pos', 0)),
                'positions': [pos.tolist() if hasattr(pos, 'tolist') else pos for pos in r['positions']],
                'velocities': [vel.tolist() if hasattr(vel, 'tolist') else vel for vel in r['velocities']],
                'forces': [force.tolist() if hasattr(force, 'tolist') else force for force in r['forces']]
            } for r in results]
        }, f, indent=2)

    # Create summary text file
    summary_file = os.path.join(results_dir, f"lifting_force_test_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("GENESIS FRICTION EVALUATION - BOX LIFTING FORCE VARIATION TEST\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Test Date: {timestamp}\n\n")
        f.write("TEST DESIGN:\n")
        f.write(f"- Box dimensions: {width} x {depth} x {height} (fixed)\n")
        f.write("- Friction coefficient: 0.5 (fixed)\n")
        f.write("- Densities tested: 0.5, 1.0, 2.0, 4.0 kg/m³\n")
        f.write("- Masses: 0.001, 0.002, 0.004, 0.008 kg\n")
        f.write("- Lifting forces: 20%, 40%, 60% of each box's gravitational force\n")
        f.write("- Initial conditions: Same position/velocity for all\n\n")

        f.write("RESULTS SUMMARY:\n")
        f.write("-" * 40 + "\n")

        # Group and summarize results
        density_groups = {}
        for result in results:
            density = result['density']
            if density not in density_groups:
                density_groups[density] = []
            density_groups[density].append(result)

        for density in sorted(density_groups.keys()):
            mass = density_groups[density][0]['mass']
            f.write(f"\nDensity {density:.1f} kg/m³ (Mass {mass:.4f} kg):\n")

            for result in sorted(density_groups[density], key=lambda x: x['lift_percentage']):
                lift_pct = result['lift_percentage']
                final_pos = np.array(result['positions'][-1])
                final_vel = np.array(result['velocities'][-1])
                initial_pos = np.array(result['positions'][0])
                distance_x = final_pos[0] - initial_pos[0]
                vel_reduction_x = result['velocities'][0][0] - final_vel[0]

                f.write(f"  Lift {lift_pct:.0f}%: Distance={distance_x:.3f}, Final Vel x={final_vel[0]:.3f}, Vel Reduction={vel_reduction_x:.3f}\n")

        f.write("\nCONCLUSION: [To be filled based on analysis]\n")
        f.write(f"\nResults saved to: {results_dir}/\n")

    print(f"\nResults saved to {results_dir}/")

def main():
    """Main test execution."""

    print("Genesis Friction Evaluation Test - BOXES LIFTING FORCE VARIATION")
    print("=" * 65)

    # Test parameters
    densities = [0.2, 1.0, 5.0, 25.0]  # kg/m³
    friction_coeff = 0.01
    simulation_steps = 300
    settling_steps = 40  # Steps to let boxes settle to equilibrium position

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Creating test scene with boxes of different masses and proportional lifting forces...")

    # Create scene
    scene, boxes, box_info = create_test_scene_with_lifting_forces(densities, friction_coeff)

    print("Created 12 boxes with varying masses and lifting forces")
    print(f"Box dimensions: {width} x {depth} x {height} (FIXED for all)")
    print(f"Friction coefficient: {friction_coeff} (FIXED for all)")
    print(f"Densities: {densities}")
    print(f"Lifting forces: 20%, 40%, 60% of each box's gravitational force")
    print(f"Test protocol: {settling_steps} settling steps, then {simulation_steps} measurement steps")

    # Run simulation
    print("\nRunning simulation...")
    results = run_simulation(scene, boxes, box_info, simulation_steps, settling_steps)

    # Analyze results
    analyze_results(results)

    # Save results
    save_results(results, box_info, timestamp, simulation_steps, settling_steps)

    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()