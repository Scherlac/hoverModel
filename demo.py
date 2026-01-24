import numpy as np
import time
import os
from environment import HovercraftEnv
from physics import HovercraftPhysics
from visualization import NullVisualizer

def run_tests():
    """Run physics tests without visualization."""
    print("Running physics tests...")

    # Test with null visualizer for pure physics testing
    physics = HovercraftPhysics({})
    env = HovercraftEnv(physics_engine=physics, visualizer=NullVisualizer({}))

    # Test hovering
    print("\nTesting hovering...")
    env.reset()
    for i in range(100):
        env.step([0, 0])  # No control
        if i % 25 == 0:
            pos = env.position
            print(f"Step {i}: Position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    env.close()

    # Test movement
    print("\nTesting movement...")
    env.reset()
    env.state[0] = -3  # Start offset
    for i in range(100):
        env.step([1, 0])  # Forward force
        if i % 25 == 0:
            pos = env.position
            print(f"Step {i}: Position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    env.close()

    # Test rotation
    print("\nTesting rotation...")
    env.reset()
    for i in range(100):
        env.step([0, 0.5])  # Rotation torque
        if i % 25 == 0:
            print(f"Step {i}: Rotation {env.orientation:.2f}")
    env.close()

    print("Tests completed.")

def create_video():
    """Create demonstration video."""
    print("Creating demo video...")

    # Use default configuration (will create Open3D visualizer)
    env = HovercraftEnv()

    # Set camera if Open3D visualizer is available
    if hasattr(env.visualizer, 'vis'):
        ctr = env.visualizer.vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0.5, 0.5, -0.7])
        ctr.set_lookat([0, 0, 1])
        ctr.set_up([0, 0, 1])

    # Create frames directory
    frames_dir = "frames"
    os.makedirs(frames_dir, exist_ok=True)

    # Run simulation and capture
    frame_count = 0
    for i in range(200):
        forward = 1.0 if i < 100 else -1.0
        rotation = 0.3 * np.sin(i * 0.1)
        env.step([forward, rotation])

        if i % 3 == 0:
            frame_path = f"{frames_dir}/frame_{frame_count:04d}.png"
            env.capture_frame(frame_path)
            print(f"Captured frame {frame_count}")
            frame_count += 1

        time.sleep(0.02)

    env.close()

    # Create video with ffmpeg
    try:
        os.system(f'ffmpeg -y -framerate 20 -i {frames_dir}/frame_%04d.png -vf "scale=1920:1060" -c:v libx264 -pix_fmt yuv420p hovercraft_demo.mp4')
        print("Video created: hovercraft_demo.mp4")
    except:
        print("FFmpeg not available, frames saved in 'frames' directory")

    # Cleanup
    import shutil
    shutil.rmtree(frames_dir)

def demo_composition():
    """Demonstrate component composition and vectorized configuration."""
    print("Demonstrating component composition...")

    # Example 1: Custom physics with heavier hovercraft
    custom_config = {
        'mass': 2.0,  # Heavier hovercraft
        'lift_mean': 20.0,  # Stronger lift
        'friction_k': 0.05  # Less friction
    }
    physics = HovercraftPhysics(custom_config)
    env = HovercraftEnv(physics_engine=physics, visualizer=NullVisualizer({}))

    print("Testing with custom physics (heavier, stronger lift, less friction)...")
    env.reset()
    for i in range(50):
        env.step([0.5, 0])  # Small forward force
        if i % 10 == 0:
            pos = env.position
            print(f"Step {i}: Position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    env.close()

    # Example 2: Vector gravity (simulating wind or different gravity direction)
    print("\nTesting with vector gravity (simulating horizontal wind)...")
    wind_config = {
        'mass': 1.0,
        'gravity': [0.5, 0.0, -9.81],  # Horizontal wind + vertical gravity
        'bounds': [[-10, 10], [-10, 10], [0, 15]]  # Simplified array format
    }
    physics_wind = HovercraftPhysics(wind_config)
    env_wind = HovercraftEnv(physics_engine=physics_wind, visualizer=NullVisualizer({}))

    env_wind.reset()
    for i in range(30):
        env_wind.step([0, 0])  # No control, let wind affect motion
        if i % 10 == 0:
            pos = env_wind.position
            print(f"Step {i}: Position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    env_wind.close()

    # Example 3: Compact bounds configuration (same as example 2 but showing it's the same)
    print("\nTesting with compact bounds array...")
    compact_config = {
        'bounds': [[-3, 3], [-3, 3], [0, 5]]  # Clean array format: [x_bounds, y_bounds, z_bounds]
    }
    physics_compact = HovercraftPhysics(compact_config)
    env_compact = HovercraftEnv(physics_engine=physics_compact, visualizer=NullVisualizer({}))

    env_compact.reset()
    env_compact.state[0] = -2.5  # Start near boundary
    for i in range(20):
        env_compact.step([0.8, 0.2])  # Forward force with rotation
        if i % 5 == 0:
            pos = env_compact.position
            print(f"Step {i}: Position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    env_compact.close()

    print("Composition demo completed.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "video":
            create_video()
        elif sys.argv[1] == "compose":
            demo_composition()
        else:
            print("Usage: python demo.py [video|compose]")
            print("  (no args) - run physics tests")
            print("  video     - create demonstration video")
            print("  compose   - demonstrate component composition")
    else:
        run_tests()