import numpy as np
import time
import os
from main import HovercraftEnv

def run_tests():
    """Run physics tests without visualization."""
    print("Running physics tests...")

    # Test hovering
    print("\nTesting hovering...")
    env = HovercraftEnv(viz=False)
    for i in range(100):
        env.step([0, 0])  # No control
        if i % 25 == 0:
            x, y, z = env.state[:3]
            print(f"Step {i}: Position ({x:.2f}, {y:.2f}, {z:.2f})")
    env.close()

    # Test movement
    print("\nTesting movement...")
    env = HovercraftEnv(viz=False)
    env.state[0] = -3  # Start offset
    for i in range(100):
        env.step([1, 0])  # Forward force
        if i % 25 == 0:
            x, y, z = env.state[:3]
            print(f"Step {i}: Position ({x:.2f}, {y:.2f}, {z:.2f})")
    env.close()

    # Test rotation
    print("\nTesting rotation...")
    env = HovercraftEnv(viz=False)
    for i in range(100):
        env.step([0, 0.5])  # Rotation torque
        if i % 25 == 0:
            theta = env.state[3]
            print(f"Step {i}: Rotation {theta:.2f}")
    env.close()

    print("Tests completed.")

def create_video():
    """Create demonstration video."""
    print("Creating demo video...")

    env = HovercraftEnv(viz=True)

    # Set camera
    ctr = env.vis.get_view_control()
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
            env.vis.capture_screen_image(frame_path)
            print(f"Captured frame {frame_count}")
            frame_count += 1

        time.sleep(0.02)

    env.close()

    # Create video with ffmpeg
    try:
        os.system(f'ffmpeg -y -framerate 20 -i {frames_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p hovercraft_demo.mp4')
        print("Video created: hovercraft_demo.mp4")
    except:
        print("FFmpeg not available, frames saved in 'frames' directory")

    # Cleanup
    import shutil
    shutil.rmtree(frames_dir)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "video":
        create_video()
    else:
        run_tests()