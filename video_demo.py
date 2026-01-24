import numpy as np
import time
import os
from main import HovercraftEnv

def create_demo_video_open3d():
    """Create a video demo using Open3D screen capture."""
    print("Creating demo video using Open3D screen capture...")

    # Create environment with visualization
    env = HovercraftEnv(viz=True)

    # Set up camera view for good demo perspective
    ctr = env.vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0.5, 0.5, -0.7])  # Angled view
    ctr.set_lookat([0, 0, 1])        # Look at center
    ctr.set_up([0, 0, 1])            # Up direction

    # Create frames directory
    frames_dir = "frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # Run simulation and capture frames
    frame_count = 0
    for i in range(300):  # Shorter for demo
        # Vary actions over time
        forward = 1.0 if i < 150 else -1.0
        rotation = 0.5 * np.sin(i * 0.1)
        action = [forward, rotation]
        env.step(action)

        # Capture frame every few steps
        if i % 3 == 0:  # Capture every 3rd frame for smoother video
            try:
                # Capture screen image directly to file
                frame_path = f"{frames_dir}/frame_{frame_count:04d}.png"
                env.vis.capture_screen_image(frame_path)
                print(f"Captured frame {frame_count}")
                frame_count += 1
            except Exception as e:
                print(f"Failed to capture frame {frame_count}: {e}")

        time.sleep(0.02)

    env.close()

    # Create video from frames using ffmpeg (if available)
    try:
        import subprocess
        output_file = "hovercraft_demo_open3d.mp4"
        # Add scale filter to ensure even dimensions for H.264
        cmd = [
            "ffmpeg", "-y", "-framerate", "15", "-i", f"{frames_dir}/frame_%04d.png",
            "-vf", "scale=1920:1060",  # Make height even (1061 -> 1060)
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", output_file
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Video saved as {output_file}")

            # Clean up frames
            import shutil
            shutil.rmtree(frames_dir)
            print("Cleaned up temporary frames")
        else:
            print(f"FFmpeg failed: {result.stderr}")
            print("Frames saved in 'frames/' directory for manual processing")

    except (subprocess.CalledProcessError, FileNotFoundError, ImportError):
        print("FFmpeg not available. Frames saved in 'frames/' directory.")
        print("To create video manually: ffmpeg -framerate 15 -i frames/frame_%04d.png -vf scale=1920:1060 -c:v libx264 -pix_fmt yuv420p hovercraft_demo.mp4")

    print("Demo video creation completed.")

if __name__ == "__main__":
    create_demo_video_open3d()