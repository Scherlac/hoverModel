"""
Demo output handlers for different demonstration modes.
Follows SOLID principles with composition and low coupling.
"""

from abc import ABC, abstractmethod
import numpy as np
import time
import os
from typing import Optional, Tuple
from environment import HovercraftEnv
from physics import HovercraftPhysics
from visualization import NullVisualizer, Visualizer
from control_sources import ControlSource


class DemoOutput(ABC):
    """Abstract base class for demo output handlers."""

    def __init__(self, env: HovercraftEnv):
        self.env = env
        self.step_count = 0

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the output handler."""
        pass

    @abstractmethod
    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        """Process a single simulation step."""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Finalize and cleanup the output handler."""
        pass

    def run_demo(self, control_source: ControlSource, steps: int) -> None:
        """Run complete demo with given control source."""
        self.initialize()

        for step in range(steps):
            control = control_source.get_control(step)
            self.env.step(control)
            self.process_step(step, control)

        self.finalize()


class NullOutput(DemoOutput):
    """Null output for testing - no visualization or logging."""

    def initialize(self) -> None:
        pass

    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        pass

    def finalize(self) -> None:
        pass


class LoggingOutput(DemoOutput):
    """Logging output for physics testing with periodic position reports."""

    def __init__(self, env: HovercraftEnv, log_interval: int = 25):
        super().__init__(env)
        self.log_interval = log_interval

    def initialize(self) -> None:
        print(f"Starting demo with {self.env.physics.__class__.__name__}")

    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        if step % self.log_interval == 0:
            pos = self.env.position
            print(f"Step {step}: Position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    def finalize(self) -> None:
        print("Demo completed.")


class VideoOutput(DemoOutput):
    """Video output with Open3D visualization and frame capture."""

    def __init__(self, env: HovercraftEnv, video_name: str, fps: int = 25):
        super().__init__(env)
        self.video_name = video_name
        self.fps = fps
        self.frames_dir = "frames"
        self.frame_count = 0

    def initialize(self) -> None:
        # Camera setup
        if hasattr(self.env.visualizer, 'vis'):
            ctr = self.env.visualizer.vis.get_view_control()
            ctr.set_zoom(0.8)
            ctr.set_front([0.7, 0.3, 0.5])
            ctr.set_lookat([0, 0, 1])
            ctr.set_up([0, 0, 1])

        os.makedirs(self.frames_dir, exist_ok=True)
        print(f"Creating video: {self.video_name}")
        print(f"Frames will be saved to: {os.path.abspath(self.frames_dir)}/")
        print(f"Frames directory exists: {os.path.exists(self.frames_dir)}")

    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        if step % max(1, 25 // self.fps) == 0:  # Frame capture rate
            frame_path = f"{self.frames_dir}/frame_{self.frame_count:04d}.png"
            print(f"Capturing frame {self.frame_count} at step {step}: {frame_path}")
            try:
                self.env.capture_frame(frame_path)
                self.frame_count += 1
                print(f"✅ Frame {self.frame_count-1} captured successfully: {os.path.exists(frame_path)}")
            except Exception as e:
                # If frame capture fails, skip this frame
                print(f"Warning: Frame capture failed at step {step}: {e}")

        time.sleep(0.02)  # Control simulation speed

    def finalize(self) -> None:
        # Create video with ffmpeg
        success = False
        try:
            result = os.system(f'ffmpeg -y -framerate {self.fps} -i {self.frames_dir}/frame_%04d.png '
                             f'-vf "scale=1920:1080" -c:v libx264 -pix_fmt yuv420p {self.video_name}')
            if result == 0:
                print(f"✅ Video created successfully: {self.video_name}")
                success = True
            else:
                print(f"❌ FFmpeg failed with exit code {result}")
        except Exception as e:
            print(f"❌ FFmpeg error: {e}")

        # Cleanup - only remove frames if video creation succeeded
        import shutil
        if success and os.path.exists(self.frames_dir):
            shutil.rmtree(self.frames_dir)
            print(f"🧹 Cleaned up temporary frames directory")
        elif not success:
            print(f"📁 Frames saved in '{self.frames_dir}' directory for debugging")


class BouncingVideoOutput(VideoOutput):
    """Specialized video output for boundary bouncing demonstrations."""

    def initialize(self) -> None:
        # Special camera setup for boundary viewing
        if hasattr(self.env.visualizer, 'vis'):
            ctr = self.env.visualizer.vis.get_view_control()
            ctr.set_zoom(0.6)
            ctr.set_front([0.8, 0.2, 0.4])
            ctr.set_lookat([0, 0, 1.5])
            ctr.set_up([0, 0, 1])

        # Set initial conditions for bouncing
        self.env.state[0] = 4.9  # Near boundaries
        self.env.state[1] = 4.9
        self.env.state[2] = 9.8
        self.env.state[4] = 0.5  # Initial velocity
        self.env.state[5] = 0.5
        self.env.state[6] = 0.2

        os.makedirs(self.frames_dir, exist_ok=True)
        print(f"Creating bouncing video: {self.video_name}")

    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        if step % 2 == 0:  # More frequent capture for smoother video
            frame_path = f"{self.frames_dir}/frame_{self.frame_count:04d}.png"
            self.env.capture_frame(frame_path)
            self.frame_count += 1

        time.sleep(0.015)  # Faster simulation for bouncing


class DemoRunner:
    """Composes control sources with output handlers for demonstrations."""

    def __init__(self, physics_config: Optional[dict] = None):
        self.physics_config = physics_config or {}

    def create_environment(self, visualizer: Optional[Visualizer] = None) -> HovercraftEnv:
        """Create environment with specified visualizer."""
        if visualizer is None:
            visualizer = NullVisualizer({})
        return HovercraftEnv(visualizer=visualizer)

    def run_test(self, control_source: ControlSource, steps: int = 50) -> None:
        """Run physics test with logging output."""
        env = self.create_environment()
        output = LoggingOutput(env)
        output.run_demo(control_source, steps)
        env.close()

    def create_video(self, control_source: ControlSource, video_name: str,
                    steps: int = 200, fps: int = 25) -> None:
        """Create demonstration video."""
        # For video creation, we need a visualizer that can capture frames
        # Try to use Open3D visualizer, fall back to creating placeholder frames
        bounds = self.physics_config.get('bounds', [[-5, 5], [-5, 5], [0, 10]])
        try:
            from visualization import Open3DVisualizer
            # Convert bounds list to dict format expected by visualizer
            bounds_dict = {
                'x': (bounds[0][0], bounds[0][1]),
                'y': (bounds[1][0], bounds[1][1]),
                'z': (bounds[2][0], bounds[2][1])
            }
            visualizer = Open3DVisualizer(bounds_dict)
        except (ImportError, Exception) as e:
            print(f"Open3D visualizer not available ({e}), using null visualizer")
            from visualization import NullVisualizer
            visualizer = NullVisualizer(bounds)

        env = HovercraftEnv(visualizer=visualizer)
        output = VideoOutput(env, video_name, fps)
        output.run_demo(control_source, steps)
        env.close()

    def create_bouncing_video(self, control_source: ControlSource, video_name: str,
                             steps: int = 300) -> None:
        """Create boundary bouncing demonstration video."""
        # For video creation, we need a visualizer that can capture frames
        bounds = self.physics_config.get('bounds', [[-5, 5], [-5, 5], [0, 10]])
        try:
            from visualization import Open3DVisualizer
            # Convert bounds list to dict format expected by visualizer
            bounds_dict = {
                'x': (bounds[0][0], bounds[0][1]),
                'y': (bounds[1][0], bounds[1][1]),
                'z': (bounds[2][0], bounds[2][1])
            }
            visualizer = Open3DVisualizer(bounds_dict)
        except (ImportError, Exception) as e:
            print(f"Open3D visualizer not available ({e}), using null visualizer")
            from visualization import NullVisualizer
            visualizer = NullVisualizer(bounds)

        env = HovercraftEnv(visualizer=visualizer)
        output = BouncingVideoOutput(env, video_name)
        output.run_demo(control_source, steps)
        env.close()