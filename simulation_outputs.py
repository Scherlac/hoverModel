"""
Simulation output handlers for different output modes.
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


class SimulationOutput(ABC):
    """Abstract base class for simulation output handlers."""

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

    def run_simulation(self, control_source: ControlSource, steps: int) -> None:
        """Run complete simulation with given control source."""
        self.initialize()

        for step in range(steps):
            control = control_source.get_control(step)
            self.env.step(control)
            self.process_step(step, control)

        self.finalize()


class NullSimulationOutput(SimulationOutput):
    """Null output for testing - no visualization or logging."""

    def initialize(self) -> None:
        pass

    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        pass

    def finalize(self) -> None:
        pass


class LoggingSimulationOutput(SimulationOutput):
    """Logging output for physics testing with periodic position reports."""

    def __init__(self, env: HovercraftEnv, log_interval: int = 25):
        super().__init__(env)
        self.log_interval = log_interval

    def initialize(self) -> None:
        print(f"Starting simulation with {self.env.physics.__class__.__name__}")

    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        if step % self.log_interval == 0:
            pos = self.env.position
            print(f"Step {step}: Position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    def finalize(self) -> None:
        print("Simulation completed.")


class VideoSimulationOutput(SimulationOutput):
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


class LiveVisualizationOutput(SimulationOutput):
    """Live visualization output with interactive Open3D display."""

    def __init__(self, env: HovercraftEnv):
        super().__init__(env)

    def initialize(self) -> None:
        print("🎮 Starting live visualization...")
        print("Press 'q' or close the window to exit")

    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        # Update visualization
        self.env.visualizer.update(self.env.state)
        self.env.visualizer.render()

        # Small delay for smooth visualization
        time.sleep(0.05)

        # Check if window is still open
        if hasattr(self.env.visualizer, 'vis'):
            if not self.env.visualizer.vis.poll_events():
                raise KeyboardInterrupt("Visualization window closed")

    def finalize(self) -> None:
        print("✅ Live visualization completed")
        self.env.visualizer.close()