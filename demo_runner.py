"""
Demo runner for orchestrating different hovercraft demonstrations.
Handles demo configuration and execution using simulation outputs.
"""

from typing import Optional, Tuple
from simulation_outputs import (
    SimulationOutput,
    LoggingSimulationOutput,
    VideoSimulationOutput,
    NullSimulationOutput
)
from environment import HovercraftEnv
from visualization import NullVisualizer, Visualizer
from control_sources import ControlSource


class DemoRunner:
    """Orchestrates different hovercraft demonstrations with various outputs."""

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
        output = LoggingSimulationOutput(env)
        output.run_simulation(control_source, steps)
        env.close()

    def create_video(self, control_source: ControlSource, video_name: str,
                    steps: int = 200, fps: int = 25, bouncing: bool = False) -> None:
        """Create demonstration video."""
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

        if bouncing:
            output = BouncingVideoDemo(env, video_name, fps)
        else:
            output = VideoSimulationOutput(env, video_name, fps)

        output.run_simulation(control_source, steps)
        env.close()


class BouncingVideoDemo(VideoSimulationOutput):
    """Demo configuration for boundary bouncing videos."""

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

        super().initialize()
        print(f"Creating bouncing video: {self.video_name}")

    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        if step % 2 == 0:  # More frequent capture for smoother video
            frame_path = f"{self.frames_dir}/frame_{self.frame_count:04d}.png"
            try:
                self.env.capture_frame(frame_path)
                self.frame_count += 1
                print(f"✅ Frame {self.frame_count-1} captured successfully")
            except Exception as e:
                print(f"Warning: Frame capture failed at step {step}: {e}")

        import time
        time.sleep(0.015)  # Faster simulation for bouncing