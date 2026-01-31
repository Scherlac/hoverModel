"""
Demo runner for orchestrating different hovercraft demonstrations.
Handles demo configuration and execution using simulation outputs.

State vector format (8 elements):
[x, y, z, theta, vx, vy, vz, omega_z]
- x, y, z: position coordinates
- theta: orientation angle (radians)
- vx, vy, vz: velocity components
- omega_z: angular velocity around z-axis
"""

from typing import Optional, Tuple
import numpy as np

from components import (
    Environment,
    SimulationOutput
)

from simulation_outputs import (
    LoggingSimulationOutput,
    VideoSimulationOutput,
    LiveVisualizationOutput,
    NullSimulationOutput
)

from default_backend import (
    DefaultBodyEnv,
    Open3DVisualizer,
)

from control_sources import ControlSource


class DemoRunner:
    """Orchestrates different hovercraft demonstrations with various outputs."""

    def __init__(self, physics_config: Optional[dict] = None):
        self.physics_config = physics_config or {}

    def create_environment(self) -> DefaultBodyEnv:
        """Create environment."""
        return DefaultBodyEnv()

    def run_test(self, control_source: ControlSource, steps: int = 50, initial_pos: tuple = None) -> None:
        """Run physics test with logging output."""
        env = self.create_environment()
        
        # Set initial position if provided
        if initial_pos is not None:
            import numpy as np
            env.state.r = np.array(initial_pos)
            env.state.v = np.zeros(3)  # Reset velocity
            env.state.theta = 0.0
            env.state.omega = 0.0
            env.state.clear_events()
        
        output = LoggingSimulationOutput(env)
        env.run_simulation(control_source, [output], steps)
        env.close()

    def run_visualization(self, control_source: ControlSource, steps: int = 200, initial_pos: tuple = None) -> None:
        """Run demo with live visualization."""
        # Create environment with Open3D visualizer
        bounds = self.physics_config.get('bounds', [[-5, 5], [-5, 5], [0, 10]])
        bounds_dict = {
            'x': (bounds[0][0], bounds[0][1]),
            'y': (bounds[1][0], bounds[1][1]),
            'z': (bounds[2][0], bounds[2][1])
        }

        try:
            from default_backend import Open3DVisualizer
            env = DefaultBodyEnv()
            visualizer = Open3DVisualizer(env)
            env.register_visualizer(visualizer)
        except (ImportError, Exception) as e:
            print(f"Open3D visualizer not available ({e}), cannot run live visualization")
            return

        # Set initial position if provided
        if initial_pos is not None:
            import numpy as np
            env.state.r = np.array(initial_pos)
            env.state.v = np.zeros(3)  # Reset velocity
            env.state.theta = 0.0
            env.state.omega = 0.0
            env.state.clear_events()

        output = LiveVisualizationOutput(env)

        try:
            env.run_simulation(control_source, [output], steps, initial_pos)
        except KeyboardInterrupt:
            print("Visualization interrupted by user")
        finally:
            env.close()

    def create_video(self, control_source: ControlSource, video_name: str,
                    steps: int = 200, fps: int = 25, bouncing: bool = False) -> None:
        """Create demonstration video."""
        # For video creation, we need a visualizer that can capture frames
        bounds = self.physics_config.get('bounds', [[-5, 5], [-5, 5], [0, 10]])
        try:
            from default_backend import Open3DVisualizer
            env = DefaultBodyEnv()
            visualizer = Open3DVisualizer(env)
            env.register_visualizer(visualizer)
        except (ImportError, Exception) as e:
            print(f"Open3D visualizer not available ({e}), using null visualizer")
            from components import NullVisualizer
            env = DefaultBodyEnv()
            visualizer = NullVisualizer(env)
            env.register_visualizer(visualizer)

        if bouncing:
            output = BouncingVideoDemo(env, video_name, fps)
        else:
            output = VideoSimulationOutput(env, video_name, fps)

        env.run_simulation(control_source, [output], steps)
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
        # Use BodyState properties for consistent state access
        self.env.state.r = np.array([4.9, 4.9, 9.8])  # Near boundaries (x, y), high altitude (z)
        self.env.state.v = np.array([0.5, 0.5, 0.2])  # Initial velocity (vx, vy, vz)

        super().initialize()
        print(f"Creating bouncing video: {self.video_name}")

    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        # Display events if any occurred (call parent method for event logging)
        super().process_step(step, control)

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