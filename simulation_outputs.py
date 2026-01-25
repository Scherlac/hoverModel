"""
Simulation output handlers for different output modes.
Follows SOLID principles with composition and low coupling.
"""

from abc import ABC, abstractmethod
import numpy as np
import time
import os
from typing import Optional, Tuple
from components import Environment, SimulationComponent, SimulationOutput
from visualization import NullVisualizer, Visualizer, Open3DVisualizer
from control_sources import ControlSource



class NullSimulationOutput(SimulationOutput):
    """Null output for testing - no visualization or logging."""

    def initialize(self) -> None:
        pass

    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        pass

    def finalize(self) -> None:
        pass


class LoggingSimulationOutput(SimulationOutput):
    """Logging output for physics testing with detailed state reports."""

    def __init__(self, env: Environment, log_interval: int = 10):
        super().__init__(env)
        self.log_interval = log_interval

    def initialize(self) -> None:
        print(f"🎯 Starting simulation with {self.env.physics.__class__.__name__}")
        print("Step | Position (x,y,z) | Velocity (vx,vy,vz) | Control (F,τ) | Theta | Omega")
        print("-" * 80)

    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        # Display events if any occurred
        if self.env.state.events:
            for event in self.env.state.events:
                print(f"      ⚡ EVENT: {event.label} at ({event.location[0]:.2f},{event.location[1]:.2f},{event.location[2]:.2f}) "
                      f"sources: {event.sources}")

        if step % self.log_interval == 0:
            pos = self.env.state.r
            vel = self.env.state.v
            theta = self.env.state.theta
            omega = self.env.state.omega
            print(f"{step:4d} | ({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}) | "
                  f"({vel[0]:6.2f},{vel[1]:6.2f},{vel[2]:6.2f}) | "
                  f"({control[0]:5.2f},{control[1]:5.2f}) | "
                  f"{theta:6.2f} | {omega:6.2f}")

    def finalize(self) -> None:
        print("-" * 80)
        print("✅ Simulation completed.")


class VideoSimulationOutput(SimulationOutput):
    """Video output with Open3D visualization and frame capture."""

    def __init__(self, env: Environment, video_name: str, fps: int = 25):
        super().__init__(env)
        self.video_name = video_name
        self.fps = fps
        self.frames_dir = "frames"
        self.frame_count = 0
        
        # Create and register visualizer
        bounds = env.bounds if hasattr(env, 'bounds') else {'x': (-5, 5), 'y': (-5, 5), 'z': (0, 10)}
        try:
            self.visualizer = Open3DVisualizer(bounds)
            env.register_visualizer(self.visualizer)
        except:
            self.visualizer = NullVisualizer(bounds)
            env.register_visualizer(self.visualizer)


    def initialize(self) -> None:
        # Camera setup
        if hasattr(self.visualizer, 'vis'):
            ctr = self.visualizer.vis.get_view_control()
            ctr.set_zoom(0.8)
            ctr.set_front([0.7, 0.3, 0.5])
            ctr.set_lookat([0, 0, 1])
            ctr.set_up([0, 0, 1])

        os.makedirs(self.frames_dir, exist_ok=True)
        print(f"Creating video: {self.video_name}")
        print(f"Frames will be saved to: {os.path.abspath(self.frames_dir)}/")
        print(f"Frames directory exists: {os.path.exists(self.frames_dir)}")

    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        # Display events if any occurred
        if self.env.state.events:
            for event in self.env.state.events:
                print(f"⚡ EVENT: {event.label} at ({event.location[0]:.2f},{event.location[1]:.2f},{event.location[2]:.2f}) "
                      f"sources: {event.sources}")

        if step % max(1, 25 // self.fps) == 0:  # Frame capture rate
            frame_path = f"{self.frames_dir}/frame_{self.frame_count:04d}.png"
            print(f"Capturing frame {self.frame_count} at step {step}: {frame_path}")
            try:
                self.visualizer.capture_frame(frame_path)
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

    def __init__(self, env: Environment):
        super().__init__(env)
        # Create and register visualizer
        bounds = env.bounds if hasattr(env, 'bounds') else {'x': (-5, 5), 'y': (-5, 5), 'z': (0, 10)}
        try:
            self.visualizer = Open3DVisualizer(bounds)
            env.register_visualizer(self.visualizer)
        except:
            self.visualizer = NullVisualizer(bounds)
            env.register_visualizer(self.visualizer)

    def initialize(self) -> None:
        print("🎮 Starting live visualization...")
        print("Press 'q' or close the window to exit")

    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        # Display events if any occurred
        if self.env.state.events:
            for event in self.env.state.events:
                print(f"⚡ EVENT: {event.label} at ({event.location[0]:.2f},{event.location[1]:.2f},{event.location[2]:.2f}) "
                      f"sources: {event.sources}")

        # Update visualization
        if self.visualizer:
            self.visualizer.update(self.env.state)
            self.visualizer.render()

        # Small delay for smooth visualization
        time.sleep(0.05)

        # Check if window is still open
        if self.visualizer and hasattr(self.visualizer, 'vis'):
            if not self.visualizer.vis.poll_events():
                raise KeyboardInterrupt("Visualization window closed")

    def finalize(self) -> None:
        print("✅ Live visualization completed")
        if self.visualizer:
            self.visualizer.close()