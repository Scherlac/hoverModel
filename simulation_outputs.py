"""
Simulation output handlers for different output modes.
Follows SOLID principles with composition and low coupling.
"""

from abc import ABC, abstractmethod
import numpy as np
import time
import os
from typing import Optional, Tuple
from components import (
    Environment, 
    SimulationComponent, 
    SimulationOutput, 
    NullVisualizer, 
    Visualizer
)
from default_backend import (
    Open3DVisualizer, 
    Open3DVisualizationOutput,
)
from control_sources import (
    ControlSource,
)



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
        print(f"Starting simulation with {self.env.physics.__class__.__name__}")
        print("Step | Position (x,y,z) | Velocity (vx,vy,vz) | Control (F,T) | Theta | Omega")
        print("-" * 80)

    def process_step(self, step: int, control) -> None:
        # Display events if any occurred
        if self.env.state.events:
            for event in self.env.state.events:
                print(f"      EVENT: {event.label} at ({event.location[0]:.2f},{event.location[1]:.2f},{event.location[2]:.2f}) "
                      f"sources: {event.sources}")

        if step % self.log_interval == 0:
            pos = self.env.state.r
            vel = self.env.state.v
            theta = self.env.state.theta
            omega = self.env.state.omega

            # Format control display based on type
            if isinstance(control, dict):
                control_str = f"dict({list(control.keys())})"
            elif isinstance(control, (list, tuple, np.ndarray)) and len(control) >= 2:
                control_str = f"({control[0]:5.2f},{control[1]:5.2f})"
            elif isinstance(control, (list, tuple, np.ndarray)) and len(control) == 1:
                control_str = f"({control[0]:5.2f})"
            else:
                control_str = f"{control}"

            print(f"{step:4d} | ({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}) | "
                  f"({vel[0]:6.2f},{vel[1]:6.2f},{vel[2]:6.2f}) | "
                  f"{control_str:>12} | "
                  f"{theta:6.2f} | {omega:6.2f}")

    def finalize(self) -> None:
        print("-" * 80)
        print("Simulation completed.")


class VideoSimulationOutput(SimulationOutput):
    """Video output with Open3D visualization and frame capture."""

    def __init__(self, env: Environment, video_name: str, fps: int = 25, render_mode: str = 'rgb'):
        super().__init__(env)
        self.video_name = video_name
        self.fps = fps
        self.frames_dir = "frames"
        self.frame_count = 0
        self.render_mode = render_mode
        
        # Choose appropriate visualizer based on environment type
        if hasattr(env, 'physics') and hasattr(env.physics, 'scene'):
            # Genesis environment - use Genesis visualizer
            from genesis_backend import GenesisVisualizer
            self.visualizer = env.get_specific_visualizer(GenesisVisualizer, render_mode=self.render_mode)
            self.is_genesis = True
        else:
            # Default environment - use Open3D visualizer
            self.visualizer = env.get_specific_visualizer(Open3DVisualizer)
            self.is_genesis = False
            
        self.visualization_output = self.visualizer.get_visualization_output()
        print(f"Live output using visualizer: {self.visualizer.__class__.__name__} and output: {self.visualization_output.__class__.__name__}")
        print(f"Render mode: {self.render_mode}")
        print(f"Using Genesis recording: {self.is_genesis}")



    def initialize(self) -> None:
        # Camera setup - match default backend camera position
        if self.visualization_output:
            self.visualization_output.set_camera(
                position=(5, 5, 5),  # Match default backend
                look_at=(0, 0, 1)    # Match default backend
            )
            self.visualization_output.set_zoom(0.6)  # Match default backend zoom

        if self.is_genesis:
            # For Genesis, start recording directly
            self.visualization_output.start_recording()
            print(f"Started Genesis video recording for: {self.video_name}")
        else:
            # For Open3D, set up frame capture
            os.makedirs(self.frames_dir, exist_ok=True)
            print(f"Creating video: {self.video_name}")
            print(f"Frames will be saved to: {os.path.abspath(self.frames_dir)}/")
            print(f"Frames directory exists: {os.path.exists(self.frames_dir)}")

    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        # Display events if any occurred
        if self.env.state.events:
            for event in self.env.state.events:
                print(f"EVENT: {event.label} at ({event.location[0]:.2f},{event.location[1]:.2f},{event.location[2]:.2f}) "
                      f"sources: {event.sources}")

        if self.is_genesis:
            # For Genesis, just render the frame (recording is automatic)
            if step % max(1, 25 // self.fps) == 0:  # Frame capture rate
                try:
                    # Call render to capture frame for recording
                    if hasattr(self.visualization_output, 'camera') and self.visualization_output.camera:
                        self.visualization_output.camera.render()
                        self.frame_count += 1
                        print(f"Genesis frame {self.frame_count} rendered at step {step}")
                except Exception as e:
                    print(f"Warning: Genesis frame render failed at step {step}: {e}")
        else:
            # For Open3D, capture frames to files
            if step % max(1, 25 // self.fps) == 0:  # Frame capture rate
                frame_path = f"{self.frames_dir}/frame_{self.frame_count:04d}.png"
                print(f"Capturing frame {self.frame_count} at step {step}: {frame_path}")
                try:
                    self.visualization_output.capture_frame(frame_path)
                    self.frame_count += 1
                    print(f"Frame {self.frame_count-1} captured successfully: {os.path.exists(frame_path)}")
                except Exception as e:
                    # If frame capture fails, skip this frame
                    print(f"Warning: Frame capture failed at step {step}: {e}")

    def finalize(self) -> None:
        if self.is_genesis:
            # For Genesis, stop recording and save video
            if self.frame_count > 0:
                try:
                    self.visualization_output.stop_recording(self.video_name, self.fps)
                    print(f"Genesis video created successfully: {self.video_name}")
                except Exception as e:
                    print(f"Genesis video creation failed: {e}")
            else:
                print("No frames rendered, cannot create Genesis video")
        else:
            # For Open3D, create video from captured frames using ffmpeg
            success = False
            if self.frame_count > 0:
                try:
                    # Use subprocess for better error handling
                    import subprocess
                    
                    # Try a simpler ffmpeg command first
                    cmd = [
                        'ffmpeg', '-y', 
                        '-framerate', str(self.fps), 
                        '-i', f'{self.frames_dir}/frame_%04d.png',
                        '-c:v', 'libx264', 
                        '-pix_fmt', 'yuv420p',
                        '-vf', 'scale=1280:720',  # Smaller resolution
                        self.video_name
                    ]
                    
                    print(f"Running FFmpeg command: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        print(f"Video created successfully: {self.video_name}")
                        success = True
                    else:
                        print(f"FFmpeg failed with exit code {result.returncode}")
                        print(f"FFmpeg stdout: {result.stdout}")
                        print(f"FFmpeg stderr: {result.stderr}")
                        
                        # Try alternative command without scaling
                        print("Trying alternative FFmpeg command without scaling...")
                        cmd_simple = [
                            'ffmpeg', '-y', 
                            '-framerate', str(self.fps), 
                            '-i', f'{self.frames_dir}/frame_%04d.png',
                            '-c:v', 'libx264', 
                            '-pix_fmt', 'yuv420p',
                            self.video_name
                        ]
                        
                        result_simple = subprocess.run(cmd_simple, capture_output=True, text=True, timeout=300)
                        if result_simple.returncode == 0:
                            print(f"Video created successfully with simple command: {self.video_name}")
                            success = True
                        else:
                            print(f"Simple FFmpeg command also failed with exit code {result_simple.returncode}")
                            print(f"FFmpeg stderr: {result_simple.stderr}")
                            
                except subprocess.TimeoutExpired:
                    print("FFmpeg timed out after 5 minutes")
                except FileNotFoundError:
                    print("FFmpeg not found. Please install FFmpeg and add it to your PATH")
                except Exception as e:
                    print(f"FFmpeg error: {e}")
            else:
                print("⚠️  No frames captured, cannot create video")

            # Cleanup frames directory
            import shutil
            if success and os.path.exists(self.frames_dir):
                try:
                    shutil.rmtree(self.frames_dir)
                    print(f"Frames directory '{self.frames_dir}' cleaned up")
                except Exception as e:
                    print(f"⚠️  Warning: Could not remove frames directory: {e}")
            elif not success:
                print(f"Frames saved in '{self.frames_dir}' directory for debugging (video creation failed)")


class LiveVisualizationOutput(SimulationOutput):
    """Live visualization output with interactive Open3D display."""

    def __init__(self, env: Environment):
        super().__init__(env)

        self.visualizer : Open3DVisualizer = env.get_specific_visualizer(Open3DVisualizer)
        self.visualization_output : Open3DVisualizationOutput = self.visualizer.get_visualization_output()
        print(f"Live output using visualizer: {self.visualizer.__class__.__name__} and output: {self.visualization_output.__class__.__name__}")

    def initialize(self) -> None:
        # Camera setup
        if self.visualization_output:
            self.visualization_output.set_camera(
                position=(10, 0, 5),
                look_at=(2.5, 0, 5)
            )
            self.visualization_output.set_zoom(0.8)

        print("Starting live visualization...")
        print("Press 'q' or close the window to exit")

    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        # Display events if any occurred
        if self.env.state.events:
            for event in self.env.state.events:
                print(f"EVENT: {event.label} at ({event.location[0]:.2f},{event.location[1]:.2f},{event.location[2]:.2f}) "
                      f"sources: {event.sources}")

        # Update visualization
        if self.visualization_output:
            self.visualization_output.render_frame()

        # Small delay for smooth visualization
        time.sleep(0.05)

        # Check if window is still open
        if self.visualizer and hasattr(self.visualizer, 'vis'):
            if not self.visualizer.vis.poll_events():
                raise KeyboardInterrupt("Visualization window closed")

    def finalize(self) -> None:
        print("Live visualization completed")
        if self.visualizer:
            self.visualizer.close()