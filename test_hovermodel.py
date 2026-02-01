import pytest
import numpy as np
import subprocess
import sys
import os
import shutil
from default_backend import DefaultBodyEnv
from control_sources import ControlSourceFactory
from simulation_outputs import LoggingSimulationOutput


def check_ffmpeg_available():
    """Check if FFmpeg is available on the system."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


def cleanup_frames_dir():
    """Clean up any remaining frames directory."""
    frames_dir = "frames"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)


def test_linear_control_causes_movement():
    """Test that linear control with positive force moves the body forward."""
    env = DefaultBodyEnv()
    body_id = env.bodies[0].id
    print(f"body_id: {body_id}")
    control = ControlSourceFactory.create_linear(body_id, 1.0)
    
    initial_pos = env.bodies[0].state.r.copy()
    env.run_simulation_with_controls([control], steps=20)
    final_pos = env.bodies[0].state.r
    
    # Body should have moved in positive x direction
    assert final_pos[0] > initial_pos[0]
    assert abs(final_pos[0] - initial_pos[0]) > 0.01  # Significant movement


def test_rotational_control_changes_orientation():
    """Test that rotational control changes the body's orientation."""
    env = DefaultBodyEnv()
    body_id = env.bodies[0].id
    control = ControlSourceFactory.create_rotational(body_id, 1.0)  # Stronger torque
    
    initial_theta = env.bodies[0].state.theta
    env.run_simulation_with_controls([control], steps=20)
    final_theta = env.bodies[0].state.theta
    
    # Orientation should have changed
    assert abs(final_theta - initial_theta) > 0.01


def test_hovering_control_maintains_stability():
    """Test that hovering control keeps the body relatively stable."""
    env = DefaultBodyEnv()
    body_id = env.bodies[0].id
    control = ControlSourceFactory.create_hovering(body_id)
    
    initial_pos = env.bodies[0].state.r.copy()
    env.run_simulation_with_controls([control], steps=20)
    final_pos = env.bodies[0].state.r
    
    # Position should not change significantly (hovering)
    assert np.allclose(final_pos, initial_pos, atol=0.1)


def test_simulation_runs_without_errors():
    """Test that simulation completes without raising exceptions."""
    env = DefaultBodyEnv()
    body_id = env.bodies[0].id
    control = ControlSourceFactory.create_linear(body_id, 0.5)
    
    # Should not raise any exceptions
    env.run_simulation_with_controls([control], steps=10)


def test_console_output_generation():
    """Test that console output handler works without errors."""
    env = DefaultBodyEnv()
    body_id = env.bodies[0].id
    control = ControlSourceFactory.create_linear(body_id, 1.0)
    output = LoggingSimulationOutput(env)
    
    env.outputs = [output]
    env.run_simulation_with_controls([control], steps=5)
    
    # Should complete without errors
    assert True


def test_boundary_collision():
    """Test that body bounces off boundaries."""
    env = DefaultBodyEnv()
    body_id = env.bodies[0].id
    # Start near boundary
    env.bodies[0].state.r = np.array([4.5, 0.0, 1.0])
    control = ControlSourceFactory.create_linear(body_id, 5.0)  # Strong force toward boundary
    
    env.run_simulation_with_controls([control], steps=50)
    
    # Should not go beyond bounds
    final_pos = env.bodies[0].state.r
    assert -5 <= final_pos[0] <= 5
    assert -5 <= final_pos[1] <= 5
    assert 0 <= final_pos[2] <= 10


def test_sinusoidal_control():
    """Test that sinusoidal control produces oscillating movement."""
    env = DefaultBodyEnv()
    body_id = env.bodies[0].id
    control = ControlSourceFactory.create_sinusoidal(body_id)
    
    env.run_simulation_with_controls([control], steps=50)
    
    # Should have some movement
    final_pos = env.bodies[0].state.r
    assert abs(final_pos[0]) > 0.01 or abs(final_pos[1]) > 0.01


def test_chaotic_control():
    """Test that chaotic control produces unpredictable movement."""
    env = DefaultBodyEnv()
    body_id = env.bodies[0].id
    control = ControlSourceFactory.create_chaotic(body_id)
    
    env.run_simulation_with_controls([control], steps=30)
    
    # Should have moved from initial position
    final_pos = env.bodies[0].state.r
    assert not np.allclose(final_pos, [0.0, 0.0, 1.0], atol=0.01)


# CLI Documentation Verification Tests

def test_cli_linear_console_example():
    """Test the CLI example: python demo.py run --control linear:force=1.0 --output console --steps 50"""
    result = subprocess.run([
        sys.executable, "demo.py", "run", 
        "--control", "linear:force=1.0", 
        "--output", "console",
        "--steps", "50"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0
    assert "Running simulation with 50 steps..." in result.stdout
    assert "Step | Position" in result.stdout
    assert "Simulation completed." in result.stdout


def test_cli_rotational_console_example():
    """Test the CLI example: python demo.py run --control rotational:torque=0.5 --output console --steps 100"""
    result = subprocess.run([
        sys.executable, "demo.py", "run", 
        "--control", "rotational:torque=0.5", 
        "--output", "console",
        "--steps", "100"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0
    assert "Running simulation with 100 steps..." in result.stdout


def test_cli_hovering_backend_example():
    """Test the CLI example: python demo.py run --control hovering --output console --backend default"""
    result = subprocess.run([
        sys.executable, "demo.py", "run", 
        "--control", "hovering", 
        "--output", "console",
        "--backend", "default"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0
    assert "default backend" in result.stdout


def test_cli_sinusoidal_console_example():
    """Test the CLI example: python demo.py run --control sinusoidal --output console --steps 100"""
    result = subprocess.run([
        sys.executable, "demo.py", "run", 
        "--control", "sinusoidal", 
        "--output", "console",
        "--steps", "100"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0
    assert "Running simulation with 100 steps..." in result.stdout


def test_cli_chaotic_console_example():
    """Test the CLI example: python demo.py run --control chaotic --output console --steps 150"""
    result = subprocess.run([
        sys.executable, "demo.py", "run", 
        "--control", "chaotic", 
        "--output", "console",
        "--steps", "150"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0
    assert "Running simulation with 150 steps..." in result.stdout


def test_cli_help_command():
    """Test the CLI help command: python demo.py --help"""
    result = subprocess.run([
        sys.executable, "demo.py", "--help"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0
    assert "run" in result.stdout
    assert "Options:" in result.stdout


def test_cli_run_help_command():
    """Test the CLI run help command: python demo.py run --help"""
    result = subprocess.run([
        sys.executable, "demo.py", "run", "--help"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0
    assert "--control" in result.stdout
    assert "--output" in result.stdout
    assert "--backend" in result.stdout


# Video Output Documentation Verification Tests

def test_cli_linear_video_example():
    """Test the CLI example: python demo.py run --control linear:force=5.0 --output video:filename=linear_demo.mp4:fps=10 --steps 100"""
    # Skip if FFmpeg is not available
    if not check_ffmpeg_available():
        pytest.skip("FFmpeg not available - skipping video test")
    
    video_file = "linear_demo.mp4"
    
    # Clean up any existing files
    if os.path.exists(video_file):
        os.remove(video_file)
    cleanup_frames_dir()
    
    result = subprocess.run([
        sys.executable, "demo.py", "run", 
        "--control", "linear:force=5.0", 
        "--output", f"video:filename={video_file}:fps=10",
        "--steps", "100"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0
    assert "Running simulation with 100 steps..." in result.stdout
    assert "Creating video:" in result.stdout
    # Video must be created since FFmpeg is available
    assert os.path.exists(video_file), "Video file should be created"
    assert os.path.getsize(video_file) > 0, "Video file should have content"
    
    # Clean up
    if os.path.exists(video_file):
        os.remove(video_file)
    cleanup_frames_dir()


def test_cli_bouncing_video_example():
    """Test the CLI example: python demo.py run --control linear:force=20.0 --output video:filename=bouncing_demo.mp4:fps=10 --start-x=0.0 --start-y=0.0 --start-z=5.0 --steps 100"""
    # Skip if FFmpeg is not available
    if not check_ffmpeg_available():
        pytest.skip("FFmpeg not available - skipping video test")
    
    video_file = "bouncing_demo.mp4"
    
    # Clean up any existing files
    if os.path.exists(video_file):
        os.remove(video_file)
    cleanup_frames_dir()
    
    result = subprocess.run([
        sys.executable, "demo.py", "run", 
        "--control", "linear:force=20.0", 
        "--output", f"video:filename={video_file}:fps=10",
        "--start-x", "0.0",
        "--start-y", "0.0", 
        "--start-z", "5.0",
        "--steps", "100"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0
    assert "Running simulation with 100 steps..." in result.stdout
    assert "Creating video:" in result.stdout
    # Video must be created since FFmpeg is available
    assert os.path.exists(video_file), "Video file should be created"
    assert os.path.getsize(video_file) > 0, "Video file should have content"
    
    # Clean up
    if os.path.exists(video_file):
        os.remove(video_file)
    cleanup_frames_dir()


def test_cli_rotational_video_example():
    """Test the CLI example: python demo.py run --control rotational:torque=1.0 --output video:filename=rotation.mp4:fps=15 --steps 150"""
    # Skip if FFmpeg is not available
    if not check_ffmpeg_available():
        pytest.skip("FFmpeg not available - skipping video test")
    
    video_file = "rotation.mp4"
    
    # Clean up any existing files
    if os.path.exists(video_file):
        os.remove(video_file)
    cleanup_frames_dir()
    
    result = subprocess.run([
        sys.executable, "demo.py", "run", 
        "--control", "rotational:torque=1.0", 
        "--output", f"video:filename={video_file}:fps=15",
        "--steps", "150"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0
    assert "Running simulation with 150 steps..." in result.stdout
    assert "Creating video:" in result.stdout
    # Video must be created since FFmpeg is available
    assert os.path.exists(video_file), "Video file should be created"
    assert os.path.getsize(video_file) > 0, "Video file should have content"
    
    # Clean up
    if os.path.exists(video_file):
        os.remove(video_file)
    cleanup_frames_dir()


def test_cli_custom_start_video_example():
    """Test the CLI example: python demo.py run --control linear:force=10.0 --output video:filename=custom_start.mp4:fps=10 --start-x=2.0 --start-y=1.0 --start-z=3.0 --steps 100"""
    # Skip if FFmpeg is not available
    if not check_ffmpeg_available():
        pytest.skip("FFmpeg not available - skipping video test")
    
    video_file = "custom_start.mp4"
    
    # Clean up any existing files
    if os.path.exists(video_file):
        os.remove(video_file)
    cleanup_frames_dir()
    
    result = subprocess.run([
        sys.executable, "demo.py", "run", 
        "--control", "linear:force=10.0", 
        "--output", f"video:filename={video_file}:fps=10",
        "--start-x", "2.0",
        "--start-y", "1.0",
        "--start-z", "3.0",
        "--steps", "100"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0
    assert "Running simulation with 100 steps..." in result.stdout
    assert "Creating video:" in result.stdout
    # Video must be created since FFmpeg is available
    assert os.path.exists(video_file), "Video file should be created"
    assert os.path.getsize(video_file) > 0, "Video file should have content"
    
    # Clean up
    if os.path.exists(video_file):
        os.remove(video_file)
    cleanup_frames_dir()


def test_cli_chaotic_video_example():
    """Test the CLI example: python demo.py run --control chaotic --output video:filename=chaos.mp4:fps=20 --steps 200"""
    # Skip if FFmpeg is not available
    if not check_ffmpeg_available():
        pytest.skip("FFmpeg not available - skipping video test")
    
    video_file = "chaos.mp4"
    
    # Clean up any existing files
    if os.path.exists(video_file):
        os.remove(video_file)
    cleanup_frames_dir()
    
    result = subprocess.run([
        sys.executable, "demo.py", "run", 
        "--control", "chaotic", 
        "--output", f"video:filename={video_file}:fps=20",
        "--steps", "200"
    ], capture_output=True, text=True, cwd=".")
    
    assert result.returncode == 0
    assert "Running simulation with 200 steps..." in result.stdout
    assert "Creating video:" in result.stdout
    # Video must be created since FFmpeg is available
    assert os.path.exists(video_file), "Video file should be created"
    assert os.path.getsize(video_file) > 0, "Video file should have content"
    
    # Clean up
    if os.path.exists(video_file):
        os.remove(video_file)
    cleanup_frames_dir()


# Live Visualization Documentation Verification Tests

def test_cli_linear_live_example():
    """Test the CLI example: python demo.py run --control linear:force=3.0 --output live --steps 100"""
    # Skip live visualization tests as they require interactive display windows
    pytest.skip("Live visualization requires interactive display environment")


def test_cli_bouncing_live_example():
    """Test the CLI example: python demo.py run --control linear:force=15.0 --output live --start-x=0.0 --start-y=0.0 --start-z=5.0 --steps 200"""
    # Skip live visualization tests as they require interactive display windows
    pytest.skip("Live visualization requires interactive display environment")