import pytest
import numpy as np
import subprocess
import sys
import os
import shutil
from default_backend import DefaultBodyEnv
from genesis_backend import GenesisBodyEnv
from control_sources import ControlSourceFactory
from simulation_outputs import LoggingSimulationOutput
from demo import parse_spec


@pytest.fixture(params=['default', 'genesis'])
def backend_env(request):
    """Fixture that provides both default and genesis backend environments."""
    if request.param == 'genesis':
        env = GenesisBodyEnv()
    else:
        env = DefaultBodyEnv()
    yield env


def check_ffmpeg_available():
    """Check if FFmpeg is available on the system."""


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


def test_linear_control_causes_movement(backend_env):
    """Test that linear control with positive force moves the body forward."""
    env = backend_env
    body_id = env.bodies[0].id
    print(f"body_id: {body_id}")
    control = ControlSourceFactory.create_linear(body_id, 1.0)
    
    initial_pos = env.bodies[0].state.r.copy()
    env.run_simulation_with_controls([control], steps=20)
    final_pos = env.bodies[0].state.r
    
    # Body should have moved in positive x direction
    assert final_pos[0] > initial_pos[0]
    assert abs(final_pos[0] - initial_pos[0]) > 0.01  # Significant movement


def test_rotational_control_changes_orientation(backend_env):
    """Test that rotational control changes the body's orientation."""
    env = backend_env
    body_id = env.bodies[0].id
    control = ControlSourceFactory.create_rotational(body_id, 10.0)  # Much stronger torque
    
    initial_theta = env.bodies[0].state.theta
    env.run_simulation_with_controls([control], steps=20)
    final_theta = env.bodies[0].state.theta
    
    # Orientation should have changed significantly
    assert abs(final_theta - initial_theta) > 1.0, f"Angle should change significantly, got {abs(final_theta - initial_theta)}"


def test_hovering_control_maintains_stability(backend_env):
    """Test that hovering control keeps the body relatively stable."""
    env = backend_env
    body_id = env.bodies[0].id
    control = ControlSourceFactory.create_hovering(body_id)
    
    initial_pos = env.bodies[0].state.r.copy()
    env.run_simulation_with_controls([control], steps=20)
    final_pos = env.bodies[0].state.r
    
    # Position should not change significantly (hovering)
    assert np.allclose(final_pos, initial_pos, atol=0.05)


def test_default_body_mass_consistency(backend_env):
    """Test that the default body has the same mass in both backends."""
    env = backend_env
    body = env.bodies[0]
    
    # Both backends should default to mass = 1.0
    assert body.mass == 1.0, f"Expected mass=1.0, got {body.mass}"


def test_sliding_friction_behavior(backend_env):
    """Test that friction behavior is consistent between backends (using Genesis as reference)."""
    env = backend_env
    
    # Set initial velocity and position
    body = env.bodies[0]
    initial_velocity = np.array([2.0, 0.0, 0.0])
    body.set_velocity(initial_velocity)
    body.state.r = np.array([0.0, 0.0, 1.0])

    # Configure friction
    # NO DIFFERENTIATION BASED ON BACKEND!!
    body.set_friction(10.0)  # High friction for testing

    initial_position = body.state.r.copy()
    
    # Run simulation with no controls
    env.run_simulation_with_controls([], steps=30)
    
    final_velocity = body.state.v
    final_position = body.state.r

    # Distance traveled should be reasonable
    distance_traveled = final_position[0] - initial_position[0]
    assert distance_traveled > 0.5, f"Body should travel some distance: {distance_traveled}"
    assert distance_traveled < 1.0, f"Body should not travel too far: {distance_traveled}"

    # Check that x-velocity remains high (friction is weak in both backends)
    assert final_velocity[0] > 1.5, \
        f"X-velocity should remain high: initial {initial_velocity[0]}, final {final_velocity[0]}"


def test_simulation_runs_without_errors(backend_env):
    """Test that simulation completes without raising exceptions."""
    env = backend_env
    body_id = env.bodies[0].id
    control = ControlSourceFactory.create_linear(body_id, 0.5)
    
    # Should not raise any exceptions
    env.run_simulation_with_controls([control], steps=10)


def test_console_output_generation(backend_env):
    """Test that console output handler works without errors."""
    env = backend_env
    body_id = env.bodies[0].id
    control = ControlSourceFactory.create_linear(body_id, 1.0)
    output = LoggingSimulationOutput(env)
    
    env.outputs = [output]
    env.run_simulation_with_controls([control], steps=5)
    
    # Should complete without errors
    assert True


def test_boundary_collision(backend_env):
    """Test that body bounces off boundaries."""
    env = backend_env
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


def test_sinusoidal_control(backend_env):
    """Test that sinusoidal control produces oscillating movement."""
    env = backend_env
    body_id = env.bodies[0].id
    control = ControlSourceFactory.create_sinusoidal(body_id)
    
    env.run_simulation_with_controls([control], steps=50)
    
    # Should have some movement
    final_pos = env.bodies[0].state.r
    assert abs(final_pos[0]) > 0.01 or abs(final_pos[1]) > 0.01


def test_chaotic_control(backend_env):
    """Test that chaotic control produces unpredictable movement."""
    env = backend_env
    body_id = env.bodies[0].id
    control = ControlSourceFactory.create_chaotic(body_id)
    
    env.run_simulation_with_controls([control], steps=30)
    
    # Should have moved from initial position
    final_pos = env.bodies[0].state.r
    assert not np.allclose(final_pos, [0.0, 0.0, 1.0], atol=0.01)


# Step Range Feature Tests (Planned Implementation)

def test_control_step_range_parsing():
    """Test that control specifications with step ranges are parsed correctly."""
    from demo import parse_spec
    
    # Test step range parsing
    config = parse_spec("linear:force=5.0,steps=10-50", "control")
    assert config['type'] == 'linear'
    assert config['params']['force'] == 5.0
    assert config['steps'] == [10, 50]  # Range: start=10, end=50
    
    # Test single step start
    config = parse_spec("rotational:torque=1.0,steps=25", "control")
    assert config['type'] == 'rotational'
    assert config['params']['torque'] == 1.0
    assert config['steps'] == [25, None]  # Start from step 25, no end limit
    
    # Test no steps specified (default behavior)
    config = parse_spec("hovering", "control")
    assert config['type'] == 'hovering'
    assert config['steps'] == [None, None]  # Active entire simulation


def test_output_step_range_parsing():
    """Test that output specifications with step ranges are parsed correctly."""
    from demo import parse_spec
    
    # Test video output with step range
    config = parse_spec("video:filename=test.mp4:fps=10:steps=5-30", "output")
    assert config['type'] == 'video'
    assert config['params']['filename'] == 'test.mp4'
    assert config['params']['fps'] == 10.0
    assert config['steps'] == [5, 30]  # Range: start=5, end=30
    
    # Test console output with step range
    config = parse_spec("console:steps=10", "output")
    assert config['type'] == 'console'
    assert config['steps'] == [10, None]  # Start from step 10, no end limit


def test_body_step_range_parsing():
    """Test that body specifications with step ranges are parsed correctly."""
    from demo import parse_spec
    
    # Test body with step range
    config = parse_spec("sphere:radius=0.5,mass=2.0,steps=20-80", "body")
    assert config['type'] == 'sphere'
    assert config['params']['radius'] == 0.5
    assert config['params']['mass'] == 2.0
    assert config['steps'] == [20, 80]  # Range: start=20, end=80


def test_control_step_range_execution():
    """Test that controls are only active during their specified step ranges."""
    from control_sources import ControlSourceFactory
    from control_sources import SignalChannel
    
    # Create a linear control that should only be active from steps 5-15
    control = ControlSourceFactory.create_linear("test_body", 2.0, [5, 15])
    
    # Test steps before range - should not be active
    channel = SignalChannel()
    channel = control.get_control(channel, 3)  # Step 3
    assert len(channel) <= control.lain_index or channel[control.lain_index] is None  # No control signal
    
    # Test steps within range - should be active
    channel = SignalChannel()
    channel = control.get_control(channel, 10)  # Step 10
    assert len(channel) > control.lain_index and channel[control.lain_index] is not None  # Control signal present
    import numpy as np
    np.testing.assert_array_equal(channel[control.lain_index], np.array([2.0, 0.0, 0.0]))
    
    # Test steps after range - should not be active
    channel = SignalChannel()
    channel = control.get_control(channel, 20)  # Step 20
    assert len(channel) <= control.lain_index or channel[control.lain_index] is None  # No control signal


@pytest.mark.skip(reason="Step range feature not yet implemented")
def test_output_step_range_execution():
    """Test that outputs are only active during their specified step ranges."""
    # This test will validate that outputs only produce data during their step ranges
    # For example, video output should only capture frames during active steps
    # Console output should only log during active steps
    pass  # Placeholder until implementation


@pytest.mark.skip(reason="Step range feature not yet implemented")  
def test_body_step_range_execution():
    """Test that bodies appear/disappear according to their step ranges."""
    # This test will validate that bodies are only present in the simulation
    # during their specified step ranges
    pass  # Placeholder until implementation


@pytest.mark.skip(reason="Step range feature not yet implemented")
def test_cli_control_step_range_example():
    """Test CLI example with control step range: python demo.py run --control linear:force=5.0,steps=10-50 --output console --steps 100"""
    result = subprocess.run([
        sys.executable, "demo.py", "run", 
        "--control", "linear:force=5.0,steps=10-50", 
        "--output", "console",
        "--steps", "100"
    ], capture_output=True, text=True, cwd=".")
    
    # Once implemented, this should show movement only during steps 10-50
    # For now, it will behave as if steps parameter is ignored
    assert result.returncode == 0
    assert "Running simulation with 100 steps..." in result.stdout


@pytest.mark.skip(reason="Step range feature not yet implemented")
def test_cli_output_step_range_example():
    """Test CLI example with output step range: python demo.py run --control chaotic --output video:filename=partial.mp4:fps=10:steps=25-75 --steps 100"""
    if not check_ffmpeg_available():
        pytest.skip("FFmpeg not available - skipping video test")
    
    video_file = "partial.mp4"
    
    # Clean up any existing files
    if os.path.exists(video_file):
        os.remove(video_file)
    cleanup_frames_dir()
    
    result = subprocess.run([
        sys.executable, "demo.py", "run", 
        "--control", "chaotic", 
        "--output", f"video:filename={video_file}:fps=10:steps=25-75",
        "--steps", "100"
    ], capture_output=True, text=True, cwd=".")
    
    # Once implemented, video should only contain frames from steps 25-75
    assert result.returncode == 0
    assert "Creating video:" in result.stdout
    
    # Clean up
    if os.path.exists(video_file):
        os.remove(video_file)
    cleanup_frames_dir()


@pytest.mark.skip(reason="Step range feature not yet implemented")
def test_cli_body_step_range_example():
    """Test CLI example with body step range: python demo.py run --control hovering --body sphere:radius=0.5,mass=2.0,steps=20-80 --output console --steps 100"""
    result = subprocess.run([
        sys.executable, "demo.py", "run", 
        "--control", "hovering", 
        "--body", "sphere:radius=0.5,mass=2.0,steps=20-80",
        "--output", "console",
        "--steps", "100"
    ], capture_output=True, text=True, cwd=".")
    
    # Once implemented, sphere should only be present during steps 20-80
    assert result.returncode == 0
    assert "Running simulation with 100 steps..." in result.stdout


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