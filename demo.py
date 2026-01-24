import click
from control_sources import ControlSourceFactory
from demo_runner import DemoRunner

# Create a global runner instance
runner = DemoRunner()


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Hovercraft demonstration system with modular control sources and outputs.

    Run physics tests and create videos of different hovercraft behaviors.
    """
    if ctx.invoked_subcommand is None:
        # Default behavior: run all tests
        run_all_tests()


@cli.command()
@click.option('--steps', default=50, help='Number of simulation steps')
def hover(steps):
    """Test hovering behavior (no control inputs)."""
    click.echo(f"🧭 Running HOVER demo ({steps} steps)")
    control = ControlSourceFactory.create_hovering()
    runner.run_test(control, steps=steps)


@cli.command()
@click.option('--force', default=0.8, help='Forward force magnitude')
@click.option('--steps', default=50, help='Number of simulation steps')
def linear(force, steps):
    """Test linear forward movement."""
    click.echo(f"➡️  Running LINEAR demo (force={force}, {steps} steps)")
    control = ControlSourceFactory.create_linear(forward_force=force)
    runner.run_test(control, steps=steps)


@cli.command()
@click.option('--torque', default=0.3, help='Rotation torque magnitude')
@click.option('--steps', default=50, help='Number of simulation steps')
def rotate(torque, steps):
    """Test rotational movement."""
    click.echo(f"🔄 Running ROTATE demo (torque={torque}, {steps} steps)")
    control = ControlSourceFactory.create_rotational(rotation_torque=torque)
    runner.run_test(control, steps=steps)


@cli.command()
@click.option('--steps', default=50, help='Number of simulation steps')
def sinusoid(steps):
    """Test combined sinusoidal movement."""
    click.echo(f"🌊 Running SINUSOID demo ({steps} steps)")
    control = ControlSourceFactory.create_sinusoidal()
    runner.run_test(control, steps=steps)


@cli.command()
@click.option('--steps', default=50, help='Number of simulation steps')
@click.option('--visualize', is_flag=True, help='Enable live 3D visualization')
def chaotic(steps, visualize):
    """Test chaotic boundary-bouncing behavior."""
    click.echo(f"🎯 Running CHAOTIC demo ({steps} steps){' with visualization' if visualize else ''}")
    control = ControlSourceFactory.create_chaotic()
    if visualize:
        runner.run_visualization(control, steps=steps)
    else:
        runner.run_test(control, steps=steps)


@cli.group()
def video():
    """Create demonstration videos."""
    pass


@video.command('hover')
@click.option('--output', default='hover_demo.mp4', help='Output video filename')
@click.option('--steps', default=200, help='Number of simulation steps')
@click.option('--fps', default=25, help='Video frame rate')
def video_hover(output, steps, fps):
    """Create hovering video."""
    click.echo(f"🎬 Creating HOVER video: {output} ({steps} steps, {fps} fps)")
    control = ControlSourceFactory.create_hovering()
    runner.create_video(control, output, steps=steps, fps=fps)


@video.command('linear')
@click.option('--force', default=0.8, help='Forward force magnitude')
@click.option('--output', default='linear_demo.mp4', help='Output video filename')
@click.option('--steps', default=200, help='Number of simulation steps')
@click.option('--fps', default=25, help='Video frame rate')
def video_linear(force, output, steps, fps):
    """Create linear movement video."""
    click.echo(f"🎬 Creating LINEAR video: {output} (force={force}, {steps} steps, {fps} fps)")
    control = ControlSourceFactory.create_linear(forward_force=force)
    runner.create_video(control, output, steps=steps, fps=fps)


@video.command('rotate')
@click.option('--torque', default=0.3, help='Rotation torque magnitude')
@click.option('--output', default='rotate_demo.mp4', help='Output video filename')
@click.option('--steps', default=200, help='Number of simulation steps')
@click.option('--fps', default=25, help='Video frame rate')
def video_rotate(torque, output, steps, fps):
    """Create rotational movement video."""
    click.echo(f"🎬 Creating ROTATE video: {output} (torque={torque}, {steps} steps, {fps} fps)")
    control = ControlSourceFactory.create_rotational(rotation_torque=torque)
    runner.create_video(control, output, steps=steps, fps=fps)


@video.command('sinusoid')
@click.option('--output', default='sinusoid_demo.mp4', help='Output video filename')
@click.option('--steps', default=200, help='Number of simulation steps')
@click.option('--fps', default=25, help='Video frame rate')
def video_sinusoid(output, steps, fps):
    """Create sinusoidal movement video."""
    click.echo(f"🎬 Creating SINUSOID video: {output} ({steps} steps, {fps} fps)")
    control = ControlSourceFactory.create_sinusoidal()
    runner.create_video(control, output, steps=steps, fps=fps)


@video.command('chaotic')
@click.option('--output', default='chaotic_demo.mp4', help='Output video filename')
@click.option('--steps', default=300, help='Number of simulation steps')
def video_chaotic(output, steps):
    """Create chaotic boundary-bouncing video."""
    click.echo(f"🎬 Creating CHAOTIC video: {output} ({steps} steps)")
    control = ControlSourceFactory.create_chaotic()
    runner.create_video(control, output, steps=steps, bouncing=True)


def run_all_tests():
    """Run all physics tests."""
    click.echo("🧪 Running all physics tests...\n")

    # Test hovering
    click.echo("Testing hovering...")
    control = ControlSourceFactory.create_hovering()
    runner.run_test(control, steps=50)

    # Test movement
    click.echo("\nTesting linear movement...")
    control = ControlSourceFactory.create_linear(0.8)
    runner.run_test(control, steps=50)

    # Test rotation
    click.echo("\nTesting rotation...")
    control = ControlSourceFactory.create_rotational(0.3)
    runner.run_test(control, steps=50)

    click.echo("\n✅ All tests completed!")


if __name__ == "__main__":
    cli()