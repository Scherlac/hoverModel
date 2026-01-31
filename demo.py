import click
import numpy as np
from control_sources import ControlSourceFactory
from demo_runner import DemoRunner
from typing import List, Dict, Any
from components import Environment
from simulation_outputs import LoggingSimulationOutput, LiveVisualizationOutput, VideoSimulationOutput
from default_backend import (
    NewtonianPhysics, DefaultBodyEnv
)
from genesis_backend import GenesisBodyEnv

runner = DemoRunner()

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None: run_all_tests()

@cli.command()
@click.option('--control', 'controls', multiple=True, required=True)
@click.option('--output', 'outputs', multiple=True, required=True)
@click.option('--body', 'bodies', multiple=True, help='Add bodies to the environment (format: type:param1=value1,param2=value2)')
@click.option('--backend', default='default', type=click.Choice(['default', 'genesis']), help='Physics backend to use')
@click.option('--start-x', default=0.0)
@click.option('--start-y', default=0.0)
@click.option('--start-z', default=1.0)
@click.option('--steps', default=50, help='Number of simulation steps')
def run(controls, outputs, bodies, backend, start_x, start_y, start_z, steps):
    initial_pos = (start_x, start_y, start_z)
    control_configs = [parse_spec(spec, 'control') for spec in controls]
    output_configs = [parse_spec(spec, 'output') for spec in outputs]
    body_configs = [parse_spec(spec, 'body') for spec in bodies] if bodies else []
    
    click.echo(f"🚀 Running simulation with {backend} backend")
    click.echo(f"   Bodies: {len(body_configs) + 1} (default + {len(body_configs)} added)")
    click.echo(f"   Controls: {len(control_configs)}")
    click.echo(f"   Outputs: {len(output_configs)}")
    
    # Create environment based on backend choice
    if backend == 'genesis':
        env = GenesisBodyEnv()
    else:
        env = DefaultBodyEnv()
    
    # Add additional bodies if specified
    for body_config in body_configs:
        add_body_to_env(env, body_config)
    
    # Create outputs (they register themselves with the environment)
    output_instances = [create_output(oc['type'], oc['params'], env) for oc in output_configs]
    
    # Create controls and associate them with bodies
    control_sources = []
    for config in control_configs:
        # Extract body ID from config, default to first body if not specified
        body_id = config['params'].pop('body', env.bodies[0].id)
        control = create_control(config['type'], config['params'], body_id)
        control_sources.append(control)
    
    # Run simulation with new control system
    click.echo(f"Running simulation with {steps} steps...")
    env.run_simulation_with_controls(control_sources, steps=steps, initial_pos=initial_pos)

def parse_spec(spec: str, spec_type: str) -> Dict[str, Any]:
    parts = spec.split(':')
    config = {'type': parts[0], 'params': {}}
    if spec_type == 'control': config['steps'] = 50
    
    for part in parts[1:]:
        if '=' in part:
            k, v = part.split('=', 1)
            if spec_type == 'control' and k == 'steps':
                config['steps'] = int(v)
            else:
                # Try to parse as vector (comma-separated numbers)
                if ',' in v:
                    try:
                        config['params'][k] = [float(x.strip()) for x in v.split(',')]
                    except ValueError:
                        config['params'][k] = v
                else:
                    # Try to parse as single number, otherwise keep as string
                    try:
                        config['params'][k] = float(v)
                    except ValueError:
                        config['params'][k] = v
    return config

def create_control(control_type: str, params: Dict[str, Any], body_id: str):
    ct = control_type.lower()
    if ct in ('hovering', 'hover'): return ControlSourceFactory.create_hovering(body_id)
    elif ct == 'linear': return ControlSourceFactory.create_linear(body_id, params.get('force', 0.8))
    elif ct in ('rotational', 'rotate'): return ControlSourceFactory.create_rotational(body_id, params.get('torque', 0.3))
    elif ct in ('sinusoidal', 'sinusoid'): return ControlSourceFactory.create_sinusoidal(body_id)
    elif ct == 'chaotic': return ControlSourceFactory.create_chaotic(body_id)
    elif ct == 'force':
        force = params.get('force', [1.0, 0.0, 0.0])
        return ControlSourceFactory.create_force(body_id, np.array(force))
    elif ct == 'torque':
        torque = params.get('torque', [0.0, 0.0, 0.1])
        return ControlSourceFactory.create_torque(body_id, np.array(torque))
    elif ct == 'angular_momentum':
        angular_momentum = params.get('angular_momentum', [0.0, 0.0, 0.05])
        return ControlSourceFactory.create_angular_momentum(body_id, np.array(angular_momentum))
    elif ct == 'position':
        target_position = params.get('target', [2.0, 0.0, 1.0])
        return ControlSourceFactory.create_position(body_id, np.array(target_position))
    elif ct == 'velocity':
        target_velocity = params.get('target', [1.0, 0.0, 0.0])
        return ControlSourceFactory.create_velocity(body_id, np.array(target_velocity))
    elif ct == 'combined':
        controls = {}
        for key, value in params.items():
            controls[key] = value
        return ControlSourceFactory.create_combined(body_id, controls)
    raise click.ClickException(f"Unknown control: {control_type}")

def create_output(output_type: str, output_params: Dict[str, Any], env: Environment):
    ot = output_type.lower()
    if ot in ('console', 'logging'): return LoggingSimulationOutput(env)
    elif ot in ('live', 'visualization'): return LiveVisualizationOutput(env)
    elif ot == 'video': return VideoSimulationOutput(env, output_params.get('filename', 'demo.mp4'), int(output_params.get('fps', 25)))
    raise click.ClickException(f"Unknown output: {output_type}")

def add_body_to_env(env: Environment, body_config: Dict[str, Any]):
    """Add a body to the environment based on configuration."""
    from default_backend import DefaultBody
    
    body_type = body_config['type'].lower()
    params = body_config['params']
    
    if hasattr(env, 'add_body'):
        # Default backend supports adding bodies
        if body_type == 'sphere':
            # Default sphere parameters
            radius = params.get('radius', 0.5)
            mass = params.get('mass', 1.0)
            position = params.get('position', [0.0, 0.0, 1.0])
            if isinstance(position, str):
                position = [float(x) for x in position.split(',')]
            position = np.array(position)
            
            # Create a DefaultBody with sphere-like properties
            # For now, we'll use DefaultBody but set initial position
            body = DefaultBody(mass=mass)
            body.state.r = position
            
            # Add body to environment
            env.add_body(body)
    else:
        # Genesis backend doesn't support adding bodies dynamically
        click.echo(f"⚠️  Warning: {type(env).__name__} backend doesn't support adding bodies dynamically. Body '{body_config}' will be ignored.")

def run_all_tests():
    click.echo("🧪 Running tests...")
    env = DefaultBodyEnv()
    body_id = env.bodies[0].id
    for name, control in [("hover", ControlSourceFactory.create_hovering(body_id)), ("linear", ControlSourceFactory.create_linear(body_id, 0.8)), ("rotate", ControlSourceFactory.create_rotational(body_id, 0.3))]:
        click.echo(f"Testing {name}...")
        runner.run_test(control, 50)
    click.echo("✅ Done!")

if __name__ == "__main__": cli()