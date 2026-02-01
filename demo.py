import click
import numpy as np
from control_sources import ControlSourceFactory
from demo_runner import DemoRunner
from typing import List, Dict, Any, Optional
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
    
    click.echo(f"Running simulation with {backend} backend")
    click.echo(f"   Bodies: {len(body_configs)} (default + {len(body_configs)} added)" if not bodies else f"   Bodies: {len(body_configs)}")
    click.echo(f"   Controls: {len(control_configs)}")
    click.echo(f"   Outputs: {len(output_configs)}")
    
    # Create environment based on backend choice
    if backend == 'genesis':
        env = GenesisBodyEnv(bodies=None if not bodies else [])
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
        body_param = config['params'].pop('body', None)
        if body_param is not None:
            try:
                body_index = int(body_param)
                if 0 <= body_index < len(env.bodies):
                    body_id = env.bodies[body_index].id
                else:
                    click.echo(f"Warning: Body index {body_index} out of range, using first body")
                    body_id = env.bodies[0].id if env.bodies else None
            except ValueError:
                # If it's not a number, assume it's a body ID
                body_id = body_param
        else:
            body_id = env.bodies[0].id if env.bodies else None
        control = create_control(config['type'], config['params'], body_id, config.get('steps'))
        control_sources.append(control)
    
    # Run simulation with new control system
    click.echo(f"Running simulation with {steps} steps...")
    env.run_simulation_with_controls(control_sources, steps=steps, initial_pos=initial_pos)

def parse_spec(spec: str, spec_type: str) -> Dict[str, Any]:
    parts = spec.split(':')
    config = {'type': parts[0], 'params': {}}
    if spec_type == 'control': config['steps'] = [None, None]  # Default: active entire simulation
    
    if len(parts) > 1:
        # For output specs, parameters are separated by ':', for control specs by ','
        separator = ':' if spec_type == 'output' else ','
        param_str = ':'.join(parts[1:])
        
        # Parse parameters, respecting brackets
        param_parts = []
        current = ''
        in_bracket = False
        for char in param_str:
            if char == '[':
                in_bracket = True
                current += char
            elif char == ']':
                in_bracket = False
                current += char
            elif char == separator and not in_bracket:
                param_parts.append(current.strip())
                current = ''
            else:
                current += char
        if current:
            param_parts.append(current.strip())
        
        for part in param_parts:
            if '=' in part:
                k, v = part.split('=', 1)
                k = k.strip()
                v = v.strip()
                
                if k == 'steps':
                    config['steps'] = parse_steps_range(v)
                else:
                # Check if this is a vector parameter (contains commas)
                    if ',' in v and k in ['force', 'torque', 'angular_momentum', 'target', 'position']:
                        try:
                            # Strip brackets if present
                            clean_v = v.strip('[]')
                            config['params'][k] = [float(x.strip()) for x in clean_v.split(',')]
                        except ValueError:
                            config['params'][k] = v
                    elif k == 'body':
                        # Keep body as string
                        config['params'][k] = v
                    else:
                        # Try to parse as single number, otherwise keep as string
                        try:
                            config['params'][k] = float(v)
                        except ValueError:
                            config['params'][k] = v
    
    return config


def parse_steps_range(steps_str: str) -> List[Optional[int]]:
    """Parse steps range string into [start, end] array format.
    
    Examples:
    - "10-50" -> [10, 50]
    - "25" -> [25, None] (from step 25 to end)
    - "-50" -> [None, 50] (from start to step 50)
    - "10-" -> [10, None] (from step 10 to end)
    """
    if '-' in steps_str:
        parts = steps_str.split('-')
        if len(parts) == 2:
            start_str, end_str = parts
            start = int(start_str) if start_str else None
            end = int(end_str) if end_str else None
            return [start, end]
    else:
        # Single number - treat as start only
        return [int(steps_str), None]

def create_control(control_type: str, params: Dict[str, Any], body_id: str, steps_range: Optional[List[Optional[int]]] = None):
    ct = control_type.lower()
    if ct in ('hovering', 'hover'): return ControlSourceFactory.create_hovering(body_id, steps_range)
    elif ct == 'linear': return ControlSourceFactory.create_linear(body_id, params.get('force', 0.8), steps_range)
    elif ct in ('rotational', 'rotate'): return ControlSourceFactory.create_rotational(body_id, params.get('torque', 0.3), steps_range)
    elif ct in ('sinusoidal', 'sinusoid'): return ControlSourceFactory.create_sinusoidal(body_id, steps_range)
    elif ct == 'chaotic': return ControlSourceFactory.create_chaotic(body_id, steps_range)
    elif ct == 'force':
        force = params.get('force', [1.0, 0.0, 0.0])
        return ControlSourceFactory.create_force(body_id, np.array(force), steps_range)
    elif ct == 'torque':
        torque = params.get('torque', [0.0, 0.0, 0.1])
        return ControlSourceFactory.create_torque(body_id, np.array(torque), steps_range)
    elif ct == 'angular_momentum':
        angular_momentum = params.get('angular_momentum', [0.0, 0.0, 0.05])
        return ControlSourceFactory.create_angular_momentum(body_id, np.array(angular_momentum), steps_range)
    elif ct == 'position':
        target_position = params.get('target', [2.0, 0.0, 1.0])
        return ControlSourceFactory.create_position(body_id, np.array(target_position), steps_range)
    elif ct == 'velocity':
        target_velocity = params.get('target', [1.0, 0.0, 0.0])
        return ControlSourceFactory.create_velocity(body_id, np.array(target_velocity), steps_range)
    elif ct == 'combined':
        controls = {}
        for key, value in params.items():
            controls[key] = value
        return ControlSourceFactory.create_combined(body_id, controls, steps_range)
    raise click.ClickException(f"Unknown control: {control_type}")

def create_output(output_type: str, output_params: Dict[str, Any], env: Environment):
    ot = output_type.lower()
    if ot in ('console', 'logging'): return LoggingSimulationOutput(env)
    elif ot in ('live', 'visualization'): return LiveVisualizationOutput(env)
    elif ot == 'video': return VideoSimulationOutput(env, output_params.get('filename', 'demo.mp4'), int(output_params.get('fps', 25)), output_params.get('render_mode', 'rgb'))
    raise click.ClickException(f"Unknown output: {output_type}")

def add_body_to_env(env: Environment, body_config: Dict[str, Any]):
    """Add a body to the environment based on configuration."""
    from default_backend import DefaultBody
    from genesis_backend import GenesisRigidBody
    
    body_type = body_config['type'].lower()
    params = body_config['params']
    
    if hasattr(env, 'add_body'):
        # Try to add body to the environment
        try:
            if body_type == 'sphere':
                # Default sphere parameters
                radius = params.get('radius', 0.5)
                mass = params.get('mass', 1.0)
                position = params.get('position', [0.0, 0.0, 1.0])
                if isinstance(position, str):
                    # Strip brackets and split
                    clean_pos = position.strip('[]')
                    position = [float(x.strip()) for x in clean_pos.split(',')]
                position = np.array(position)
                
                # For Genesis backend, create a GenesisRigidBody
                if hasattr(env, 'physics') and hasattr(env.physics, 'add_body'):
                    # Genesis backend
                    genesis_config = {
                        'type': 'sphere',
                        'mass': mass,
                        'radius': radius,
                        'initial_pos': position.tolist()
                    }
                    genesis_body = env.physics.add_body(genesis_config)
                    if genesis_body:
                        env.add_body(genesis_body)
                    else:
                        click.echo(f"⚠️  Warning: Could not add sphere body to Genesis environment")
                else:
                    # Default backend
                    body = DefaultBody(mass=mass)
                    body.state.r = position
                    env.add_body(body)
        except Exception as e:
            click.echo(f"⚠️  Warning: Could not add body to environment: {e}")
    else:
        # Environment doesn't support adding bodies
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