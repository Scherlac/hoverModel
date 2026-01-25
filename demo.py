import click
from control_sources import ControlSourceFactory
from demo_runner import DemoRunner
from typing import List, Dict, Any
from components import Environment
from simulation_outputs import LoggingSimulationOutput, LiveVisualizationOutput, VideoSimulationOutput
from environment import HovercraftEnv

runner = DemoRunner()

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None: run_all_tests()

@cli.command()
@click.option('--control', 'controls', multiple=True, required=True)
@click.option('--output', 'outputs', multiple=True, required=True)
@click.option('--start-x', default=0.0)
@click.option('--start-y', default=0.0)
@click.option('--start-z', default=1.0)
def run(controls, outputs, start_x, start_y, start_z):
    initial_pos = (start_x, start_y, start_z)
    control_configs = [parse_spec(spec, 'control') for spec in controls]
    output_configs = [parse_spec(spec, 'output') for spec in outputs]
    
    click.echo(f"🚀 Running {len(control_configs)} control(s), {len(output_configs)} output(s)")
    
    # Create environment first
    env = HovercraftEnv()
    
    # Create outputs (they register themselves with the environment)
    output_instances = [create_output(oc['type'], oc['params'], env) for oc in output_configs]
    
    body_id = env.bodies[0].id
    for config in control_configs:
        control = create_control(config['type'], config['params'], body_id)
        # Use environment's run_simulation method
        print(f"Running simulation with {config['steps']} steps")
        env.run_simulation(control, steps=config['steps'], initial_pos=initial_pos)

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
                config['params'][k] = float(v) if spec_type == 'control' and k != 'steps' else v
    return config

def create_control(control_type: str, params: Dict[str, Any], body_id: str):
    ct = control_type.lower()
    if ct in ('hovering', 'hover'): return ControlSourceFactory.create_hovering(body_id)
    elif ct == 'linear': return ControlSourceFactory.create_linear(body_id, params.get('force', 0.8))
    elif ct in ('rotational', 'rotate'): return ControlSourceFactory.create_rotational(body_id, params.get('torque', 0.3))
    elif ct in ('sinusoidal', 'sinusoid'): return ControlSourceFactory.create_sinusoidal(body_id)
    elif ct == 'chaotic': return ControlSourceFactory.create_chaotic(body_id)
    raise click.ClickException(f"Unknown control: {control_type}")

def create_output(output_type: str, output_params: Dict[str, Any], env: Environment):
    ot = output_type.lower()
    if ot in ('console', 'logging'): return LoggingSimulationOutput(env)
    elif ot in ('live', 'visualization'): return LiveVisualizationOutput(env)
    elif ot == 'video': return VideoSimulationOutput(env, output_params.get('filename', 'demo.mp4'), int(output_params.get('fps', 25)))
    raise click.ClickException(f"Unknown output: {output_type}")

def run_all_tests():
    click.echo("🧪 Running tests...")
    env = HovercraftEnv()
    body_id = env.bodies[0].id
    for name, control in [("hover", ControlSourceFactory.create_hovering(body_id)), ("linear", ControlSourceFactory.create_linear(body_id, 0.8)), ("rotate", ControlSourceFactory.create_rotational(body_id, 0.3))]:
        click.echo(f"Testing {name}...")
        runner.run_test(control, 50)
    click.echo("✅ Done!")

if __name__ == "__main__": cli()