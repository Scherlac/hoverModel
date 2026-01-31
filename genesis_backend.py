"""
Genesis backend implementations for the hovercraft simulation.

This module contains concrete implementations using the Genesis physics engine,
providing an alternative to the default Open3D + NumPy backend.
"""

import numpy as np
import genesis as gs
from typing import Dict, Any, Optional, Tuple, List
import pathlib
import time

from components import PhysicsEngine, Body, Environment, Visualizer, VisualizationOutput
from state import BodyState, PhysicsEvent

ASSET_ROOT = pathlib.Path(__file__).parent / "assets"


class GenesisPhysics(PhysicsEngine):
    """Genesis-based physics engine for body simulation.

    Uses Genesis for realistic physics simulation with collision detection,
    rigid body dynamics, and complex mesh support.
    """

    def __init__(self, config: dict):
        """Initialize Genesis physics engine."""
        # Initialize Genesis
        gs.init(backend=gs.cpu)

        # Store configuration
        self.config = config
        self.dt = config.get('dt', 0.01)

        # Create scene
        self.scene = gs.Scene(show_viewer=False)  # We'll handle visualization separately

        # Add ground plane
        self.ground = self.scene.add_entity(gs.morphs.Plane())

        # Scene will be built when bodies are added
        self.built = False

        # Store bounds
        bounds_config = config.get('bounds', [[-5, 5], [-5, 5], [0, 10]])
        self.bounds = {
            'x': tuple(bounds_config[0]),
            'y': tuple(bounds_config[1]),
            'z': tuple(bounds_config[2])
        }

    def add_body(self, body_config: dict) -> 'GenesisRigidBody':
        """Add a body to the Genesis scene."""
        if self.built:
            raise RuntimeError("Cannot add bodies after scene is built")

        # Create Genesis entity based on body type
        body_type = body_config.get('type', 'hovercraft')

        if body_type == 'hovercraft':
            # Load hovercraft mesh
            mesh_file = body_config.get('mesh_file', ASSET_ROOT / "hoverBody_main.obj")
            entity = self.scene.add_entity(
                gs.morphs.Mesh(
                    file=str(mesh_file),
                    scale=body_config.get('scale', 0.1),
                    pos=body_config.get('initial_pos', (0.0, 0.0, 1.0)),
                )
            )
        elif body_type == 'sphere':
            entity = self.scene.add_entity(
                gs.morphs.Sphere(
                    radius=body_config.get('radius', 0.1),
                    pos=body_config.get('initial_pos', (0.0, 0.0, 1.0)),
                )
            )
        else:
            raise ValueError(f"Unsupported body type: {body_type}")

        # Create wrapper body
        genesis_body = GenesisRigidBody(entity, body_config)
        return genesis_body

    def build_scene(self):
        """Build the Genesis scene."""
        if not self.built:
            self.scene.build()
            self.built = True

    def step(self, body: 'GenesisRigidBody', action: np.ndarray, dt: float) -> BodyState:
        """
        Apply Genesis physics to a body.

        Args:
            body: GenesisRigidBody object
            action: Control inputs [forward_force, rotation_torque]
            dt: Time step duration

        Returns:
            next_state: Updated BodyState object
        """
        if not self.built:
            self.build_scene()

        # Set control forces on the body
        body.set_control(action)

        # Step the simulation
        self.scene.step()

        # Get updated state
        return body.get_state()

    def get_bounds(self) -> dict:
        """Return environment bounds."""
        return self.bounds

    def close(self):
        """Clean up Genesis resources."""
        if hasattr(self, 'scene'):
            # Genesis cleanup
            pass


class GenesisRigidBody(Body):
    """Genesis-based rigid body implementation."""

    def __init__(self, entity, config: dict):
        """Initialize Genesis rigid body."""
        # Initialize with Body's required parameters
        mass = config.get('mass', 1.0)
        moment_of_inertia = config.get('moment_of_inertia', 0.1)
        shape = config.get('shape', {'type': 'mesh', 'mesh_path': 'hoverBody_main.obj'})
        
        super(GenesisRigidBody, self).__init__(mass, moment_of_inertia, shape)
        
        self.entity = entity
        self.config = config

        # Override the state with our own
        self.state = BodyState()
        self.state.r = np.array(config.get('initial_pos', (0.0, 0.0, 1.0)))
        self.state.v = np.zeros(3)
        self.state.theta = config.get('initial_theta', 0.0)
        self.state.omega = 0.0

        # Control state
        self.control_force = np.zeros(3)
        self.control_torque = np.zeros(3)

    def get_forces(self, action: np.ndarray, environment_state: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """
        Calculate forces and torques acting on the body.

        Args:
            action: Control inputs [forward_force, rotation_torque]
            environment_state: Environment conditions

        Returns:
            Tuple of (force_vector, torque_scalar)
        """
        # Control forces
        forward_force = action[0]
        rotation_torque = action[1]

        # Convert to 3D force vector
        force = np.array([forward_force, 0.0, 0.0])
        
        # Add gravity
        gravity = np.array(environment_state.get('gravity', [0.0, 0.0, -9.81]))
        force += self.mass * gravity

        return force, rotation_torque

    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get movement bounds for this body type."""
        # Return default bounds - could be made configurable
        return {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0),
            'z': (0.0, 10.0)
        }

    def set_control(self, action: np.ndarray):
        """Set control forces and torques."""
        # action is [forward_force, rotation_torque]
        forward_force = action[0]
        rotation_torque = action[1]

        # Convert to 3D forces
        # Assuming forward is along x-axis in body frame
        self.control_force = np.array([forward_force, 0.0, 0.0])
        self.control_torque = np.array([0.0, 0.0, rotation_torque])

        # Apply forces to Genesis entity
        # Genesis uses DOF forces: [fx, fy, fz, tx, ty, tz]
        try:
            force_vector = np.concatenate([self.control_force, self.control_torque])
            self.entity.set_dofs_force(force_vector)
        except Exception as e:
            print(f"Warning: Could not set forces on Genesis entity: {e}")
            # Try alternative method
            try:
                self.entity.control_dofs_force(force_vector)
            except Exception as e2:
                print(f"Warning: Alternative force method also failed: {e2}")

    def get_state(self) -> BodyState:
        """Get current body state from Genesis entity."""
        try:
            # Get position from Genesis
            pos = self.entity.get_pos()
            self.state.r = np.array(pos)

            # Get orientation (quaternion to angle)
            quat = self.entity.get_quat()
            # Convert quaternion to rotation around z-axis (theta)
            # For simplicity, extract theta from quaternion
            # This is an approximation for 2D rotation
            self.state.theta = 2 * np.arctan2(quat[2], quat[3])  # Simplified quaternion to angle

            # Get velocities from Genesis DOFs
            dofs_velocity = self.entity.get_dofs_velocity()
            if len(dofs_velocity) >= 6:
                self.state.v = np.array(dofs_velocity[:3])  # Linear velocity
                self.state.omega = dofs_velocity[5]  # Angular velocity around z

        except Exception as e:
            print(f"Warning: Could not get state from Genesis entity: {e}")

        return self.state


class GenesisBodyEnv(Environment):
    """
    Environment using Genesis physics engine.

    Provides Genesis-based physics simulation with collision detection
    and complex rigid body dynamics.
    """

    def __init__(self,
                 config: dict = None,
                 bodies: list[Body] = None):
        """
        Initialize Genesis environment.

        Args:
            config: Environment configuration
            bodies: List of bodies (should be GenesisRigidBody instances)
        """
        super(GenesisBodyEnv, self).__init__()
        self.config = config or self._default_config()
        self.dt = self.config.get('dt', 0.01)

        # Set bounds from config
        bounds_list = self.config.get('bounds', [[-5, 5], [-5, 5], [0, 10]])
        self.bounds = {
            'x': tuple(bounds_list[0]),
            'y': tuple(bounds_list[1]),
            'z': tuple(bounds_list[2])
        }

        # Initialize Genesis physics
        self.physics = GenesisPhysics(self.config)

        # Initialize bodies
        if bodies is None:
            # Default: single hovercraft
            hovercraft_config = {
                'type': 'hovercraft',
                'mass': self.config.get('mass', 1.0),
                'moment_of_inertia': self.config.get('momentum', 0.1),
                'initial_pos': (0.0, 0.0, 1.0),
                'scale': 0.1,
                'mesh_file': ASSET_ROOT / "hoverBody_main.obj"
            }
            genesis_body = self.physics.add_body(hovercraft_config)
            self.bodies = [genesis_body]
        else:
            self.bodies = bodies

        # Build the scene
        self.physics.build_scene()

        # For backward compatibility
        self.state = self.bodies[0].state if self.bodies else None
        
        # Step counter for outputs
        self.step_count = 0

    def _default_config(self) -> dict:
        """Default environment configuration."""
        return {
            'mass': 1.0,
            'momentum': 0.1,
            'gravity': [0.0, 0.0, -9.81],
            'dt': 0.01,
            'bounds': [[-5, 5], [-5, 5], [0, 10]]
        }

    def step(self, action: np.ndarray) -> BodyState:
        """
        Advance environment by one time step.

        Args:
            action: Control inputs [forward_force, rotation_torque]

        Returns:
            Updated body state
        """
        # Step physics for the first body (single body support for now)
        state = self.physics.step(self.bodies[0], action, self.dt)

        # Update visualizers
        for viz in self.visualizers:
            viz.update(state)

        # Process outputs
        for output in self.outputs:
            output.process_step(self.step_count, (action[0], action[1]))
            self.step_count += 1

        return state

    def run_simulation(self, control_source, outputs=None, steps=50, initial_pos=None):
        """Run simulation with control source and multiple outputs."""
        if outputs is not None:
            self.outputs = outputs

        # Set initial position if provided
        if initial_pos:
            self.bodies[0].state.r = np.array(initial_pos)
            self.bodies[0].state.v = np.zeros(3)
            self.bodies[0].state.theta = 0.0
            self.bodies[0].state.omega = 0.0

        # Initialize all outputs
        for output in self.outputs:
            output.initialize()

        # Run simulation steps
        for step in range(steps):
            from control_sources import SignalChannel
            channel = SignalChannel()
            channel = control_source.get_control(channel, step)
            control_input = channel[control_source.lain_index]
            if isinstance(control_input, (tuple, list)):
                control_input = np.array(control_input)
            elif isinstance(control_input, (int, float)):
                control_input = np.array([0.0, control_input])  # Assume torque
            state = self.step(control_input)
            print(f"Step {step}: Position {state.r}, Velocity {state.v}")

        # Finalize outputs
        for output in self.outputs:
            output.finalize()

    def get_bodies(self) -> list[Body]:
        """Get all bodies in the environment."""
        return self.bodies.copy()

    def get_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        return self.config


class GenesisVisualizer(Visualizer):
    """Genesis-based visualizer using built-in viewer."""

    def __init__(self, env: Environment):
        super(GenesisVisualizer, self).__init__(env)
        self.env = env
        # Genesis handles its own visualization
        self.viewer_active = False

    def update(self, state: BodyState):
        """Update visualization - Genesis handles this internally."""
        pass

    def get_visualization_output(self) -> VisualizationOutput:
        """Get visualization output handler."""
        return GenesisVisualizationOutput(self)

    def close(self):
        """Clean up visualization resources."""
        pass


class GenesisVisualizationOutput(VisualizationOutput):
    """Genesis visualization output handler."""

    def __init__(self, visualizer: GenesisVisualizer):
        super(GenesisVisualizationOutput, self).__init__(visualizer)

    def set_camera(self, position: Tuple[float, float, float], look_at: Tuple[float, float, float]) -> None:
        """Set camera position - not directly supported in Genesis viewer."""
        pass

    def render_frame(self) -> None:
        """Render frame - handled by Genesis internally."""
        pass

    def capture_frame(self, filename: str) -> None:
        """Capture frame to file - not directly supported."""
        # Could potentially use Genesis screenshot functionality
        pass