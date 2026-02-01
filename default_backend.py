"""
Default backend implementations for the hovercraft simulation.

This module contains concrete implementations of the abstract interfaces
defined in components.py, providing the default Open3D + NumPy physics backend.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import pathlib

from components import PhysicsEngine, Body, Environment, Visualizer, VisualizationOutput
from state import BodyState, PhysicsEvent
from control_sources import SignalChannel

ASSET_ROOT = pathlib.Path(__file__).parent / "assets"


class NewtonianPhysics(PhysicsEngine):
    """General Newtonian physics engine for body simulation.

    Works with any Body object that implements get_forces() method.
    Handles multi-body interactions and environmental effects.
    """

    def __init__(self, config: dict):
        """Initialize Newtonian physics engine."""
        # Global environment properties
        gravity_config = config.get('gravity', [0.0, 0.0, -9.81])
        self.gravity_vector = np.array(gravity_config)

        # Bounds configuration - simplified array format: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        bounds_config = config.get('bounds', [[-5, 5], [-5, 5], [0, 10]])
        bounds_array = np.array(bounds_config)
        self.bounds_min = bounds_array[:, 0]
        self.bounds_max = bounds_array[:, 1]

        # Store bounds in dict format for compatibility
        self.bounds = {
            'x': (self.bounds_min[0], self.bounds_max[0]),
            'y': (self.bounds_min[1], self.bounds_max[1]),
            'z': (self.bounds_min[2], self.bounds_max[2])
        }

    def step(self, body: Body, action: np.ndarray, dt: float) -> BodyState:
        """
        Apply Newtonian physics to a body.

        Args:
            body: Body object with physical properties and current state
            action: Control inputs [forward_force, rotation_torque]
            dt: Time step duration

        Returns:
            next_state: Updated BodyState object
        """
        # Get current state
        current_state = body.get_state()

        # Environment state for force calculations
        environment_state = {
            'gravity': self.gravity_vector,
            'bounds': self.bounds,
            'dt': dt
        }

        # Get forces and torques from the body
        total_force, total_torque = body.get_forces(action, environment_state)

        # Physics integration (vectorized)
        acceleration = total_force / body.mass          # F = ma
        alpha = total_torque / body.moment_of_inertia   # T = Iα

        # State integration
        v_new = current_state.v + acceleration * dt
        omega_new = current_state.omega + alpha * dt

        r_new = current_state.r + v_new * dt
        theta_new = current_state.theta + omega_new * dt

        # Boundary collision handling (vectorized bounce)
        # Detect collisions before clipping
        collision_events = []
        for i, axis in enumerate(['x', 'y', 'z']):
            if r_new[i] <= self.bounds_min[i] or r_new[i] >= self.bounds_max[i]:
                # Create collision event
                event_location = r_new.copy()
                event_location[i] = self.bounds_min[i] if r_new[i] <= self.bounds_min[i] else self.bounds_max[i]
                collision_events.append(PhysicsEvent(
                    location=event_location,
                    radius=0.5,  # Event influence radius
                    label=f"boundary_collision_{axis}",
                    sources=["bounds", f"{axis}_boundary", "body"]
                ))

        r_new = np.clip(r_new, self.bounds_min, self.bounds_max)
        # Bounce velocity when hitting bounds
        v_new = np.where(r_new == self.bounds_min, -v_new, v_new)
        v_new = np.where(r_new == self.bounds_max, -v_new, v_new)

        # Create new state with events
        new_state = BodyState(r=r_new, v=v_new, theta=theta_new, omega=omega_new, events=collision_events)
        return new_state

    def step_multiple(self, bodies: list[Body], actions: list[np.ndarray], dt: float) -> list[BodyState]:
        """
        Step multiple bodies with potential interactions.

        Args:
            bodies: List of Body objects
            actions: List of actions corresponding to each body
            dt: Time step duration

        Returns:
            List of updated BodyState objects
        """
        new_states = []
        for body, action in zip(bodies, actions):
            new_state = self.step(body, action, dt)
            new_states.append(new_state)
        return new_states

    def get_bounds(self) -> dict:
        """Return environment bounds."""
        return self.bounds


class DefaultBody(Body):
    """DefaultBody-specific body with lifting force and control characteristics."""

    def __init__(self,
                 mass: float = 1.0,
                 moment_of_inertia: float = 0.1,
                 lift_force_mean: float = 10.0,
                 lift_force_std: float = 1.0,
                 rotational_noise_mean: float = 0.1,
                 rotational_noise_std: float = 0.5,
                 friction_coefficient: float = 0.1,
                 shape: Optional[Dict[str, Any]] = None):
        """
        Initialize hovercraft body.

        Args:
            mass: DefaultBody mass in kg
            moment_of_inertia: Rotational inertia
            lift_force_mean: Mean lifting force
            lift_force_std: Standard deviation of lifting force
            rotational_noise_mean: Mean rotational noise
            rotational_noise_std: Standard deviation of rotational noise
            friction_coefficient: Ground friction coefficient
            shape: Shape description (mesh, etc.)
        """
        super(DefaultBody, self).__init__(mass, moment_of_inertia, shape)

        # DefaultBody-specific properties
        self.lift_force_mean = lift_force_mean
        self.lift_force_std = lift_force_std
        self.rotational_noise_mean = rotational_noise_mean
        self.rotational_noise_std = rotational_noise_std
        self.friction_coefficient = friction_coefficient

        # Initialize applied controls
        self.applied_force = np.zeros(3)
        self.applied_torque = 0.0

    def get_forces(self, action: np.ndarray, environment_state: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """
        Calculate hovercraft-specific forces and torques.

        Args:
            action: [forward_force, rotation_torque] (legacy support)
            environment_state: Environment conditions

        Returns:
            Tuple of (total_force_vector, total_torque_scalar)
        """
        # Check if we have applied controls, otherwise use legacy action
        if hasattr(self, 'applied_force') and self.applied_force is not None:
            if isinstance(self.applied_force, np.ndarray) and self.applied_force.ndim > 0:
                forward_force = self.applied_force[0]  # x-component
                applied_force_z = self.applied_force[2] if len(self.applied_force) > 2 else 0.0
            else:
                # applied_force is a scalar, use it as forward force
                forward_force = float(self.applied_force)
                applied_force_z = 0.0
        else:
            forward_force = action[0]

        if hasattr(self, 'applied_torque') and self.applied_torque is not None:
            if isinstance(self.applied_torque, np.ndarray) and self.applied_torque.ndim > 0:
                rotation_torque = self.applied_torque[2] if len(self.applied_torque) > 2 else self.applied_torque[0]
            else:
                rotation_torque = float(self.applied_torque)
        else:
            rotation_torque = action[1]

        # Environment forces
        gravity = environment_state.get('gravity', np.array([0.0, 0.0, -9.81]))

        # DefaultBody-specific forces
        lift_force = np.random.normal(self.lift_force_mean, self.lift_force_std)
        lift_vector = np.array([0.0, 0.0, lift_force])

        # Add any applied z-force
        if 'applied_force_z' in locals():
            lift_vector[2] += applied_force_z

        # Forward force in direction of orientation
        forward_vector = forward_force * np.array([
            np.cos(self.state.theta),
            np.sin(self.state.theta),
            0.0
        ])

        # Friction force (proportional to height and velocity)
        friction_force = -self.friction_coefficient * self.state.r[2] * self.state.v

        # Rotational noise
        rotational_noise = np.random.normal(self.rotational_noise_mean, self.rotational_noise_std)

        # Total forces and torques
        total_force = gravity * self.mass + lift_vector + forward_vector + friction_force
        total_torque = rotation_torque + rotational_noise

        return total_force, total_torque

    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get hovercraft movement bounds."""
        return {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0),
            'z': (0.0, 10.0)
        }

    def apply_force(self, force: np.ndarray) -> None:
        """Apply a force vector to the hovercraft."""
        # Store the applied force for use in get_forces
        self.applied_force = force

    def apply_torque(self, torque: np.ndarray) -> None:
        """Apply a torque vector to the hovercraft."""
        # For 2D hovercraft, we mainly care about z-torque
        self.applied_torque = torque[2] if len(torque) > 2 else torque[0]

    def apply_angular_momentum(self, angular_momentum: np.ndarray) -> None:
        """Apply angular momentum to the hovercraft."""
        # Angular momentum affects rotational velocity
        # L = I * ω, so ω = L / I
        self.state.omega += angular_momentum[2] / self.moment_of_inertia

    def set_position_target(self, target_position: np.ndarray) -> None:
        """Set position control target."""
        self.position_target = target_position

    def set_velocity_target(self, target_velocity: np.ndarray) -> None:
        """Set velocity control target."""
        self.velocity_target = target_velocity

    def copy(self) -> 'DefaultBody':
        """Create a copy of this hovercraft."""
        hovercraft = DefaultBody(
            mass=self.mass,
            moment_of_inertia=self.moment_of_inertia,
            lift_force_mean=self.lift_force_mean,
            lift_force_std=self.lift_force_std,
            rotational_noise_mean=self.rotational_noise_mean,
            rotational_noise_std=self.rotational_noise_std,
            friction_coefficient=self.friction_coefficient,
            shape=self.shape
        )
        hovercraft.state = self.state.copy()
        return hovercraft

    def __repr__(self) -> str:
        return f"DefaultBody(mass={self.mass}, state={self.state})"

ASSET_ROOT = pathlib.Path(__file__).parent / "assets"

class Open3DVisualizationOutput(VisualizationOutput):
    """Visualization output handler for Open3D visualizer."""

    def __init__(self, visualizer: "Open3DVisualizer"):
        super(Open3DVisualizationOutput, self).__init__(visualizer)
        # camera params
        self.camera_position = np.array([5, 5, 5])
        self.camera_look_at = np.array([0, 0, 1])
        self.up_vector = np.array([0, 0, 1])
        self.zoom = 0.6

        self._set_camera()


    def _set_camera(self) -> None:
        ctr = self.visualizer.vis.get_view_control()
        ctr.set_lookat(self.camera_look_at)
        ctr.set_front(self.camera_position)
        ctr.set_up(self.up_vector)  # Z-up
        ctr.set_zoom(self.zoom)

    def set_camera(self, position: Tuple[float, float, float], look_at: Tuple[float, float, float]) -> None:
        self.camera_position = np.array(position)
        self.camera_look_at = np.array(look_at)
        self._set_camera()

    def set_zoom(self, zoom: float) -> None:
        self.zoom = zoom
        self._set_camera()

    def render_frame(self) -> None:
        self.visualizer.render()

    def capture_frame(self, filename: str) -> None:
        self.visualizer.capture_frame(filename)


class Open3DVisualizer(Visualizer):
    """Open3D-based 3D visualization."""

    def __init__(self, env):
        super(Open3DVisualizer, self).__init__(env)
        try:
            import open3d as o3d
            self.o3d = o3d
        except ImportError:
            raise ImportError("Open3D required for 3D visualization")

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="DefaultBody Simulation", width=1024, height=768, visible=True)

        # Setup environment geometry
        self._setup_environment(self.env.bounds)

        # DefaultBody geometry
        body_shape = self.env.bodies[0].shape
        mesh_path = body_shape.get('mesh_path') if isinstance(body_shape, dict) else None
        if mesh_path:
            abs_mesh_path = ASSET_ROOT / mesh_path
            print(f"Loading hovercraft mesh from {str(abs_mesh_path)}")
            self.hovercraft = self.o3d.io.read_triangle_mesh(str(abs_mesh_path))
            print(f"Mesh loaded: vertices={len(self.hovercraft.vertices)}, triangles={len(self.hovercraft.triangles)}")
            if not self.hovercraft.has_vertices():
                print(f"Warning: Failed to load mesh from {str(abs_mesh_path)}, falling back to cylinder")
                self.hovercraft = self.o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=0.5)
        else:
            print(f"Falling back to default hovercraft geometry (cylinder)")
            self.hovercraft = self.o3d.geometry.TriangleMesh.create_cylinder(radius=1.0, height=0.5)
        self.hovercraft.paint_uniform_color([0, 0.5, 1])
        self.vis.add_geometry(self.hovercraft)
        self.hovercraft_original_vertices = np.asarray(self.hovercraft.vertices).copy()
        self.hovercraft_center = self.hovercraft.get_center()

        # Set default camera view
        self._setup_camera()

    def _setup_environment(self, bounds):
        """Setup static environment geometry."""
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds.values()

        # Boundary fence
        points = [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_min]  # close
        ]
        lines = [[i, i+1] for i in range(len(points)-1)]
        colors = [[1, 0, 0] for _ in lines]
        line_set = self.o3d.geometry.LineSet()
        line_set.points = self.o3d.utility.Vector3dVector(points)
        line_set.lines = self.o3d.utility.Vector2iVector(lines)
        line_set.colors = self.o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(line_set)

        # Ground plane
        ground = self.o3d.geometry.TriangleMesh.create_box(
            width=x_max-x_min, height=y_max-y_min, depth=0.1
        )
        ground.translate([x_min, y_min, z_min-0.1])
        ground.paint_uniform_color([0.5, 0.5, 0.5])
        self.vis.add_geometry(ground)

    def _setup_camera(self):
        """Setup default camera view."""
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0.5, 0.5, -0.7])  # Angled view
        ctr.set_lookat([0, 0, 1])        # Look at center
        ctr.set_up([0, 0, 1])            # Up direction

    def update(self, state: BodyState):
        """Update hovercraft position and orientation."""
        with open('debug.log', 'a') as f:
            f.write(f"Visualizer update called with position {state.r}\n")

        # Reset and transform hovercraft
        self.hovercraft.vertices = self.o3d.utility.Vector3dVector(self.hovercraft_original_vertices)
        R = self.o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, 0.1*state.theta])
        # R = self.o3d.geometry.get_rotation_matrix_from_axis_angle([0.01, 0, 0.01])
        self.hovercraft.rotate(R, center=self.hovercraft_center)
        self.hovercraft.translate(state.r, relative=False)
        self.hovercraft.compute_vertex_normals()


        self.vis.update_geometry(self.hovercraft)
        self.vis.poll_events()
        self.vis.update_renderer()

    def render(self) -> None:
        """Render the current frame."""
        self.vis.poll_events()
        self.vis.update_renderer()

    def capture_frame(self, filename: str) -> None:
        self.render()
        self.vis.capture_screen_image(filename)

    def get_visualization_output(self) -> VisualizationOutput:
        return Open3DVisualizationOutput(self)


    def close(self):
        """Clean up Open3D resources."""
        self.vis.destroy_window()


class DefaultBodyEnv(Environment):
    """
    Environment containing physical bodies with proper separation of concerns.

    The environment:
    - Contains Body objects (not just state)
    - Manages body interactions and environmental conditions
    - Delegates physics calculations to PhysicsEngine
    """

    def __init__(self,
                 physics_engine: PhysicsEngine = None,
                 config: dict = None,
                 bodies: list[Body] = None):
        """
        Initialize environment with bodies and dependencies.

        Args:
            physics_engine: Physics simulation component
            config: Environment configuration
            bodies: List of bodies in the environment (default: single hovercraft)
        """
        super(DefaultBodyEnv, self).__init__()
        self.config = config or self._default_config()
        self.dt = self.config.get('dt', 0.01)

        # Set bounds from config
        bounds_list = self.config.get('bounds', [[-5, 5], [-5, 5], [0, 10]])
        self.bounds = {
            'x': tuple(bounds_list[0]),
            'y': tuple(bounds_list[1]),
            'z': tuple(bounds_list[2])
        }

        # Dependency injection with defaults
        self.physics = physics_engine or NewtonianPhysics(self.config)

        # Initialize bodies
        if bodies is None:
            # Default: single hovercraft
            hovercraft_config = {
                'mass': self.config.get('mass', 1.0),
                'moment_of_inertia': self.config.get('momentum', 0.1),
                'lift_force_mean': self.config.get('lift_mean', 10.0),
                'lift_force_std': self.config.get('lift_std', 1.0),
                'rotational_noise_mean': self.config.get('rot_mean', 0.1),
                'rotational_noise_std': self.config.get('rot_std', 0.5),
                'friction_coefficient': self.config.get('friction_k', 0.1)
            }
            self.bodies = [DefaultBody(**hovercraft_config)]
        else:
            self.bodies = bodies

        self.state = self.bodies[0].state

    def _default_config(self) -> dict:
        """Default environment configuration."""
        return {
            'mass': 1.0,
            'momentum': 0.1,
            'gravity': [0.0, 0.0, -9.81],
            'dt': 0.01,
            'lift_mean': 10.0,
            'lift_std': 1.0,
            'rot_mean': 0.1,
            'rot_std': 0.5,
            'friction_k': 0.1,
            'bounds': [[-5, 5], [-5, 5], [0, 10]]
        }

    def step(self, action: np.ndarray) -> BodyState:
        """
        Advance environment by one time step.

        For single-body environments (backward compatibility), returns the body state.
        For multi-body environments, use step_multiple().

        Args:
            action: [forward_force, rotation_torque] for the primary body

        Returns:
            next_state: Updated BodyState of the primary body
        """
        if len(self.bodies) == 1:
            # Single body case (backward compatibility)
            new_state = self.physics.step(self.bodies[0], action, self.dt)
            self.bodies[0].set_state(new_state)
            return new_state
        else:
            # Multi-body case - delegate to step_multiple
            return self.step_multiple([action])[0]

    def step_multiple(self, actions: list[np.ndarray]) -> list[BodyState]:
        """
        Advance multiple bodies by one time step.

        Args:
            actions: List of actions for each body

        Returns:
            List of updated BodyState objects for each body
        """
        if len(actions) != len(self.bodies):
            raise ValueError(f"Number of actions ({len(actions)}) must match number of bodies ({len(self.bodies)})")

        # Step all bodies
        new_states = self.physics.step_multiple(self.bodies, actions, self.dt)

        # Update body states
        for body, new_state in zip(self.bodies, new_states):
            body.set_state(new_state)

        return new_states


    def reset(self) -> BodyState:
        """Reset environment to initial state."""
        for body in self.bodies:
            body.reset()
        return self.bodies[0].get_state() if self.bodies else BodyState()

    # Convenience properties for primary body access (backward compatibility)
    @property
    def state(self) -> BodyState:
        """Get primary body state (for backward compatibility)."""
        return self.bodies[0].get_state() if self.bodies else BodyState()

    @state.setter
    def state(self, new_state: BodyState) -> None:
        """Set primary body state (for backward compatibility)."""
        if self.bodies:
            self.bodies[0].set_state(new_state)

    @property
    def body(self) -> Body:
        """Get primary body."""
        return self.bodies[0] if self.bodies else None

    # Convenience properties for state access - delegate to primary body
    @property
    def position(self) -> np.ndarray:
        """Get current position [x, y, z]."""
        return self.state.r

    @property
    def orientation(self) -> float:
        """Get current orientation theta."""
        return self.state.theta

    @property
    def velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy, vz]."""
        return self.state.v

    @property
    def angular_velocity(self) -> float:
        """Get current angular velocity omega_z."""
        return self.state.omega

    def save_state(self, filepath: str) -> None:
        """Save current state to file."""
        self.state.save(filepath)

    def load_state(self, filepath: str) -> BodyState:
        """Load state from file."""
        loaded_state = BodyState.load(filepath)
        if self.bodies:
            self.bodies[0].set_state(loaded_state)
        return loaded_state

    def get_state_dict(self) -> dict:
        """Get current state as dictionary."""
        return self.state.to_dict()

    def set_state_dict(self, state_dict: dict) -> None:
        """Set state from dictionary."""
        new_state = BodyState.from_dict(state_dict)
        if self.bodies:
            self.bodies[0].set_state(new_state)

    def add_body(self, body: Body) -> None:
        """Add a body to the environment."""
        self.bodies.append(body)

    def remove_body(self, body: Body) -> None:
        """Remove a body from the environment."""
        if body in self.bodies:
            self.bodies.remove(body)

    def get_body_by_id(self, body_id: str) -> Optional[Body]:
        """Get a body by its ID."""
        for body in self.bodies:
            if body.id == body_id:
                return body
        return None

    def run_simulation_with_controls(self, control_sources: List["ControlSource"], outputs=None, steps=50, initial_pos=None):
        """
        Run simulation with multiple control sources using the new control system.

        Args:
            control_sources: List of ControlSource objects
            outputs: List of output handlers
            steps: Number of simulation steps
            initial_pos: Initial position for bodies
        """
        if outputs is not None:
            self.outputs = outputs

        # Set initial position if provided
        if initial_pos:
            for body in self.bodies:
                body.state.r = np.array(initial_pos)
                body.state.v = np.zeros(3)
                body.state.theta = 0.0
                body.state.omega = 0.0
                body.state.clear_events()

        # Initialize all outputs
        for output in self.outputs:
            output.initialize()

        # Run simulation steps
        for step in range(steps):
            # Create control channel and get controls from all sources
            channel = SignalChannel()
            for control_source in control_sources:
                channel = control_source.get_control(channel, step)

            # Apply controls to bodies
            self.apply_controls_from_channel(channel, self.dt)

            # Step physics for all bodies (using applied controls)
            for body in self.bodies:
                # Use zero action since controls are applied directly to bodies
                body.state = self.physics.step(body, np.zeros(2), self.dt)

            # Update visualizers
            for visualizer in self.visualizers:
                if self.bodies:
                    visualizer.update(self.bodies[0].state)

            # Process outputs
            for output in self.outputs:
                # For backward compatibility, pass the first body's control
                first_control = channel[control_sources[0].lain_index] if control_sources else np.zeros(2)
                if isinstance(first_control, (tuple, list)):
                    first_control = np.array(first_control)
                elif isinstance(first_control, (int, float)):
                    first_control = np.array([0.0, first_control])
                output.process_step(step, first_control)

        # Finalize outputs
        for output in self.outputs:
            output.finalize()

    def run_simulation(self, control_source, outputs=None, steps=50, initial_pos=None):
        """Run simulation with control source and multiple outputs."""
        if outputs is not None:
            self.outputs = outputs
        # Set initial position if provided
        if initial_pos:
            self.state.r = np.array(initial_pos)
            self.state.v = np.zeros(3)
            self.state.theta = self.state.omega = 0.0
            self.state.clear_events()

        # Initialize all outputs
        for output in self.outputs:
            output.initialize()

        # Run simulation steps
        for step in range(steps):
            channel = SignalChannel()
            channel = control_source.get_control(channel, step)
            control_input = channel[control_source.lain_index]
            if isinstance(control_input, (tuple, list)):
                control_input = np.array(control_input)
            elif isinstance(control_input, (int, float)):
                control_input = np.array([0.0, control_input])  # Assume torque or something
            self.step(control_input)

            # process step for all visualizers/outputs
            for visualizer in self.visualizers:
                visualizer.update(self.state)

            for output in self.outputs:
                output.process_step(step, control_input)

        # Finalize all outputs
        for output in self.outputs:
            output.finalize()
        """Run simulation with control source and multiple outputs."""
        if outputs is not None:
            self.outputs = outputs
        # Set initial position if provided
        if initial_pos:
            self.state.r = np.array(initial_pos)
            self.state.v = np.zeros(3)
            self.state.theta = self.state.omega = 0.0
            self.state.clear_events()

        # Initialize all outputs
        for output in self.outputs:
            output.initialize()

        # Run simulation steps
        for step in range(steps):
            channel = SignalChannel()
            channel = control_source.get_control(channel, step)
            control_input = channel[control_source.lain_index]
            if isinstance(control_input, (tuple, list)):
                control_input = np.array(control_input)
            elif isinstance(control_input, (int, float)):
                control_input = np.array([0.0, control_input])  # Assume torque or something
            self.step(control_input)

            # process step for all visualizers/outputs
            for visualizer in self.visualizers:
                visualizer.update(self.state)

            for output in self.outputs:
                output.process_step(step, control_input)

        # Finalize all outputs
        for output in self.outputs:
            output.finalize()


    def get_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        return self.config

    def close(self):
        """Clean up all registered visualizers."""
        for visualizer in self.visualizers:
            visualizer.close()