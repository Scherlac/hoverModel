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
        # Initialize Genesis only if not already initialized
        if not gs._initialized:
            gs.init(backend=gs.cpu)

        # Store configuration
        self.config = config
        self.dt = config.get('dt', 0.01)

        # Create scene with built-in viewer enabled
        self.scene = gs.Scene(
            show_viewer=True,  # Enable Genesis built-in viewer for live visualization
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(5, 5, 5),  # Match default backend camera position
                camera_lookat=(0.0, 0.0, 1.0),  # Look at hovercraft initial position
                camera_fov=40,
            ),
            sim_options=gs.options.SimOptions(
                dt=self.dt,
            ),
        )

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
        # For Genesis, we need to rebuild the scene if it's already built
        if self.built:
            # This is a limitation - Genesis doesn't support dynamic body addition
            # We'll create the body but it won't be added to the physics simulation
            print(f"Warning: Cannot add body '{body_config}' to already-built Genesis scene")
            return None
            
        # Create Genesis entity based on body type
        body_type = body_config.get('type', 'hovercraft')

        if body_type == 'hovercraft':
            # Load hovercraft mesh
            mesh_file = body_config.get('mesh_file', ASSET_ROOT / "hoverBody_main.obj")
            entity = self.scene.add_entity(
                gs.morphs.Mesh(
                    file=str(mesh_file),
                    scale=body_config.get('scale', 1.0),  # Changed from 0.1 to 1.0 to match Open3D size
                    pos=tuple(body_config.get('initial_pos', [0.0, 0.0, 1.0])),
                )
            )
        elif body_type == 'sphere':
            entity = self.scene.add_entity(
                gs.morphs.Sphere(
                    radius=body_config.get('radius', 0.1),
                    pos=tuple(body_config.get('initial_pos', [0.0, 0.0, 1.0])),
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
            # Add boundary visualizations
            self._add_boundary_visualizations()
            
            # Add a camera for rendering before building
            if not hasattr(self, 'camera'):
                self.camera = self.scene.add_camera(
                    res=(1280, 720),
                    pos=(5, 5, 5),  # Match default backend camera position
                    lookat=(0, 0, 1),  # Match default backend look-at
                    fov=60,
                    up=(0, 0, 1),  # Z-up to match default backend
                    GUI=False
                )
            self.scene.build()
            self.built = True

    def ensure_scene_built(self):
        """Ensure the scene is built."""
        if not self.built:
            self.build_scene()

    def _add_boundary_visualizations(self):
        """Add visual elements to show environment boundaries."""
        x_min, x_max = self.bounds['x']
        y_min, y_max = self.bounds['y']
        z_min, z_max = self.bounds['z']

        # Add boundary lines using thin boxes
        # Issue: this implementation creates rigid bodies
        # that interacts with gravity and other bodies.
        # Boundary walls should ideally be static/non-movable.
        # visually transparent or semi-transparent.
        # For now, we comment this out to avoid interference.        
        # # Add boundary walls as thin boxes
        # wall_thickness = 0.05
        
        # # X boundaries (front and back walls)
        # self.scene.add_entity(gs.morphs.Box(
        #     pos=(x_min, (y_min + y_max) / 2, (z_min + z_max) / 2),
        #     size=(wall_thickness, y_max - y_min, z_max - z_min)
        # ))
        
        # self.scene.add_entity(gs.morphs.Box(
        #     pos=(x_max, (y_min + y_max) / 2, (z_min + z_max) / 2),
        #     size=(wall_thickness, y_max - y_min, z_max - z_min)
        # ))
        
        # # Y boundaries (left and right walls)
        # self.scene.add_entity(gs.morphs.Box(
        #     pos=((x_min + x_max) / 2, y_min, (z_min + z_max) / 2),
        #     size=(x_max - x_min, wall_thickness, z_max - z_min)
        # ))
        
        # self.scene.add_entity(gs.morphs.Box(
        #     pos=((x_min + x_max) / 2, y_max, (z_min + z_max) / 2),
        #     size=(x_max - x_min, wall_thickness, z_max - z_min)
        # ))
        
        # # Z boundaries (ceiling) - floor is already the ground plane
        # self.scene.add_entity(gs.morphs.Box(
        #     pos=((x_min + x_max) / 2, (y_min + y_max) / 2, z_max),
        #     size=(x_max - x_min, y_max - y_min, wall_thickness)
        # ))

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

    def apply_force(self, force: np.ndarray) -> None:
        """Apply a force vector to the Genesis body."""
        self.control_force = np.array(force, dtype=float)

    def apply_torque(self, torque: np.ndarray) -> None:
        """Apply a torque vector to the Genesis body."""
        self.control_torque = np.array(torque, dtype=float)

    def apply_angular_momentum(self, angular_momentum: np.ndarray) -> None:
        """Apply angular momentum to the Genesis body."""
        # Convert angular momentum to torque: τ = dL/dt
        # For simplicity, apply as direct torque
        self.control_torque = angular_momentum

    def set_position_target(self, target_position: np.ndarray) -> None:
        """Set position control target."""
        self.position_target = target_position

    def set_velocity_target(self, target_velocity: np.ndarray) -> None:
        """Set velocity control target."""
        self.velocity_target = target_velocity

    def apply_control(self, control_kind: str, control_value: Any) -> None:
        """
        Apply a control of the specified kind to the Genesis body.

        Args:
            control_kind: Type of control ('force', 'torque', 'angular_momentum', 'position', 'velocity')
            control_value: Value for the control
        """
        if control_kind == 'force':
            self.apply_force(np.array(control_value))
        elif control_kind == 'torque':
            self.apply_torque(np.array(control_value))
        elif control_kind == 'angular_momentum':
            self.apply_angular_momentum(np.array(control_value))
        elif control_kind == 'position':
            self.set_position_target(np.array(control_value))
        elif control_kind == 'velocity':
            self.set_velocity_target(np.array(control_value))
        elif control_kind == 'lifting thrust':
            # Lifting thrust is an upward force (positive z-direction)
            self.apply_force(np.array([0.0, 0.0, float(control_value)]))
        elif control_kind == 'combined':
            # Handle combined controls
            for kind, value in control_value.items():
                self.apply_control(kind, value)
        else:
            print(f"Warning: Unknown control kind '{control_kind}'")

        # Apply accumulated forces to Genesis entity
        self._apply_forces_to_entity()

    def _apply_forces_to_entity(self):
        """Apply accumulated forces and torques to the Genesis entity."""
        # Combine all forces and torques
        total_force = np.array(getattr(self, 'control_force', np.zeros(3)), dtype=float)
        total_torque = np.array(getattr(self, 'control_torque', np.zeros(3)), dtype=float)

        # Apply forces to Genesis entity
        # Genesis uses DOF forces: [fx, fy, fz, tx, ty, tz]
        force_vector = np.concatenate([total_force, total_torque])
        try:
            self.entity.control_dofs_force(force_vector)
        except Exception as e:
            print(f"Warning: Could not set forces on Genesis entity: {e}")

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
            self.entity.control_dofs_force(force_vector)
        except Exception as e:
            print(f"Warning: Could not set forces on Genesis entity: {e}")

    def get_state(self) -> BodyState:
        """Get current body state from Genesis entity."""
        try:
            # Get position from Genesis
            pos = self.entity.get_pos()
            self.state.r = np.array(pos)

            # Get orientation (quaternion to angle)
            quat = self.entity.get_quat()
            # Convert quaternion [w, x, y, z] to rotation around z-axis (theta)
            # theta = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
            w, x, y, z = quat[0], quat[1], quat[2], quat[3]
            self.state.theta = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

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
        self.bodies = []
        if bodies is None:
            # Default: single hovercraft
            hovercraft_config = {
                'type': 'hovercraft',
                'mass': self.config.get('mass', 1.0),
                'moment_of_inertia': self.config.get('momentum', 0.1),
                'initial_pos': (0.0, 0.0, 1.0),
                'scale': 1.0,  # Match Open3D visualizer size
                'mesh_file': ASSET_ROOT / "hoverBody_main.obj"
            }
            genesis_body = self.physics.add_body(hovercraft_config)
            self.bodies = [genesis_body]
        else:
            self.bodies = bodies

        # Don't build the scene yet - wait until all bodies are added
        # self.physics.build_scene()

        # For backward compatibility
        self.state = self.bodies[0].state if self.bodies else None
        
        # Step counter for outputs
        self.step_count = 0

    def add_body(self, body: Body) -> None:
        """Add a body to the environment."""
        if isinstance(body, GenesisRigidBody):
            self.bodies.append(body)
        else:
            # Convert other body types to Genesis bodies
            body_config = {
                'type': 'sphere',  # Default to sphere for unknown types
                'mass': body.mass,
                'moment_of_inertia': body.moment_of_inertia,
                'initial_pos': body.state.r.tolist(),
                'radius': 0.1
            }
            genesis_body = self.physics.add_body(body_config)
            if genesis_body:
                self.bodies.append(genesis_body)
        
        # Update state to first body
        self.state = self.bodies[0].state if self.bodies else None

    def get_body_by_id(self, body_id: str) -> Optional[Body]:
        """Get a body by its ID."""
        for body in self.bodies:
            if body.id == body_id:
                return body
        return None

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

    def run_simulation_with_controls(self, control_sources: List["ControlSource"], outputs=None, steps=50, initial_pos=None):
        """
        Run simulation with multiple control sources using the new control system.

        Args:
            control_sources: List of ControlSource objects
            outputs: List of output handlers
            steps: Number of simulation steps
            initial_pos: Initial position for bodies
        """
        # Build the scene now that all bodies are added
        self.physics.build_scene()

        if outputs is not None:
            self.outputs = outputs

        # Set initial position if provided
        if initial_pos:
            for body in self.bodies:
                body.state.r = np.array(initial_pos)
                body.state.v = np.zeros(3)
                body.state.theta = 0.0
                body.state.omega = 0.0

        # Initialize all outputs
        for output in self.outputs:
            output.initialize()

        # Run simulation steps
        for step in range(steps):
            # Create control channel and get controls from all sources
            from control_sources import SignalChannel
            channel = SignalChannel()
            for control_source in control_sources:
                channel = control_source.get_control(channel, step)

            # Apply controls to bodies
            self.apply_controls_from_channel(channel, self.dt)

            # Step physics for all bodies
            for body in self.bodies:
                # For Genesis, controls are applied directly to entities
                # Just step the scene
                self.physics.scene.step()

                # Get updated state
                if self.bodies:
                    state = self.bodies[0].get_state()
                    print(f"Step {step}: Position {state.r}, Velocity {state.v}")

            # Update visualizers
            for viz in self.visualizers:
                if self.bodies:
                    viz.update(self.bodies[0].state)

            # Process outputs
            for output in self.outputs:
                # For backward compatibility, pass the first body's control
                first_control = channel[control_sources[0].lain_index] if control_sources else np.zeros(2)
                if isinstance(first_control, (tuple, list)):
                    first_control = np.array(first_control)
                elif isinstance(first_control, (int, float)):
                    first_control = np.array([0.0, first_control])
                output.process_step(self.step_count, first_control)
                self.step_count += 1

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

    def __init__(self, env: Environment, render_mode: str = 'rgb'):
        super(GenesisVisualizer, self).__init__(env)
        self.env = env
        self.render_mode = render_mode
        # Genesis handles its own visualization
        self.viewer_active = False

    def update(self, state: BodyState):
        """Update visualization - Genesis handles this internally."""
        pass

    def get_visualization_output(self) -> VisualizationOutput:
        """Get visualization output handler."""
        return GenesisVisualizationOutput(self, self.render_mode)

    def close(self):
        """Clean up visualization resources."""
        pass


class GenesisVisualizationOutput(VisualizationOutput):
    """Genesis visualization output handler."""

    def __init__(self, visualizer: GenesisVisualizer, render_mode: str = 'rgb'):
        super(GenesisVisualizationOutput, self).__init__(visualizer)
        self.zoom = 0.6
        self.scene = visualizer.env.physics.scene
        # Camera will be created when scene is built
        self.camera = None
        self.render_mode = render_mode
        self.render_mode = render_mode  # 'rgb', 'depth', 'segmentation', 'normal'

    def set_camera(self, position: Tuple[float, float, float], look_at: Tuple[float, float, float]) -> None:
        """Set camera position."""
        if self.camera:
            self.camera.set_pose(pos=position, lookat=look_at)

    def set_zoom(self, zoom: float) -> None:
        """Set zoom level."""
        self.zoom = zoom
        # Adjust camera position based on zoom
        if self.camera:
            current_pos = self.camera.pos
            lookat = self.camera.lookat
            direction = np.array(lookat) - np.array(current_pos)
            direction = direction / np.linalg.norm(direction)
            new_pos = np.array(lookat) - direction * (10 / zoom)  # Adjust distance based on zoom
            self.camera.set_pose(pos=tuple(new_pos), lookat=lookat)

    def render_frame(self) -> None:
        """Render frame - handled by Genesis internally."""
        pass

    def start_recording(self) -> None:
        """Start video recording using Genesis camera."""
        # Ensure scene is built and camera is available
        if hasattr(self.visualizer, 'env') and hasattr(self.visualizer.env, 'physics'):
            self.visualizer.env.physics.ensure_scene_built()
            if self.camera is None:
                self.camera = getattr(self.visualizer.env.physics, 'camera', None)
        
        if self.camera and self.scene.is_built:
            try:
                self.camera.start_recording()
                print("Genesis camera recording started")
            except Exception as e:
                print(f"Failed to start Genesis recording: {e}")
                raise RuntimeError(f"Genesis recording not supported: {e}")
        else:
            error_msg = f"Camera not available ({self.camera is not None}) or scene not built ({self.scene.is_built if hasattr(self.scene, 'is_built') else 'no is_built attr'})"
            print(error_msg)
            raise RuntimeError(error_msg)

    def stop_recording(self, filename: str, fps: int = 25) -> None:
        """Stop video recording and save to file."""
        if self.camera:
            try:
                self.camera.stop_recording(save_to_filename=filename, fps=fps)
                print(f"Genesis video saved: {filename}")
            except Exception as e:
                print(f"Failed to stop Genesis recording: {e}")
                raise RuntimeError(f"Genesis recording save failed: {e}")
        else:
            print("No camera available for stopping recording")

    def capture_frame(self, filename: str, render_mode: str = 'rgb') -> None:
        """Capture frame to file using Genesis camera rendering.
        
        Args:
            filename: Output filename for the captured frame
            render_mode: Type of rendering ('rgb', 'depth', 'segmentation', 'normal')
        """
        # Ensure scene is built before rendering
        if hasattr(self.visualizer, 'env') and hasattr(self.visualizer.env, 'physics'):
            self.visualizer.env.physics.ensure_scene_built()
            # Get camera reference after scene is built
            if self.camera is None:
                self.camera = getattr(self.visualizer.env.physics, 'camera', None)
            
        if self.camera and self.scene.is_built:
            try:
                # Render the camera view using Genesis camera render method
                # Returns (rgb, depth, segmentation, normal) tuple
                if self.render_mode == 'rgb':
                    render_result = self.camera.render(rgb=True, depth=False, segmentation=False, normal=False)
                    image_data = render_result[0]  # RGB image
                    is_rgb = True
                elif self.render_mode == 'depth':
                    render_result = self.camera.render(rgb=False, depth=True, segmentation=False, normal=False)
                    image_data = render_result[1]  # Depth image
                    is_rgb = False
                elif self.render_mode == 'segmentation':
                    render_result = self.camera.render(rgb=False, depth=False, segmentation=True, normal=False)
                    image_data = render_result[2]  # Segmentation image
                    is_rgb = False
                elif self.render_mode == 'normal':
                    render_result = self.camera.render(rgb=False, depth=False, segmentation=False, normal=True)
                    image_data = render_result[3]  # Normal image
                    is_rgb = False
                else:
                    raise ValueError(f"Unsupported render_mode: {self.render_mode}")
                
                # Convert to PIL Image and save
                from PIL import Image
                import numpy as np
                
                # Convert from tensor to numpy array if needed
                if hasattr(image_data, 'cpu'):
                    image_data = image_data.cpu().numpy()
                
                if is_rgb:
                    # RGB images: ensure values are in 0-255 range
                    image_data = (image_data * 255).astype(np.uint8)
                    img = Image.fromarray(image_data)
                else:
                    # Depth/Segmentation/Normal: normalize to 0-255 range
                    if image_data.dtype != np.uint8:
                        # Normalize to 0-1 range first if needed
                        if image_data.max() > 1.0:
                            image_data = image_data / image_data.max()
                        # Convert to 0-255
                        image_data = (image_data * 255).astype(np.uint8)
                    
                    if render_mode == 'depth':
                        # Depth images are typically single channel
                        img = Image.fromarray(image_data.squeeze(), mode='L')
                    else:
                        # Segmentation and normal might be multi-channel
                        img = Image.fromarray(image_data)
                
                img.save(filename)
                print(f"Frame captured successfully ({self.render_mode}): {filename}")
            except Exception as e:
                print(f"Genesis rendering failed: {e}")
                print("Genesis 3D rendering requires BatchRenderer which is only available on Linux x86-64")
                print(f"Cannot create visualization frame: {filename}")
                raise RuntimeError(f"Genesis rendering not supported on this platform: {e}")
        else:
            error_msg = f"Camera not available ({self.camera is not None}) or scene not built ({self.scene.is_built if hasattr(self.scene, 'is_built') else 'no is_built attr'})"
            print(error_msg)
            raise RuntimeError(error_msg)

