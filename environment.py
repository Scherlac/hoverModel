import numpy as np
from physics import PhysicsEngine, HovercraftPhysics
from body import Body, Hovercraft
from state import BodyState
from components import Environment, SimulationComponent, SimulationOutput
from typing import (
    List, Optional,
    Dict, Tuple, Any,

)

class HovercraftEnv(Environment):
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
        super(HovercraftEnv, self).__init__()
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
        self.physics = physics_engine or HovercraftPhysics(self.config)

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
            self.bodies = [Hovercraft(**hovercraft_config)]
        else:
            self.bodies = bodies
    
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

    def get_bodies(self) -> list[Body]:
        """Get all bodies in the environment."""
        return self.bodies.copy()


    def run_simulation(self, control_source, steps: int, initial_pos=None):
        """Run simulation with control source and multiple outputs."""
        # Set initial position if provided
        if initial_pos:
            import numpy as np
            self.state.r = np.array(initial_pos)
            self.state.v = np.zeros(3)
            self.state.theta = self.state.omega = 0.0
            self.state.clear_events()

        # Initialize all outputs
        for output in self.outputs:
            output.initialize()

        # Run simulation steps
        for step in range(steps):
            control_input = control_source.get_control(step)
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