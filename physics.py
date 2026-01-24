from abc import ABC, abstractmethod
import numpy as np
from state import BodyState
from body import Body


class PhysicsEngine(ABC):
    """Abstract base class for physics simulation."""

    @abstractmethod
    def __init__(self, config: dict):
        """Initialize physics engine with configuration."""
        pass

    @abstractmethod
    def step(self, body: Body, action: np.ndarray, dt: float) -> BodyState:
        """Compute next state for a body given action and time step."""
        pass

    @abstractmethod
    def get_bounds(self) -> dict:
        """Return environment bounds."""
        pass


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
        r_new = np.clip(r_new, self.bounds_min, self.bounds_max)
        # Bounce velocity when hitting bounds
        v_new = np.where(r_new == self.bounds_min, -v_new, v_new)
        v_new = np.where(r_new == self.bounds_max, -v_new, v_new)

        # Create new state
        new_state = BodyState(r=r_new, v=v_new, theta=theta_new, omega=omega_new)
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


# Backward compatibility alias
HovercraftPhysics = NewtonianPhysics