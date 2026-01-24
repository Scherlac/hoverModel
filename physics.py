from abc import ABC, abstractmethod
import numpy as np
from state import BodyState

class PhysicsEngine(ABC):
    """Abstract base class for physics simulation."""

    @abstractmethod
    def __init__(self, config: dict):
        """Initialize physics engine with configuration."""
        pass

    @abstractmethod
    def step(self, state: BodyState, action: np.ndarray, dt: float) -> BodyState:
        """Compute next state given current state and action."""
        pass

    @abstractmethod
    def get_bounds(self) -> dict:
        """Return environment bounds."""
        pass

class HovercraftPhysics(PhysicsEngine):
    """Newtonian physics for hovercraft simulation using vector operations.

    State vector format (8 elements):
    [x, y, z, theta, vx, vy, vz, omega_z]
    - x, y, z: position coordinates
    - theta: orientation angle (radians)
    - vx, vy, vz: velocity components
    - omega_z: angular velocity around z-axis
    """

    def __init__(self, config: dict):
        self.mass = config.get('mass', 1.0)
        self.I = config.get('momentum', 0.1)

        # Gravity vector (3D) - allows for non-vertical gravity or wind effects
        gravity_config = config.get('gravity', [0.0, 0.0, -9.81])
        self.gravity_vector = np.array(gravity_config)

        # Force parameters (scalars for now, could be extended to vectors)
        self.lift_mean = config.get('lift_mean', 10.0)
        self.lift_std = config.get('lift_std', 1.0)
        self.rot_mean = config.get('rot_mean', 0.1)
        self.rot_std = config.get('rot_std', 0.5)
        self.friction_k = config.get('friction_k', 0.1)

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

    def step(self, state: BodyState, action: np.ndarray, dt: float) -> BodyState:
        """Apply Newtonian physics using vectorized operations.

        Args:
            state: Current BodyState object
            action: Action vector [forward_force, rotation_torque]
            dt: Time step duration

        Returns:
            next_state: Updated BodyState object
        """
        # Extract state components using BodyState properties
        r = state.r.copy()      # position vector [x, y, z]
        theta = state.theta     # orientation angle
        v = state.v.copy()      # velocity vector [vx, vy, vz]
        omega = state.omega     # angular velocity

        # Unpack action
        F_forward, T_torque = action

        # Random forces
        F_lift = np.random.normal(self.lift_mean, self.lift_std)
        T_rot = np.random.normal(self.rot_mean, self.rot_mean)

        # Forward force in direction of orientation
        F_dir = F_forward * np.array([np.cos(theta), np.sin(theta), 0.0])

        # Friction force (proportional to height)
        F_friction = -self.friction_k * r[2] * v

        # Total force and torque
        F_total = F_dir + F_friction + np.array([0.0, 0.0, F_lift]) + self.mass * self.gravity_vector
        T_total = T_rot + T_torque

        # Integrate (vectorized)
        a = F_total / self.mass          # acceleration
        alpha = T_total / self.I         # angular acceleration

        v_new = v + a * dt               # velocity integration
        omega_new = omega + alpha * dt   # angular velocity integration

        r_new = r + v_new * dt           # position integration
        theta_new = theta + omega_new * dt  # orientation integration

        # Boundary collisions (vectorized bounce)
        r_new = np.clip(r_new, self.bounds_min, self.bounds_max)
        v_new = np.where(r_new == self.bounds_min, -v_new, v_new)
        v_new = np.where(r_new == self.bounds_max, -v_new, v_new)

        # Reconstruct state vector
        new_state = np.concatenate([r_new, [theta_new], v_new, [omega_new]])
        return BodyState(r=r_new, v=v_new, theta=theta_new, omega=omega_new)

    def get_bounds(self) -> dict:
        return self.bounds