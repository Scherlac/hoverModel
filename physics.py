from abc import ABC, abstractmethod
import numpy as np

class PhysicsEngine(ABC):
    """Abstract base class for physics simulation."""

    @abstractmethod
    def __init__(self, config: dict):
        """Initialize physics engine with configuration."""
        pass

    @abstractmethod
    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        """Compute next state given current state and action."""
        pass

    @abstractmethod
    def get_bounds(self) -> dict:
        """Return environment bounds."""
        pass

class HovercraftPhysics(PhysicsEngine):
    """Newtonian physics for hovercraft simulation using vector operations."""

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

    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        """Apply Newtonian physics to compute next state using vector operations."""
        # Unpack state vector
        position = state[:3]      # [x, y, z]
        theta = state[3]          # orientation
        velocity = state[4:7]     # [vx, vy, vz]
        omega_z = state[7]        # angular velocity

        # Unpack action
        forward_force, rotation_torque = action

        # Random forces (vectorized)
        F_lift = np.random.normal(self.lift_mean, self.lift_std)
        T_rot = np.random.normal(self.rot_mean, self.rot_mean)

        # Controlled forces (vectorized)
        # Forward force in direction of orientation
        forward_direction = np.array([np.cos(theta), np.sin(theta), 0.0])
        F_forward = forward_force * forward_direction

        # Friction force (proportional to height and velocity)
        # F_friction = -k * z * v for each component
        friction_coefficient = self.friction_k * position[2]  # z component
        F_friction = -friction_coefficient * velocity

        # Total force vector (3D)
        F_total = F_forward + F_friction + np.array([0.0, 0.0, F_lift]) + self.mass * self.gravity_vector

        # Torque (scalar for 2D rotation)
        T_total = T_rot + rotation_torque

        # Accelerations (vectorized)
        acceleration = F_total / self.mass
        alpha_z = T_total / self.I

        # Integrate velocities (vectorized)
        velocity_new = velocity + acceleration * dt
        omega_z_new = omega_z + alpha_z * dt

        # Integrate positions (vectorized)
        position_new = position + velocity_new * dt
        theta_new = theta + omega_z_new * dt

        # Boundary collisions (vectorized bounce)
        # Check lower bounds
        below_min = position_new < self.bounds_min
        position_new = np.where(below_min, self.bounds_min, position_new)
        velocity_new = np.where(below_min, -velocity_new, velocity_new)

        # Check upper bounds
        above_max = position_new > self.bounds_max
        position_new = np.where(above_max, self.bounds_max, position_new)
        velocity_new = np.where(above_max, -velocity_new, velocity_new)

        # Reconstruct state vector
        return np.concatenate([position_new, [theta_new], velocity_new, [omega_z_new]])

    def get_bounds(self) -> dict:
        return self.bounds