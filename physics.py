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
    """Newtonian physics for hovercraft simulation."""

    def __init__(self, config: dict):
        self.mass = config.get('mass', 1.0)
        self.I = config.get('momentum', 0.1)
        self.gravity = config.get('gravity', -9.81)

        # Force parameters
        self.lift_mean = config.get('lift_mean', 10.0)
        self.lift_std = config.get('lift_std', 1.0)
        self.rot_mean = config.get('rot_mean', 0.1)
        self.rot_std = config.get('rot_std', 0.5)
        self.friction_k = config.get('friction_k', 0.1)

        # Bounds
        self.bounds = {
            'x': config.get('x_bounds', (-5, 5)),
            'y': config.get('y_bounds', (-5, 5)),
            'z': config.get('z_bounds', (0, 10))
        }

    def step(self, state: np.ndarray, action: np.ndarray, dt: float) -> np.ndarray:
        """Apply Newtonian physics to compute next state."""
        forward_force, rotation_torque = action
        x, y, z, theta, vx, vy, vz, omega_z = state

        # Random forces
        F_lift = np.random.normal(self.lift_mean, self.lift_std)
        T_rot = np.random.normal(self.rot_mean, self.rot_std)

        # Controlled forces
        F_forward_x = forward_force * np.cos(theta)
        F_forward_y = forward_force * np.sin(theta)
        T_control = rotation_torque

        # Friction
        F_friction_x = -self.friction_k * z * vx
        F_friction_y = -self.friction_k * z * vy
        F_friction_z = -self.friction_k * z * vz

        # Total forces
        Fx = F_forward_x + F_friction_x
        Fy = F_forward_y + F_friction_y
        Fz = F_lift + self.mass * self.gravity + F_friction_z
        Tz = T_rot + T_control

        # Accelerations
        ax, ay, az = Fx / self.mass, Fy / self.mass, Fz / self.mass
        alpha_z = Tz / self.I

        # Integrate velocities
        vx += ax * dt
        vy += ay * dt
        vz += az * dt
        omega_z += alpha_z * dt

        # Integrate positions
        x += vx * dt
        y += vy * dt
        z += vz * dt
        theta += omega_z * dt

        # Boundary collisions (bounce)
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = self.bounds.values()
        if x < x_min: x, vx = x_min, -vx
        elif x > x_max: x, vx = x_max, -vx
        if y < y_min: y, vy = y_min, -vy
        elif y > y_max: y, vy = y_max, -vy
        if z < z_min: z, vz = z_min, -vz
        elif z > z_max: z, vz = z_max, -vz

        return np.array([x, y, z, theta, vx, vy, vz, omega_z])

    def get_bounds(self) -> dict:
        return self.bounds