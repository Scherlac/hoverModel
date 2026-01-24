import numpy as np
from physics import PhysicsEngine, HovercraftPhysics
from visualization import Visualizer, Open3DVisualizer, NullVisualizer

class HovercraftEnv:
    """
    Composable hovercraft environment using dependency injection.

    High cohesion: Single responsibility - orchestrate physics and visualization
    Low coupling: Depends on abstractions, not concrete implementations
    Good composition: Built from interchangeable components
    """

    def __init__(self,
                 physics_engine: PhysicsEngine = None,
                 visualizer: Visualizer = None,
                 config: dict = None):
        """
        Initialize environment with injected dependencies.

        Args:
            physics_engine: Physics simulation component
            visualizer: Visualization component
            config: Environment configuration
        """
        self.config = config or self._default_config()
        self.dt = self.config.get('dt', 0.01)

        # Dependency injection with defaults
        self.physics = physics_engine or HovercraftPhysics(self.config)
        self.visualizer = visualizer

        # Initialize state
        self.state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Lazy initialization of visualizer
        if self.visualizer is None:
            self.visualizer = self._create_default_visualizer()

    def _default_config(self) -> dict:
        """Default environment configuration."""
        return {
            'mass': 1.0,
            'momentum': 0.1,
            'gravity': -9.81,
            'dt': 0.01,
            'lift_mean': 10.0,
            'lift_std': 1.0,
            'rot_mean': 0.1,
            'rot_std': 0.5,
            'friction_k': 0.1,
            'x_bounds': (-5, 5),
            'y_bounds': (-5, 5),
            'z_bounds': (0, 10),
            'visualization': True
        }

    def _create_default_visualizer(self) -> Visualizer:
        """Create default visualizer based on config."""
        if self.config.get('visualization', True):
            try:
                return Open3DVisualizer(self.physics.get_bounds())
            except ImportError:
                print("Open3D not available, using null visualizer")
                return NullVisualizer(self.physics.get_bounds())
        else:
            return NullVisualizer(self.physics.get_bounds())

    def step(self, action: np.ndarray) -> np.ndarray:
        """
        Advance environment by one time step.

        Args:
            action: [forward_force, rotation_torque]

        Returns:
            next_state: Updated environment state
        """
        self.state = self.physics.step(self.state, action, self.dt)
        self.visualizer.update(self.state)
        return self.state

    def render(self):
        """Render current state."""
        self.visualizer.render()

    def close(self):
        """Clean up resources."""
        self.visualizer.close()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.visualizer.update(self.state)
        return self.state

    # Convenience properties for state access
    @property
    def position(self) -> np.ndarray:
        """Get current position [x, y, z]."""
        return self.state[:3]

    @property
    def orientation(self) -> float:
        """Get current orientation theta."""
        return self.state[3]

    @property
    def velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy, vz]."""
        return self.state[4:7]

    @property
    def angular_velocity(self) -> float:
        """Get current angular velocity omega_z."""
        return self.state[7]

    def capture_frame(self, filename: str):
        """Capture current visualization frame."""
        self.visualizer.capture_frame(filename)