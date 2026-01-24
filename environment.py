import numpy as np
from physics import PhysicsEngine, HovercraftPhysics
from visualization import Visualizer, Open3DVisualizer, NullVisualizer
from state import BodyState

class HovercraftEnv:
    """
    Composable hovercraft environment using dependency injection.

    State vector format (8 elements):
    [x, y, z, theta, vx, vy, vz, omega_z]
    - x, y, z: position coordinates
    - theta: orientation angle (radians)
    - vx, vy, vz: velocity components
    - omega_z: angular velocity around z-axis

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

        # Initialize state - environment owns the state
        self.state = BodyState()

        # Lazy initialization of visualizer
        if self.visualizer is None:
            self.visualizer = self._create_default_visualizer()

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
            'bounds': [[-5, 5], [-5, 5], [0, 10]],
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

    def step(self, action: np.ndarray) -> BodyState:
        """
        Advance environment by one time step.

        Args:
            action: [forward_force, rotation_torque]

        Returns:
            next_state: Updated BodyState object
        """
        self.state = self.physics.step(self.state, action, self.dt)
        self.visualizer.update(np.array(self.state))
        return self.state

    def render(self):
        """Render current state."""
        self.visualizer.render()

    def close(self):
        """Clean up resources."""
        self.visualizer.close()

    def reset(self) -> BodyState:
        """Reset environment to initial state."""
        self.state.reset()
        self.visualizer.update(np.array(self.state))
        return self.state

    # Convenience properties for state access - delegate to BodyState
    @property
    def position(self) -> np.ndarray:
        """Get current position [x, y, z]."""
        return self.state.position

    @property
    def orientation(self) -> float:
        """Get current orientation theta."""
        return self.state.orientation

    @property
    def velocity(self) -> np.ndarray:
        """Get current velocity [vx, vy, vz]."""
        return self.state.velocity

    @property
    def angular_velocity(self) -> float:
        """Get current angular velocity omega_z."""
        return self.state.angular_velocity

    def save_state(self, filepath: str) -> None:
        """Save current state to file."""
        self.state.save(filepath)

    def load_state(self, filepath: str) -> BodyState:
        """Load state from file."""
        self.state = BodyState.load(filepath)
        self.visualizer.update(np.array(self.state))
        return self.state

    def get_state_dict(self) -> dict:
        """Get current state as dictionary."""
        return self.state.to_dict()

    def set_state_dict(self, state_dict: dict) -> None:
        """Set state from dictionary."""
        self.state = BodyState.from_dict(state_dict)
        self.visualizer.update(np.array(self.state))

    def capture_frame(self, filename: str) -> None:
        """Capture current visualization frame."""
        self.visualizer.capture_frame(filename)