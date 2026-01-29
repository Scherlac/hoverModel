
from abc import ABC, abstractmethod

from typing import (
    Tuple, List,
    Any, Dict, Optional
)
import numpy as np

from state import BodyState



class SimulationComponent(ABC):
    """Abstract base class for simulation output handlers."""

    def __init__(self, env: "Environment"):
        self.env = env
        self.step_count = 0
        self.register_component()

    @abstractmethod
    def register_component(self) -> None:
        """Register a simulation component."""
        pass

class SimulationOutput(SimulationComponent):
    """Abstract base class for simulation output handlers."""

    def __init__(self, env: "Environment"):
        super(SimulationOutput, self).__init__(env)

    def register_component(self):
        self.env.register_output(self)

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the output handler."""
        pass

    @abstractmethod
    def process_step(self, step: int, control: Tuple[float, float]) -> None:
        """Process a single simulation step."""
        pass

    @abstractmethod
    def finalize(self) -> None:
        """Finalize and cleanup the output handler."""
        pass

class VisualizationOutput(ABC):
    """Abstract base class for visualization output handlers."""

    def __init__(self, visualizer: "Visualizer"):
        self.visualizer = visualizer

    @abstractmethod
    def set_camera(self, position: Tuple[float, float, float], look_at: Tuple[float, float, float]) -> None:
        """Set camera position and orientation."""
        pass

    @abstractmethod
    def render_frame(self) -> None:
        """Render the current frame."""
        pass

    @abstractmethod
    def capture_frame(self, filename: str) -> None:
        """Capture current frame to file."""
        pass

class Visualizer(SimulationComponent):
    """Abstract base class for visualization backends."""

    def __init__(self, env: "Environment"):
        super(Visualizer, self).__init__(env)
        self.bounds : Dict[str, Tuple[float, float]] = {}
        self.register_component()

    def register_component(self) -> None:
        self.env.register_visualizer(self)

    @abstractmethod
    def update(self, state: BodyState):
        """Update visualization with current state."""
        """ Call this every simulation step to update the display. """
        pass

    @abstractmethod
    def get_visualization_output(self) -> VisualizationOutput:
        """Get visualization output handler."""
        pass

    @abstractmethod
    def close(self):
        """Clean up visualization resources."""
        pass

class NullVisualizationOutput(VisualizationOutput):
    """No-op visualization output."""

    def __init__(self, visualizer: "Visualizer"):
        super(NullVisualizationOutput, self).__init__(visualizer)

    def set_camera(self, position: Tuple[float, float, float], look_at: Tuple[float, float, float]) -> None:
        pass

    def render_frame(self) -> None:
        pass

    def capture_frame(self, filename: str) -> None:
        pass

class NullVisualizer(Visualizer):
    """No-op visualizer for headless operation."""

    def __init__(self, env: "Environment"):
        super(NullVisualizer, self).__init__(env)

    def update(self, state: BodyState):
        pass

    def get_visualization_output(self) -> VisualizationOutput:
        return NullVisualizationOutput(self)

    def close(self):
        pass


class Environment(ABC):
    """Abstract base class for simulation environments."""
    def __init__(self):
        self.outputs: List["SimulationOutput"] = []
        self.visualizers: List["Visualizer"] = []

    def register_output(self, output: "SimulationOutput") -> None:
        """Register a simulation output component."""
        self.outputs.append(output)

    def register_visualizer(self, visualizer: "Visualizer") -> None:
        """Register a visualization component."""
        self.visualizers.append(visualizer)

    def get_specific_visualizer(self, visualizer_type: type) -> Any:
        """Get a specific visualizer by type."""
        for viz in self.visualizers:
            if isinstance(viz, visualizer_type):
                return viz
        try:

            return visualizer_type(self)
        except Exception as e:
            print(f"Error creating visualizer {visualizer_type}: {e}")
            return NullVisualizer(self)

    @property
    def bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get environment bounds."""
        return self._bounds
    @bounds.setter
    def bounds(self, value: Dict[str, Tuple[float, float]]) -> None:
        self._bounds = value

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        pass

# SRC: https://stackoverflow.com/q/60104854
class Body(ABC):
    """Abstract base class for physical bodies with mass, shape, and dynamics."""

    _next_id = 0

    def __init__(self,
                 mass: float = 1.0,
                 moment_of_inertia: float = 0.1,
                 shape: Optional[Dict[str, Any]] = None):
        """
        Initialize physical body.

        Args:
            mass: Body mass in kg
            moment_of_inertia: Rotational inertia around z-axis
            shape: Shape description (sphere, box, etc.)
        """
        self.id = f"body_{Body._next_id}"
        Body._next_id += 1
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia
        self.shape = shape or {'type': 'mesh', 'mesh_path': 'hoverBody_main.obj'}
        # Initialize state
        self.state = BodyState()

    @abstractmethod
    def get_forces(self, action: np.ndarray, environment_state: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """
        Calculate forces and torques acting on the body.

        Args:
            action: Control inputs [forward_force, rotation_torque]
            environment_state: Environment conditions (gravity, wind, etc.)

        Returns:
            Tuple of (force_vector, torque_scalar)
        """
        pass

    @abstractmethod
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get movement bounds for this body type."""
        pass

    def reset(self) -> None:
        """Reset body state to initial conditions."""
        self.state.reset()

    def get_state(self) -> BodyState:
        """Get current body state."""
        return self.state

    def set_state(self, state: BodyState) -> None:
        """Set body state."""
        self.state = state

    def copy(self) -> 'Body':
        """Create a copy of this body."""
        # This is abstract - concrete implementations should override
        raise NotImplementedError


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


