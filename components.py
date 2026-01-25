
from abc import ABC, abstractmethod

from typing import (
    Tuple, List,
    Any, Dict,

)

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

class SimulationComponent(ABC):
    """Abstract base class for simulation output handlers."""

    def __init__(self, env: Environment):
        self.env = env
        self.step_count = 0
        self.register_component()

    @abstractmethod
    def register_component(self) -> None:
        """Register a simulation component."""
        pass

class SimulationOutput(SimulationComponent):
    """Abstract base class for simulation output handlers."""

    def __init__(self, env: Environment):
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

class Visualizer(SimulationComponent):
    """Abstract base class for visualization backends."""

    def __init__(self, env: Environment):
        super(Visualizer, self).__init__(env)
        self.bounds : Dict[str, Tuple[float, float]] = {}
        self.register_component()

    def register_component(self) -> None:
        self.env.register_visualizer(self)

    @abstractmethod
    def update(self, state):
        """Update visualization with current state."""
        pass

    @abstractmethod
    def render(self):
        """Render the current frame."""
        pass

    @abstractmethod
    def close(self):
        """Clean up visualization resources."""
        pass

    @abstractmethod
    def capture_frame(self, filename: str):
        """Capture current frame to file."""
        pass
