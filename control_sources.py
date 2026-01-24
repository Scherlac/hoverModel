"""
Control source generators for hovercraft demonstrations.
Follows SOLID principles with high cohesion, low coupling, and composition.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional


class ControlSource(ABC):
    """Abstract base class for control signal generators."""

    @abstractmethod
    def get_control(self, step: int) -> Tuple[float, float]:
        """Generate control signal for given step.

        Args:
            step: Current simulation step

        Returns:
            Tuple of (forward_force, rotation_torque)
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return human-readable description of this control source."""
        pass


class HoveringControl(ControlSource):
    """Generates zero control signals for hovering tests."""

    def get_control(self, step: int) -> Tuple[float, float]:
        return (0.0, 0.0)

    def get_description(self) -> str:
        return "Hovering (no control inputs)"


class LinearMovementControl(ControlSource):
    """Generates constant forward movement."""

    def __init__(self, forward_force: float = 0.8):
        self.forward_force = forward_force

    def get_control(self, step: int) -> Tuple[float, float]:
        return (self.forward_force, 0.0)

    def get_description(self) -> str:
        return f"Linear movement (forward_force={self.forward_force})"


class RotationalControl(ControlSource):
    """Generates pure rotational movement."""

    def __init__(self, rotation_torque: float = 0.3):
        self.rotation_torque = rotation_torque

    def get_control(self, step: int) -> Tuple[float, float]:
        return (0.0, self.rotation_torque)

    def get_description(self) -> str:
        return f"Rotational movement (torque={self.rotation_torque})"


class SinusoidalControl(ControlSource):
    """Generates sinusoidal combined movement."""

    def __init__(self,
                 forward_amplitude: float = 1.0,
                 rotation_amplitude: float = 0.3,
                 forward_freq: float = 0.05,
                 rotation_freq: float = 0.1):
        self.forward_amp = forward_amplitude
        self.rotation_amp = rotation_amplitude
        self.forward_freq = forward_freq
        self.rotation_freq = rotation_freq

    def get_control(self, step: int) -> Tuple[float, float]:
        forward = self.forward_amp * np.sin(step * self.forward_freq)
        rotation = self.rotation_amp * np.cos(step * self.rotation_freq)
        return (forward, rotation)

    def get_description(self) -> str:
        return f"Sinusoidal movement (amp_f={self.forward_amp}, amp_r={self.rotation_amp})"


class ChaoticControl(ControlSource):
    """Generates chaotic movement for boundary testing."""

    def __init__(self,
                 forward_amplitude: float = 20.0,  # Increased for more aggressive movement
                 rotation_amplitude: float = 5.0,  # Increased for more aggressive rotation
                 forward_freq: float = 0.12,
                 rotation_freq: float = 0.18):
        self.forward_amp = forward_amplitude
        self.rotation_amp = rotation_amplitude
        self.forward_freq = forward_freq
        self.rotation_freq = rotation_freq

    def get_control(self, step: int) -> Tuple[float, float]:
        forward = self.forward_amp * np.sin(step * self.forward_freq)
        rotation = self.rotation_amp * np.cos(step * self.rotation_freq)
        return (forward, rotation)

    def get_description(self) -> str:
        return f"Chaotic boundary movement (amp_f={self.forward_amp}, amp_r={self.rotation_amp})"


class ControlSourceFactory:
    """Factory for creating control sources."""

    @staticmethod
    def create_hovering() -> ControlSource:
        return HoveringControl()

    @staticmethod
    def create_linear(forward_force: float = 0.8) -> ControlSource:
        return LinearMovementControl(forward_force)

    @staticmethod
    def create_rotational(rotation_torque: float = 0.3) -> ControlSource:
        return RotationalControl(rotation_torque)

    @staticmethod
    def create_sinusoidal() -> ControlSource:
        return SinusoidalControl()

    @staticmethod
    def create_chaotic() -> ControlSource:
        return ChaoticControl()