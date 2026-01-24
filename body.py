"""
Physical body representations with properties and behaviors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
from state import BodyState


class Body(ABC):
    """Abstract base class for physical bodies with mass, shape, and dynamics."""

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
        self.mass = mass
        self.moment_of_inertia = moment_of_inertia
        self.shape = shape or {'type': 'sphere', 'radius': 0.5}

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


class Hovercraft(Body):
    """Hovercraft-specific body with lifting force and control characteristics."""

    def __init__(self,
                 mass: float = 1.0,
                 moment_of_inertia: float = 0.1,
                 lift_force_mean: float = 10.0,
                 lift_force_std: float = 1.0,
                 rotational_noise_mean: float = 0.1,
                 rotational_noise_std: float = 0.5,
                 friction_coefficient: float = 0.1):
        """
        Initialize hovercraft body.

        Args:
            mass: Hovercraft mass in kg
            moment_of_inertia: Rotational inertia
            lift_force_mean: Mean lifting force
            lift_force_std: Standard deviation of lifting force
            rotational_noise_mean: Mean rotational noise
            rotational_noise_std: Standard deviation of rotational noise
            friction_coefficient: Ground friction coefficient
        """
        super().__init__(mass, moment_of_inertia, {'type': 'hovercraft', 'radius': 0.5})

        # Hovercraft-specific properties
        self.lift_force_mean = lift_force_mean
        self.lift_force_std = lift_force_std
        self.rotational_noise_mean = rotational_noise_mean
        self.rotational_noise_std = rotational_noise_std
        self.friction_coefficient = friction_coefficient

    def get_forces(self, action: np.ndarray, environment_state: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        """
        Calculate hovercraft-specific forces and torques.

        Args:
            action: [forward_force, rotation_torque]
            environment_state: Environment conditions

        Returns:
            Tuple of (total_force_vector, total_torque_scalar)
        """
        forward_force, rotation_torque = action

        # Environment forces
        gravity = environment_state.get('gravity', np.array([0.0, 0.0, -9.81]))

        # Hovercraft-specific forces
        lift_force = np.random.normal(self.lift_force_mean, self.lift_force_std)
        lift_vector = np.array([0.0, 0.0, lift_force])

        # Forward force in direction of orientation
        forward_vector = forward_force * np.array([
            np.cos(self.state.theta),
            np.sin(self.state.theta),
            0.0
        ])

        # Friction force (proportional to height and velocity)
        friction_force = -self.friction_coefficient * self.state.r[2] * self.state.v

        # Rotational noise
        rotational_noise = np.random.normal(self.rotational_noise_mean, self.rotational_noise_std)

        # Total forces and torques
        total_force = gravity * self.mass + lift_vector + forward_vector + friction_force
        total_torque = rotation_torque + rotational_noise

        return total_force, total_torque

    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get hovercraft movement bounds."""
        return {
            'x': (-5.0, 5.0),
            'y': (-5.0, 5.0),
            'z': (0.0, 10.0)
        }

    def copy(self) -> 'Hovercraft':
        """Create a copy of this hovercraft."""
        hovercraft = Hovercraft(
            mass=self.mass,
            moment_of_inertia=self.moment_of_inertia,
            lift_force_mean=self.lift_force_mean,
            lift_force_std=self.lift_force_std,
            rotational_noise_mean=self.rotational_noise_mean,
            rotational_noise_std=self.rotational_noise_std,
            friction_coefficient=self.friction_coefficient
        )
        hovercraft.state = self.state.copy()
        return hovercraft

    def __repr__(self) -> str:
        return f"Hovercraft(mass={self.mass}, state={self.state})"