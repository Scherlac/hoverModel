"""
Control source generators for hovercraft demonstrations.
Follows SOLID principles with high cohesion, low coupling, and composition.
"""

from abc import ABC, abstractmethod
from blinker import Signal
import numpy as np
from typing import (
    Tuple, 
    List,
    Optional,
    Union,
    Dict,
    Hashable,
    Any,
)

class LainInfo():
    """Type alias for Lain information dictionary."""

    index : int = 0
    # name : str
    # typeinfo : type
    # description : str

    def __init__(
            self, 
            id: str, 
            kind: str, 
            typeinfo: type, 
            description: str):
        
        LainInfo.index += 1
        self.index = LainInfo.index
        
        self.kind = kind
        self.typeinfo = typeinfo
        self.id = id
        self.description = description

class SignalChannel(List[Union[float, np.ndarray]]):
    """Type alias for control signal channels."""
    
    def __init__(self, *args: Union[float, np.ndarray]):
        super().__init__(args)

        # update the size to match the number of registered lains
        while len(self) < len(self.lain_info):
            self.append(None)

    lain_info : Dict[Tuple[str, str, str], LainInfo] = {}

    @classmethod
    def register_signal_source(
        cls,
        id: str,
        kind: Optional[str] = None,
        typeinfo: Optional[type] = float,
        description: Optional[str] = None,
    ) -> LainInfo:
        """Register a control signal source with Lain.

        Args:
            id: Unique identifier for the body to control
            kind: describes the kind of signal source
            typeinfo: Data type of the signal
            description: Human-readable description
        Returns:
            LainInfo object for the registered signal source
        """
        typename = typeinfo.__name__
        lain_key = (id, kind, typename)
        if cls.lain_info.get(lain_key):
            return cls.lain_info[lain_key]
        
        info: LainInfo = LainInfo(id, kind, typeinfo, description)
        cls.lain_info[lain_key] = info
        return info


class ControlSource(ABC):
    """Abstract base class for control signal generators."""

    def __init__(
            self, 
            id: str,
            steps_range: Optional[List[Optional[int]]] = None,
            kind: Optional[str] = "force",
            typeinfo: Optional[type] = np.ndarray,
            description: Optional[str] = "Provides control signals for body movement",
            ):
        super().__init__()
        self.lain_info = SignalChannel.register_signal_source(
            id = id,
            kind = kind,
            typeinfo = typeinfo,
            description = description,
        )
        """
        Initialize control source.
        Args:
            id: Unique identifier for body to control
            steps_range: [start, end] range where control is active, None means entire simulation
            kind: Kind of the control source
            description: Human-readable description
        """

        self.lain_index = self.lain_info.index
        self.steps_range = steps_range or [None, None]

    def is_active_at_step(self, step: int) -> bool:
        """Check if this control is active at the given step.
        
        Args:
            step: Current simulation step (0-based)
            
        Returns:
            True if control should be active at this step
        """
        start, end = self.steps_range
        if start is not None and step < start:
            return False
        if end is not None and step >= end:
            return False
        return True

    @abstractmethod
    def get_control(self, channel : SignalChannel, step: int) -> SignalChannel:
        """Generate control signal for given step.

        Args:
            channel: Input signal channel to insert control into
            step: Current simulation step

        Returns:
            Tuple of (forward_force, rotation_torque)
        """
        pass



class LinearMovementControl(ControlSource):
    """Generates constant forward movement."""

    def __init__(
            self, id: str, 
            forward_force: float = 0.8,
            steps_range: Optional[List[Optional[int]]] = None):
        
        super(LinearMovementControl, self).__init__(
            id = id,
            steps_range = steps_range
        )
        self.forward_force = forward_force

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        if not self.is_active_at_step(step):
            return channel
        while len(channel) <= self.lain_index:
            channel.append(None)
        channel[self.lain_index] = np.array([self.forward_force, 0.0, 0.0])
        return channel

    def get_description(self) -> str:
        return f"Linear movement (forward_force={self.forward_force})"


class RotationalControl(ControlSource):
    """Generates pure rotational movement."""

    def __init__(self, id: str, rotation_torque: float = 0.3, steps_range: Optional[List[Optional[int]]] = None):
        super(RotationalControl, self).__init__(id=id, steps_range=steps_range, kind='torque')
        self.rotation_torque = rotation_torque

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        if not self.is_active_at_step(step):
            return channel
        while len(channel) <= self.lain_index:
            channel.append(None)
        channel[self.lain_index] = np.array([0.0, 0.0, self.rotation_torque])
        return channel

    def get_description(self) -> str:
        return f"Rotational movement (torque={self.rotation_torque})"


class SinusoidalControl(ControlSource):
    """Generates sinusoidal combined movement."""

    def __init__(self,
                 id: str,
                 forward_amplitude: float = 1.0,
                 rotation_amplitude: float = 0.3,
                 forward_freq: float = 0.05,
                 rotation_freq: float = 0.1,
                 steps_range: Optional[List[Optional[int]]] = None):
        super(SinusoidalControl, self).__init__(
            id = id,
            steps_range = steps_range
        )
        self.forward_amp = forward_amplitude
        self.rotation_amp = rotation_amplitude
        self.forward_freq = forward_freq
        self.rotation_freq = rotation_freq

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        if not self.is_active_at_step(step):
            return channel
        while len(channel) <= self.lain_index:
            channel.append(None)
        forward = self.forward_amp * np.sin(step * self.forward_freq)
        rotation = self.rotation_amp * np.cos(step * self.rotation_freq)
        channel[self.lain_index] = np.array([forward, 0.0, 0.0])
        return channel

    def get_description(self) -> str:
        return f"Sinusoidal movement (amp_f={self.forward_amp}, amp_r={self.rotation_amp})"


class ChaoticControl(ControlSource):
    """Generates chaotic movement for boundary testing."""

    def __init__(self,
                id: str,
                 forward_amplitude: float = 20.0,  # Increased for more aggressive movement
                 rotation_amplitude: float = 5.0,  # Increased for more aggressive rotation
                 forward_freq: float = 0.12,
                 rotation_freq: float = 0.18,
                 steps_range: Optional[List[Optional[int]]] = None):
        super(ChaoticControl, self).__init__(
            id = id,
            steps_range = steps_range
        )
        self.forward_amp = forward_amplitude
        self.rotation_amp = rotation_amplitude
        self.forward_freq = forward_freq
        self.rotation_freq = rotation_freq

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        if not self.is_active_at_step(step):
            return channel
        while len(channel) <= self.lain_index:
            channel.append(None)
        forward = self.forward_amp * np.sin(step * self.forward_freq)
        rotation = self.rotation_amp * np.cos(step * self.rotation_freq)
        channel[self.lain_index] = np.array([forward, 0.0, 0.0])
        return channel

    def get_description(self) -> str:
        return f"Chaotic boundary movement (amp_f={self.forward_amp}, amp_r={self.rotation_amp})"

class HoveringControl(ControlSource):
    """Generates zero control signals for hovering tests."""

    def __init__(self, id: str, steps_range: Optional[List[Optional[int]]] = None):
        super(HoveringControl, self).__init__(
            id = id,
            steps_range = steps_range,
            kind = "lifting thrust",
            typeinfo = float,
            description = "Provides control signals for ground effects (hovering)",
        )

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        if not self.is_active_at_step(step):
            return channel
        while len(channel) <= self.lain_index:
            channel.append(None)
        channel[self.lain_index] = 10.0  # Provide 10N upward thrust to counteract gravity
        return channel

    def get_description(self) -> str:
        return "Hovering (no control inputs)"


class ForceControl(ControlSource):
    """Generates constant force control signals."""

    def __init__(self, id: str, force: np.ndarray = np.array([1.0, 0.0, 0.0]), steps_range: Optional[List[Optional[int]]] = None):
        super(ForceControl, self).__init__(
            id = id,
            steps_range = steps_range,
            kind = "force",
            typeinfo = np.ndarray,
            description = "Provides constant 3D force vector control",
        )
        self.force = force

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        if not self.is_active_at_step(step):
            return channel
        while len(channel) <= self.lain_index:
            channel.append(None)
        channel[self.lain_index] = self.force.copy()
        return channel

    def get_description(self) -> str:
        return f"Constant force (force={self.force})"


class TorqueControl(ControlSource):
    """Generates constant torque control signals."""

    def __init__(self, id: str, torque: np.ndarray = np.array([0.0, 0.0, 0.1]), steps_range: Optional[List[Optional[int]]] = None):
        super(TorqueControl, self).__init__(
            id = id,
            steps_range = steps_range,
            kind = "torque",
            typeinfo = np.ndarray,
            description = "Provides constant 3D torque vector control",
        )
        self.torque = torque

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        if not self.is_active_at_step(step):
            return channel
        while len(channel) <= self.lain_index:
            channel.append(None)
        channel[self.lain_index] = self.torque.copy()
        return channel

    def get_description(self) -> str:
        return f"Constant torque (torque={self.torque})"


class AngularMomentumControl(ControlSource):
    """Generates angular momentum control signals."""

    def __init__(self, id: str, angular_momentum: np.ndarray = np.array([0.0, 0.0, 0.05]), steps_range: Optional[List[Optional[int]]] = None):
        super(AngularMomentumControl, self).__init__(
            id = id,
            steps_range = steps_range,
            kind = "angular_momentum",
            typeinfo = np.ndarray,
            description = "Provides angular momentum vector control",
        )
        self.angular_momentum = angular_momentum

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        if not self.is_active_at_step(step):
            return channel
        while len(channel) <= self.lain_index:
            channel.append(None)
        channel[self.lain_index] = self.angular_momentum.copy()
        return channel

    def get_description(self) -> str:
        return f"Angular momentum (L={self.angular_momentum})"


class PositionControl(ControlSource):
    """Generates position setpoint control signals."""

    def __init__(self, id: str, target_position: np.ndarray = np.array([2.0, 0.0, 1.0]), steps_range: Optional[List[Optional[int]]] = None):
        super(PositionControl, self).__init__(
            id = id,
            steps_range = steps_range,
            kind = "position",
            typeinfo = np.ndarray,
            description = "Provides target position setpoint control",
        )
        self.target_position = target_position

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        if not self.is_active_at_step(step):
            return channel
        while len(channel) <= self.lain_index:
            channel.append(None)
        channel[self.lain_index] = self.target_position.copy()
        return channel

    def get_description(self) -> str:
        return f"Position control (target={self.target_position})"


class VelocityControl(ControlSource):
    """Generates velocity setpoint control signals."""

    def __init__(self, id: str, target_velocity: np.ndarray = np.array([1.0, 0.0, 0.0]), steps_range: Optional[List[Optional[int]]] = None):
        super(VelocityControl, self).__init__(
            id = id,
            steps_range = steps_range,
            kind = "velocity",
            typeinfo = np.ndarray,
            description = "Provides target velocity setpoint control",
        )
        self.target_velocity = target_velocity

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        if not self.is_active_at_step(step):
            return channel
        while len(channel) <= self.lain_index:
            channel.append(None)
        channel[self.lain_index] = self.target_velocity.copy()
        return channel

    def get_description(self) -> str:
        return f"Velocity control (target={self.target_velocity})"


class CombinedControl(ControlSource):
    """Combines multiple control types for a single body."""

    def __init__(self, id: str, controls: Dict[str, Any], steps_range: Optional[List[Optional[int]]] = None):
        """
        Initialize combined control.

        Args:
            id: Body ID to control
            controls: Dict mapping control kinds to values, e.g.:
                     {'force': np.array([1.0, 0.0, 0.0]),
                      'torque': np.array([0.0, 0.0, 0.1]),
                      'angular_momentum': np.array([0.0, 0.0, 0.05])}
            steps_range: [start, end] range where control is active
        """
        super(CombinedControl, self).__init__(
            id = id,
            steps_range = steps_range,
            kind = "combined",
            typeinfo = dict,
            description = "Provides multiple control types simultaneously",
        )
        self.controls = controls

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        if not self.is_active_at_step(step):
            return channel
        while len(channel) <= self.lain_index:
            channel.append(None)
        channel[self.lain_index] = self.controls.copy()
        return channel

    def get_description(self) -> str:
        return f"Combined control ({list(self.controls.keys())})"


class ControlSourceFactory:
    """Factory for creating control sources."""

    @staticmethod
    def create_hovering(id: str, steps_range: Optional[List[Optional[int]]] = None) -> ControlSource:
        return HoveringControl(id, steps_range)

    @staticmethod
    def create_linear(id: str, forward_force: float = 0.8, steps_range: Optional[List[Optional[int]]] = None) -> ControlSource:
        return LinearMovementControl(id, forward_force, steps_range)

    @staticmethod
    def create_rotational(id: str, rotation_torque: float = 0.3, steps_range: Optional[List[Optional[int]]] = None) -> ControlSource:
        return RotationalControl(id, rotation_torque, steps_range)

    @staticmethod
    def create_sinusoidal(id: str, steps_range: Optional[List[Optional[int]]] = None) -> ControlSource:
        return SinusoidalControl(id, steps_range=steps_range)

    @staticmethod
    def create_chaotic(id: str, steps_range: Optional[List[Optional[int]]] = None) -> ControlSource:
        return ChaoticControl(id, steps_range=steps_range)

    @staticmethod
    def create_force(id: str, force: np.ndarray = np.array([1.0, 0.0, 0.0]), steps_range: Optional[List[Optional[int]]] = None) -> ControlSource:
        return ForceControl(id, force, steps_range)

    @staticmethod
    def create_torque(id: str, torque: np.ndarray = np.array([0.0, 0.0, 0.1]), steps_range: Optional[List[Optional[int]]] = None) -> ControlSource:
        return TorqueControl(id, torque, steps_range)

    @staticmethod
    def create_angular_momentum(id: str, angular_momentum: np.ndarray = np.array([0.0, 0.0, 0.05]), steps_range: Optional[List[Optional[int]]] = None) -> ControlSource:
        return AngularMomentumControl(id, angular_momentum, steps_range)

    @staticmethod
    def create_position(id: str, target_position: np.ndarray = np.array([2.0, 0.0, 1.0]), steps_range: Optional[List[Optional[int]]] = None) -> ControlSource:
        return PositionControl(id, target_position, steps_range)

    @staticmethod
    def create_velocity(id: str, target_velocity: np.ndarray = np.array([1.0, 0.0, 0.0]), steps_range: Optional[List[Optional[int]]] = None) -> ControlSource:
        return VelocityControl(id, target_velocity, steps_range)

    @staticmethod
    def create_combined(id: str, controls: Dict[str, Any], steps_range: Optional[List[Optional[int]]] = None) -> ControlSource:
        return CombinedControl(id, controls, steps_range)