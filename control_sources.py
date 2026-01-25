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
        
        # self.index = LainInfo.index
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
            kind: Optional[str] = "x-force,z-torque",
            typeinfo: Optional[type] = Tuple[float, float],
            description: Optional[str] = "Provides control signals for body movement with force and torque",
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
            kind: Kind of the control source
            description: Human-readable description
        """

        self.lain_index = self.lain_info.index

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
            forward_force: float = 0.8):
        
        super(LinearMovementControl, self).__init__(
            id = id
        )
        self.forward_force = forward_force

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        while len(channel) <= self.lain_index:
            channel.append(None)
        channel[self.lain_index] = (self.forward_force, 0.0)
        return channel

    def get_description(self) -> str:
        return f"Linear movement (forward_force={self.forward_force})"


class RotationalControl(ControlSource):
    """Generates pure rotational movement."""

    def __init__(self, id: str, rotation_torque: float = 0.3):
        super(RotationalControl, self).__init__(id=id)
        self.rotation_torque = rotation_torque

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        while len(channel) <= self.lain_index:
            channel.append(None)
        channel[self.lain_index] = (0.0, self.rotation_torque)
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
                 rotation_freq: float = 0.1):
        super(SinusoidalControl, self).__init__(
            id = id
        )
        self.forward_amp = forward_amplitude
        self.rotation_amp = rotation_amplitude
        self.forward_freq = forward_freq
        self.rotation_freq = rotation_freq

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        while len(channel) <= self.lain_index:
            channel.append(None)
        forward = self.forward_amp * np.sin(step * self.forward_freq)
        rotation = self.rotation_amp * np.cos(step * self.rotation_freq)
        channel[self.lain_index] = (forward, rotation)
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
                 rotation_freq: float = 0.18):
        super(ChaoticControl, self).__init__(
            id = id
        )
        self.forward_amp = forward_amplitude
        self.rotation_amp = rotation_amplitude
        self.forward_freq = forward_freq
        self.rotation_freq = rotation_freq

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        while len(channel) <= self.lain_index:
            channel.append(None)
        forward = self.forward_amp * np.sin(step * self.forward_freq)
        rotation = self.rotation_amp * np.cos(step * self.rotation_freq)
        channel[self.lain_index] = (forward, rotation)
        return channel

    def get_description(self) -> str:
        return f"Chaotic boundary movement (amp_f={self.forward_amp}, amp_r={self.rotation_amp})"

class HoveringControl(ControlSource):
    """Generates zero control signals for hovering tests."""

    def __init__(self, id: str):
        super(HoveringControl, self).__init__(
            id = id,
            kind = "lifting thrust",
            typeinfo = float,
            description = "Provides control signals for ground effects (hovering)",
        )

    def get_control(self, channel: SignalChannel, step: int) -> SignalChannel:
        while len(channel) <= self.lain_index:
            channel.append(None)
        channel[self.lain_index] = 0.0
        return channel

    def get_description(self) -> str:
        return "Hovering (no control inputs)"


class ControlSourceFactory:
    """Factory for creating control sources."""

    @staticmethod
    def create_hovering(id: str) -> ControlSource:
        return HoveringControl(id)

    @staticmethod
    def create_linear(id: str, forward_force: float = 0.8) -> ControlSource:
        return LinearMovementControl(id, forward_force)

    @staticmethod
    def create_rotational(id: str, rotation_torque: float = 0.3) -> ControlSource:
        return RotationalControl(id, rotation_torque)

    @staticmethod
    def create_sinusoidal(id: str) -> ControlSource:
        return SinusoidalControl(id)

    @staticmethod
    def create_chaotic(id: str) -> ControlSource:
        return ChaoticControl(id)