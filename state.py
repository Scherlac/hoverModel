"""
Body state with vector representations.
"""

from typing import Dict, Any
import numpy as np
import json


class BodyState:
    """Physical body state: r (position), v (velocity), theta (orientation), omega (angular velocity)."""

    def __init__(self, r=None, v=None, theta=0.0, omega=0.0):
        self.r = r if r is not None else np.zeros(3)  # position vector
        self.v = v if v is not None else np.zeros(3)  # velocity vector
        self.theta = theta                            # orientation angle
        self.omega = omega                            # angular velocity

    @classmethod
    def default(cls):
        return cls(r=np.array([0.0, 0.0, 1.0]))

    def __array__(self):
        return np.concatenate([self.r, [self.theta], self.v, [self.omega]])

    def copy(self):
        return BodyState(self.r.copy(), self.v.copy(), self.theta, self.omega)

    def reset(self):
        """Reset state to default values."""
        default = self.default()
        self.r = default.r.copy()
        self.v = default.v.copy()
        self.theta = default.theta
        self.omega = default.omega

    def to_dict(self):
        return {
            'x': float(self.r[0]), 'y': float(self.r[1]), 'z': float(self.r[2]),
            'vx': float(self.v[0]), 'vy': float(self.v[1]), 'vz': float(self.v[2]),
            'theta': float(self.theta), 'omega': float(self.omega)
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            r=np.array([d.get('x', 0.0), d.get('y', 0.0), d.get('z', 1.0)]),
            v=np.array([d.get('vx', 0.0), d.get('vy', 0.0), d.get('vz', 0.0)]),
            theta=d.get('theta', 0.0),
            omega=d.get('omega', 0.0)
        )

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    def __repr__(self):
        return f"BodyState(r={self.r}, v={self.v}, theta={self.theta:.3f}, omega={self.omega:.3f})"