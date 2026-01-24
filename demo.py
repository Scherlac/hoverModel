import numpy as np
import time
import os
from environment import HovercraftEnv
from physics import HovercraftPhysics
from visualization import NullVisualizer
from control_sources import ControlSourceFactory
from demo_outputs import DemoRunner

def run_tests():
    """Run basic physics tests using the new modular system."""
    print("Running physics tests...")

    runner = DemoRunner()

    # Test hovering
    print("\nTesting hovering...")
    control = ControlSourceFactory.create_hovering()
    runner.run_test(control, steps=50)

    # Test movement
    print("\nTesting movement...")
    control = ControlSourceFactory.create_linear(0.8)
    runner.run_test(control, steps=50)

    # Test rotation
    print("\nTesting rotation...")
    control = ControlSourceFactory.create_rotational(0.3)
    runner.run_test(control, steps=50)

    print("Tests completed.")

def create_demo_video(bouncing=False):
    """Create demonstration video using the new modular system."""
    runner = DemoRunner()

    if bouncing:
        control = ControlSourceFactory.create_chaotic()
        runner.create_bouncing_video(control, "hovercraft_bouncing_demo.mp4", steps=300)
    else:
        control = ControlSourceFactory.create_sinusoidal()
        runner.create_video(control, "hovercraft_demo.mp4", steps=200, fps=25)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "video":
            create_demo_video(bouncing=False)
        elif sys.argv[1] == "bounce":
            create_demo_video(bouncing=True)
        else:
            print("Usage: python demo.py [video|bounce]")
            print("  (no args) - run physics tests")
            print("  video     - create regular demo video")
            print("  bounce    - create boundary bouncing video")
    else:
        run_tests()