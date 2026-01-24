import numpy as np
import time
from main import HovercraftEnv  # Assuming main.py has the HovercraftEnv class

def test_hovering():
    """Test the hovercraft hovering with random forces."""
    print("Testing hovering behavior...")
    env = HovercraftEnv()
    for _ in range(200):  # Shorter for demo
        action = [0.0, 0.0]  # No control forces
        env.step(action)
        time.sleep(0.02)
    env.close()
    print("Hovering test completed.")

def test_forward_movement():
    """Test forward movement and bouncing."""
    print("Testing forward movement and boundary bouncing...")
    env = HovercraftEnv()
    env.state[0] = -4  # Start near left border
    for _ in range(300):
        action = [2.0, 0.0]  # Constant forward force
        env.step(action)
        time.sleep(0.02)
    env.close()
    print("Forward movement test completed.")

def test_rotation():
    """Test rotational control."""
    print("Testing rotational control...")
    env = HovercraftEnv()
    for _ in range(200):
        action = [0.0, 1.0]  # Rotational torque
        env.step(action)
        time.sleep(0.02)
    env.close()
    print("Rotation test completed.")

def test_combined():
    """Test combined forces and physics."""
    print("Testing combined physics simulation...")
    env = HovercraftEnv()
    for i in range(500):
        # Vary actions over time
        forward = 1.0 if i < 250 else -1.0
        rotation = 0.5 * np.sin(i * 0.1)
        action = [forward, rotation]
        env.step(action)
        time.sleep(0.01)
    env.close()
    print("Combined test completed.")

def run_all_tests():
    """Run all demonstration tests."""
    print("Starting hovercraft simulation demonstrations...")
    test_hovering()
    time.sleep(1)  # Pause between tests
    test_forward_movement()
    time.sleep(1)
    test_rotation()
    time.sleep(1)
    test_combined()
    print("All demonstrations completed.")

if __name__ == "__main__":
    run_all_tests()