import numpy as np
import time
import os
import argparse
from environment import HovercraftEnv
from physics import HovercraftPhysics
from visualization import NullVisualizer
from control_sources import ControlSourceFactory
from demo_outputs import DemoRunner

def run_single_demo(control_type: str, steps: int = 50):
    """Run a single demonstration with specified control type."""
    runner = DemoRunner()

    # Map control type strings to factory methods
    control_map = {
        'hover': ControlSourceFactory.create_hovering,
        'linear': lambda: ControlSourceFactory.create_linear(0.8),
        'rotate': lambda: ControlSourceFactory.create_rotational(0.3),
        'sinusoid': ControlSourceFactory.create_sinusoidal,
        'chaotic': ControlSourceFactory.create_chaotic
    }

    if control_type not in control_map:
        print(f"Error: Unknown control type '{control_type}'")
        print("Available types: hover, linear, rotate, sinusoid, chaotic")
        return

    control = control_map[control_type]()
    print(f"\n=== Running {control_type.upper()} Demo ===")
    print(f"Control: {control.get_description()}")
    print(f"Steps: {steps}")
    runner.run_test(control, steps=steps)

def create_video_demo(control_type: str, output_file: str, steps: int = 200, fps: int = 25, bouncing: bool = False):
    """Create a video demonstration."""
    runner = DemoRunner()

    # Map control type strings to factory methods
    control_map = {
        'hover': ControlSourceFactory.create_hovering,
        'linear': lambda: ControlSourceFactory.create_linear(0.8),
        'rotate': lambda: ControlSourceFactory.create_rotational(0.3),
        'sinusoid': ControlSourceFactory.create_sinusoidal,
        'chaotic': ControlSourceFactory.create_chaotic
    }

    if control_type not in control_map:
        print(f"Error: Unknown control type '{control_type}'")
        print("Available types: hover, linear, rotate, sinusoid, chaotic")
        return

    control = control_map[control_type]()
    print(f"\n=== Creating {control_type.upper()} Video ===")
    print(f"Control: {control.get_description()}")
    print(f"Output: {output_file}")
    print(f"Steps: {steps}, FPS: {fps}")

    if bouncing or control_type == 'chaotic':
        runner.create_bouncing_video(control, output_file, steps=steps)
    else:
        runner.create_video(control, output_file, steps=steps, fps=fps)

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

def show_help():
    """Show comprehensive help information."""
    print("Hovercraft Demo System")
    print("======================")
    print()
    print("Available Commands:")
    print("  python demo.py                    - Run all physics tests")
    print("  python demo.py <control>          - Run single demo")
    print("  python demo.py video <control>    - Create video demo")
    print()
    print("Control Types:")
    print("  hover     - Hovering (no control inputs)")
    print("  linear    - Linear forward movement")
    print("  rotate    - Pure rotational movement")
    print("  sinusoid  - Combined sinusoidal motion")
    print("  chaotic   - Chaotic boundary testing")
    print()
    print("Examples:")
    print("  python demo.py hover              - Test hovering")
    print("  python demo.py linear             - Test linear movement")
    print("  python demo.py video sinusoid     - Create sinusoidal video")
    print("  python demo.py video chaotic      - Create boundary bouncing video")
    print("  python demo.py video linear --steps 100 --fps 30 --output my_demo.mp4")
    print()
    print("Options:")
    print("  --steps STEPS     Number of simulation steps (default: 50 for tests, 200 for video)")
    print("  --fps FPS         Video frame rate (default: 25)")
    print("  --output FILE     Output video filename (default: hovercraft_demo.mp4)")
    print("  --help, -h        Show this help message")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Hovercraft demonstration system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                    # Run all tests
  python demo.py hover              # Test hovering
  python demo.py video sinusoid     # Create video
  python demo.py video chaotic      # Create bouncing video
        """
    )

    parser.add_argument('command', nargs='?', help='Command: test type or "video"')
    parser.add_argument('control_type', nargs='?', help='Control type: hover, linear, rotate, sinusoid, chaotic')
    parser.add_argument('--steps', type=int, default=None, help='Number of simulation steps')
    parser.add_argument('--fps', type=int, default=25, help='Video frame rate')
    parser.add_argument('--output', default=None, help='Output video filename')

    args = parser.parse_args()

    # Handle help
    if args.command == 'help' or (not args.command and not args.control_type):
        show_help()
        return

    # Default steps based on command type
    if args.steps is None:
        args.steps = 50 if args.command != 'video' else 200

    # Default output filename
    if args.output is None and args.command == 'video':
        args.output = 'hovercraft_demo.mp4'

    # Run all tests
    if not args.command:
        run_tests()
        return

    # Single demo
    if args.command in ['hover', 'linear', 'rotate', 'sinusoid', 'chaotic']:
        run_single_demo(args.command, steps=args.steps)
        return

    # Video demo
    if args.command == 'video':
        if not args.control_type:
            print("Error: video command requires a control type")
            print("Usage: python demo.py video <control_type>")
            return

        bouncing = args.control_type == 'chaotic'
        create_video_demo(args.control_type, args.output, steps=args.steps, fps=args.fps, bouncing=bouncing)
        return

    # Unknown command
    print(f"Error: Unknown command '{args.command}'")
    show_help()

if __name__ == "__main__":
    main()