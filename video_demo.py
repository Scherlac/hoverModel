import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from main import HovercraftEnv
import time

def create_demo_video():
    """Create an MKV video demonstrating the hovercraft simulation."""
    print("Creating demo video...")

    # Run simulation and collect data
    env = HovercraftEnv(viz=False)
    positions = []
    orientations = []

    # Run for 500 steps with varying actions
    for i in range(500):
        # Vary actions over time
        forward = 1.0 if i < 250 else -1.0
        rotation = 0.5 * np.sin(i * 0.1)
        action = [forward, rotation]
        env.step(action)

        x, y, z, theta, _, _, _, _ = env.state
        positions.append((x, y, z))
        orientations.append(theta)

        if i % 100 == 0:
            print(f"Simulating step {i}/500")

    env.close()

    # Create animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Top-down view
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    ax1.set_title('Top-Down View')
    ax1.set_xlabel('X position')
    ax1.set_ylabel('Y position')
    ax1.grid(True)

    # Draw fence
    fence_x = [-5, 5, 5, -5, -5]
    fence_y = [-5, -5, 5, 5, -5]
    ax1.plot(fence_x, fence_y, 'r-', linewidth=2, label='Fence')

    # Hovercraft representation (triangle)
    hovercraft_patch = ax1.scatter([], [], c='blue', s=100, marker='^', label='Hovercraft')

    # Height plot
    ax2.set_xlim(0, 500)
    ax2.set_ylim(0, 12)
    ax2.set_title('Height Over Time')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Z position')
    ax2.grid(True)

    height_line, = ax2.plot([], [], 'g-', linewidth=2, label='Height')

    ax1.legend()
    ax2.legend()

    def animate(frame):
        # Update hovercraft position
        x, y, z = positions[frame]
        theta = orientations[frame]

        # Rotate the triangle based on orientation
        hovercraft_patch.set_offsets([(x, y)])

        # Update height plot
        height_line.set_data(range(frame+1), [pos[2] for pos in positions[:frame+1]])

        return hovercraft_patch, height_line

    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(positions),
                                   interval=50, blit=True, repeat=False)

    # Save as video
    # Note: Requires ffmpeg installed for video saving
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Hovercraft Sim'), bitrate=1800)
        anim.save('hovercraft_demo.mkv', writer=writer)
        print("Video saved as hovercraft_demo.mkv")
    except (RuntimeError, KeyError):
        print("FFmpeg not available. Saving as GIF instead...")
        anim.save('hovercraft_demo.gif', writer='pillow', fps=20)
        print("Animation saved as hovercraft_demo.gif")

    plt.close(fig)
    print("Demo video creation completed.")

if __name__ == "__main__":
    create_demo_video()