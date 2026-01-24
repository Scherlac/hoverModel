from abc import ABC, abstractmethod
import numpy as np

class Visualizer(ABC):
    """Abstract base class for visualization backends."""

    @abstractmethod
    def __init__(self, bounds: dict):
        """Initialize visualizer with environment bounds."""
        pass

    @abstractmethod
    def update(self, state: np.ndarray):
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

class Open3DVisualizer(Visualizer):
    """Open3D-based 3D visualization."""

    def __init__(self, bounds: dict):
        try:
            import open3d as o3d
            self.o3d = o3d
        except ImportError:
            raise ImportError("Open3D required for 3D visualization")

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Hovercraft Simulation", width=1024, height=768, visible=True)

        # Setup environment geometry
        self._setup_environment(bounds)

        # Hovercraft geometry
        self.hovercraft = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3, height=0.2)
        self.hovercraft.compute_vertex_normals()
        self.hovercraft.paint_uniform_color([0, 0.5, 1])
        self.vis.add_geometry(self.hovercraft)
        self.hovercraft_original_vertices = np.asarray(self.hovercraft.vertices).copy()

        # Set default camera view
        self._setup_camera()

    def _setup_environment(self, bounds):
        """Setup static environment geometry."""
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = bounds.values()

        # Boundary fence
        points = [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_min]  # close
        ]
        lines = [[i, i+1] for i in range(len(points)-1)]
        colors = [[1, 0, 0] for _ in lines]
        line_set = self.o3d.geometry.LineSet()
        line_set.points = self.o3d.utility.Vector3dVector(points)
        line_set.lines = self.o3d.utility.Vector2iVector(lines)
        line_set.colors = self.o3d.utility.Vector3dVector(colors)
        self.vis.add_geometry(line_set)

        # Ground plane
        ground = self.o3d.geometry.TriangleMesh.create_box(
            width=x_max-x_min, height=y_max-y_min, depth=0.1
        )
        ground.translate([x_min, y_min, z_min-0.1])
        ground.paint_uniform_color([0.5, 0.5, 0.5])
        self.vis.add_geometry(ground)

    def _setup_camera(self):
        """Setup default camera view."""
        ctr = self.vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0.5, 0.5, -0.7])  # Angled view
        ctr.set_lookat([0, 0, 1])        # Look at center
        ctr.set_up([0, 0, 1])            # Up direction

    def update(self, state: np.ndarray):
        """Update hovercraft position and orientation."""
        x, y, z, theta, _, _, _, _ = state

        # Reset and transform hovercraft
        self.hovercraft.vertices = self.o3d.utility.Vector3dVector(self.hovercraft_original_vertices)
        self.hovercraft.translate([x, y, z], relative=False)

        R = self.o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, theta])
        self.hovercraft.rotate(R, center=[x, y, z])

        self.vis.update_geometry(self.hovercraft)
        self.vis.poll_events()
        self.vis.update_renderer()

    def render(self):
        """Render current frame."""
        self.vis.poll_events()
        self.vis.update_renderer()
        # Small delay to allow window to update
        import time
        time.sleep(0.01)

    def close(self):
        """Clean up Open3D resources."""
        self.vis.destroy_window()

    def capture_frame(self, filename: str):
        """Capture current frame to image file."""
        self.vis.capture_screen_image(filename)

class NullVisualizer(Visualizer):
    """No-op visualizer for headless operation."""

    def __init__(self, bounds: dict):
        pass

    def update(self, state: np.ndarray):
        pass

    def render(self):
        pass

    def close(self):
        pass

    def capture_frame(self, filename: str):
        pass