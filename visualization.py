from abc import ABC, abstractmethod
import numpy as np
from typing import (
    Tuple,
)

from state import BodyState
from components import Visualizer, VisualizationOutput

class Open3DVisualizationOutput(VisualizationOutput):
    """Visualization output handler for Open3D visualizer."""

    def __init__(self, visualizer: "Open3DVisualizer"):
        super(Open3DVisualizationOutput, self).__init__(visualizer)
        # camera params
        self.camera_position = np.array([5, 5, 5])
        self.camera_look_at = np.array([0, 0, 1])
        self.up_vector = np.array([0, 0, 1])
        self.zoom = 0.8

        self._set_camera()


    def _set_camera(self) -> None:
        ctr = self.visualizer.vis.get_view_control()
        ctr.set_lookat(self.camera_look_at)
        ctr.set_front(self.camera_position)
        ctr.set_up(self.up_vector)  # Z-up
        ctr.set_zoom(self.zoom)

    def set_camera(self, position: Tuple[float, float, float], look_at: Tuple[float, float, float]) -> None:
        self.camera_position = np.array(position)
        self.camera_look_at = np.array(look_at)
        self._set_camera()

    def set_zoom(self, zoom: float) -> None:
        self.zoom = zoom
        self._set_camera()

    def render(self) -> None:
        self.visualizer.render()

    def capture_frame(self, filename: str) -> None:
        self.visualizer.capture_frame(filename)

class Open3DVisualizer(Visualizer):
    """Open3D-based 3D visualization."""

    def __init__(self, env):
        super(Open3DVisualizer, self).__init__(env)
        try:
            import open3d as o3d
            self.o3d = o3d
        except ImportError:
            raise ImportError("Open3D required for 3D visualization")

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="Hovercraft Simulation", width=1024, height=768, visible=True)

        # Setup environment geometry
        self._setup_environment(self.env.bounds)

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

    def update(self, state: BodyState):
        """Update hovercraft position and orientation."""

        # Reset and transform hovercraft
        self.hovercraft.vertices = self.o3d.utility.Vector3dVector(self.hovercraft_original_vertices)
        self.hovercraft.translate(state.r, relative=False)

        R = self.o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, state.theta])
        self.hovercraft.rotate(R, center=state.r)

        self.vis.update_geometry(self.hovercraft)
        self.vis.poll_events()
        self.vis.update_renderer()

    def get_visualization_output(self) -> VisualizationOutput:
        return Open3DVisualizationOutput(self)


    def close(self):
        """Clean up Open3D resources."""
        self.vis.destroy_window()


