import numpy as np
import time

# open3d imported conditionally

class HovercraftEnv:
    def __init__(self, viz=True):
        # Constants
        self.mass = 1.0  # kg
        self.I = 0.1  # moment of inertia
        self.dt = 0.01  # time step
        self.gravity = -9.81  # m/s^2

        # Bounds
        self.x_min, self.x_max = -5, 5
        self.y_min, self.y_max = -5, 5
        self.z_min = 0
        self.z_max = 10

        # State: [x, y, z, theta, vx, vy, vz, omega_z]
        self.state = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Forces parameters
        self.lift_mean = 10.0  # mean lifting force
        self.lift_std = 1.0
        self.rot_mean = 0.1  # slight offset
        self.rot_std = 0.5
        self.friction_k = 0.1  # friction proportional to z

        self.viz = viz
        if self.viz:
            import open3d as o3d
            self.o3d = o3d
            # Visualization
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window()

            # Fence: rectangle
            self.fence_lines = []
            points = [
                [self.x_min, self.y_min, 0],
                [self.x_max, self.y_min, 0],
                [self.x_max, self.y_max, 0],
                [self.x_min, self.y_max, 0],
                [self.x_min, self.y_min, 0]  # close
            ]
            lines = [[i, i+1] for i in range(len(points)-1)]
            colors = [[1, 0, 0] for _ in lines]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            self.vis.add_geometry(line_set)

            # Hovercraft: simple cylinder for clear visualization
            self.hovercraft = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3, height=0.2)
            self.hovercraft.compute_vertex_normals()
            self.hovercraft.paint_uniform_color([0, 0.5, 1])  # Bright blue
            self.vis.add_geometry(self.hovercraft)

            # Store original vertices for resetting
            self.hovercraft_original_vertices = np.asarray(self.hovercraft.vertices).copy()

            # Ground
            ground = o3d.geometry.TriangleMesh.create_box(width=10, height=10, depth=0.1)
            ground.translate([-5, -5, -0.1])
            ground.paint_uniform_color([0.5, 0.5, 0.5])
            self.vis.add_geometry(ground)

    def step(self, action):
        # action: [forward_force, rotation_torque]
        forward_force, rotation_torque = action

        x, y, z, theta, vx, vy, vz, omega_z = self.state

        # Forces
        F_lift = np.random.normal(self.lift_mean, self.lift_std)
        T_rot = np.random.normal(self.rot_mean, self.rot_std)

        # Controlled forces
        F_forward_x = forward_force * np.cos(theta)
        F_forward_y = forward_force * np.sin(theta)
        T_control = rotation_torque

        # Friction: proportional to z and velocity
        F_friction_x = -self.friction_k * z * vx
        F_friction_y = -self.friction_k * z * vy
        F_friction_z = -self.friction_k * z * vz

        # Total forces
        Fx = F_forward_x + F_friction_x
        Fy = F_forward_y + F_friction_y
        Fz = F_lift + self.mass * self.gravity + F_friction_z
        Tz = T_rot + T_control

        # Accelerations
        ax = Fx / self.mass
        ay = Fy / self.mass
        az = Fz / self.mass
        alpha_z = Tz / self.I

        # Update velocities
        vx += ax * self.dt
        vy += ay * self.dt
        vz += az * self.dt
        omega_z += alpha_z * self.dt

        # Update positions
        x += vx * self.dt
        y += vy * self.dt
        z += vz * self.dt
        theta += omega_z * self.dt

        # Bounce on borders
        if x < self.x_min:
            x = self.x_min
            vx = -vx
        elif x > self.x_max:
            x = self.x_max
            vx = -vx
        if y < self.y_min:
            y = self.y_min
            vy = -vy
        elif y > self.y_max:
            y = self.y_max
            vy = -vy
        if z < self.z_min:
            z = self.z_min
            vz = -vz
        elif z > self.z_max:
            z = self.z_max
            vz = -vz

        # Update state
        self.state = np.array([x, y, z, theta, vx, vy, vz, omega_z])

        # Update visualization
        self.update_vis()

    def update_vis(self):
        if not self.viz:
            return
        x, y, z, theta, _, _, _, _ = self.state
        # Reset vertices to original
        self.hovercraft.vertices = self.o3d.utility.Vector3dVector(self.hovercraft_original_vertices)
        # Translate
        self.hovercraft.translate([x, y, z], relative=False)
        # Rotate around center
        R = self.o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, theta])
        self.hovercraft.rotate(R, center=[x, y, z])
        # Update visualization
        self.vis.update_geometry(self.hovercraft)
        self.vis.poll_events()
        self.vis.update_renderer()

    def render(self):
        if self.viz:
            self.vis.poll_events()
            self.vis.update_renderer()

    def close(self):
        if self.viz:
            self.vis.destroy_window()
