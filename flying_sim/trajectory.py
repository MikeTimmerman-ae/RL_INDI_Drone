import configs.config
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


class Trajectory:

    def __init__(self, config):
        self.min_dist = config.traj_config.min_dist
        self.max_dist = config.traj_config.max_dist
        self.v_des = config.traj_config.v_des
        self.traj_time = config.traj_config.tf

        self.config = config
        self.trajectory_xyz()

    def position_ref(self, t: float) -> np.ndarray:
        return np.array([
            self.traj_x(t),
            self.traj_y(t),
            self.traj_z(t)
        ])

    def velocity_ref(self, t: float) -> np.ndarray:
        return np.array([
            self.traj_dx(t),
            self.traj_dy(t),
            self.traj_dz(t)
        ])

    def acceleration_ref(self, t: float) -> np.ndarray:
        return np.array([
            self.traj_ddx(t),
            self.traj_ddy(t),
            self.traj_ddz(t)
        ])

    def set_seed(self, seed):
        np.random.seed(seed)

    def hover_trajectory(self):
        self.time = [0, self.traj_time]
        self.waypoints = np.array([self.config.traj_config.init_pos,
                                  self.config.traj_config.init_pos])

        # Fit cubic spline waypoints
        self.traj_x = CubicSpline(self.time, self.waypoints[:, 0], bc_type="clamped")
        self.traj_y = CubicSpline(self.time, self.waypoints[:, 1], bc_type="clamped")
        self.traj_z = CubicSpline(self.time, self.waypoints[:, 2], bc_type="clamped")

        self.traj_dx = self.traj_x.derivative(nu=1)
        self.traj_dy = self.traj_y.derivative(nu=1)
        self.traj_dz = self.traj_z.derivative(nu=1)

        self.traj_ddx = self.traj_x.derivative(nu=2)
        self.traj_ddy = self.traj_y.derivative(nu=2)
        self.traj_ddz = self.traj_z.derivative(nu=2)

    def random_spline_trajectory(self):
        self.time = [0]
        self.waypoints = np.array([self.config.traj_config.init_pos])
        # Generate random waypoints at a min distance of 1m
        while self.time[-1] < self.traj_time:
            x = np.random.uniform(low=0.0, high=100.0, size=None)
            y = np.random.uniform(low=0.0, high=100.0, size=None)
            z = np.random.uniform(low=-30.0, high=0.0, size=None)

            new_waypoint = np.array([x, y, z])

            if self.min_dist < np.linalg.norm(self.waypoints[-1] - new_waypoint) < self.max_dist:
                self.time.append(self.time[-1] + np.linalg.norm(self.waypoints[-1, :] - new_waypoint) / self.v_des)
                self.waypoints = np.vstack((self.waypoints, new_waypoint))

        # Fit cubic spline waypoints
        self.traj_x = CubicSpline(self.time, self.waypoints[:, 0], bc_type="clamped")
        self.traj_y = CubicSpline(self.time, self.waypoints[:, 1], bc_type="clamped")
        self.traj_z = CubicSpline(self.time, self.waypoints[:, 2], bc_type="clamped")

        self.traj_dx = self.traj_x.derivative(nu=1)
        self.traj_dy = self.traj_y.derivative(nu=1)
        self.traj_dz = self.traj_z.derivative(nu=1)

        self.traj_ddx = self.traj_x.derivative(nu=2)
        self.traj_ddy = self.traj_y.derivative(nu=2)
        self.traj_ddz = self.traj_z.derivative(nu=2)

    def trajectory_xz(self):
        self.time = [0]
        self.waypoints = np.array([[0, 0, 0], [2, 0, 3], [6, 0, 4], [10, 0, 4], [13, 0, 3],
                                   [15, 0, 0], [13, 0, -3], [10, 0, -4], [6, 0, -4], [2, 0, -3],
                                   [0, 0, 0], [-2, 0, 3], [-6, 0, 4], [-10, 0, 4], [-13, 0, 3],
                                   [-15, 0, 0], [-13, 0, -3], [-10, 0, -4], [-6, 0, -4], [-2, 0, -3],
                                   [0, 0, 0], [2, 0, 3], [6, 0, 4], [10, 0, 4]])
        for i, waypoint in enumerate(self.waypoints[1:]):
            dist = np.linalg.norm(waypoint - self.waypoints[i-1])
            self.time.append(self.time[-1] + dist / self.v_des)

        # Fit cubic spline waypoints
        self.traj_x = CubicSpline(self.time, self.waypoints[:, 0], bc_type="clamped")
        self.traj_y = CubicSpline(self.time, self.waypoints[:, 1], bc_type="clamped")
        self.traj_z = CubicSpline(self.time, self.waypoints[:, 2], bc_type="clamped")

        self.traj_dx = self.traj_x.derivative(nu=1)
        self.traj_dy = self.traj_y.derivative(nu=1)
        self.traj_dz = self.traj_z.derivative(nu=1)

        self.traj_ddx = self.traj_x.derivative(nu=2)
        self.traj_ddy = self.traj_y.derivative(nu=2)
        self.traj_ddz = self.traj_z.derivative(nu=2)

    def trajectory_xy(self):
        self.time = [0]
        self.waypoints = np.array([[0, 0, 0], [2, 3, 0], [6, 4, 0], [10, 4, 0], [13, 3, 0],
                                   [15, 0, 0], [13, -3, 0], [10, -4, 0], [6, -4, 0], [2, -3, 0],
                                   [0, 0, 0], [-2, 3, 0], [-6, 4, 0], [-10, 4, 0], [-13, 3, 0],
                                   [-15, 0, 0], [-13, -3, 0], [-10, -4, 0], [-6, -4, 0], [-2, -3, 0],
                                   [0, 0, 0], [2, 3, 0], [6, 4, 0], [10, 4, 0]])
        for i, waypoint in enumerate(self.waypoints[1:]):
            dist = np.linalg.norm(waypoint - self.waypoints[i-1])
            self.time.append(self.time[-1] + dist / self.v_des)

        # Fit cubic spline waypoints
        self.traj_x = CubicSpline(self.time, self.waypoints[:, 0], bc_type="clamped")
        self.traj_y = CubicSpline(self.time, self.waypoints[:, 1], bc_type="clamped")
        self.traj_z = CubicSpline(self.time, self.waypoints[:, 2], bc_type="clamped")

        self.traj_dx = self.traj_x.derivative(nu=1)
        self.traj_dy = self.traj_y.derivative(nu=1)
        self.traj_dz = self.traj_z.derivative(nu=1)

        self.traj_ddx = self.traj_x.derivative(nu=2)
        self.traj_ddy = self.traj_y.derivative(nu=2)
        self.traj_ddz = self.traj_z.derivative(nu=2)

    def trajectory_xyz(self):
        self.time = [0]
        self.waypoints = np.array([[0, 0, 0], [2, 3, 0.5], [6, 4, 1], [10, 4, 1], [13, 3, 0.5],
                                   [15, 0, 0], [13, -3, -0.5], [10, -4, -1], [6, -4, -1], [2, -3, -0.5],
                                   [0, 0, 0], [-2, 3, 0.5], [-6, 4, 1], [-10, 4, 1], [-13, 3, 0.5],
                                   [-15, 0, 0], [-13, -3, -0.5], [-10, -4, -1], [-6, -4, -1], [-2, -3, -0.5],
                                   [0, 0, 0], [2, 3, 0.5], [6, 4, 1], [10, 4, 1]])
        for i, waypoint in enumerate(self.waypoints[1:]):
            dist = np.linalg.norm(waypoint - self.waypoints[i-1])
            self.time.append(self.time[-1] + dist / self.v_des)

        # Fit cubic spline waypoints
        self.traj_x = CubicSpline(self.time, self.waypoints[:, 0], bc_type="clamped")
        self.traj_y = CubicSpline(self.time, self.waypoints[:, 1], bc_type="clamped")
        self.traj_z = CubicSpline(self.time, self.waypoints[:, 2], bc_type="clamped")

        self.traj_dx = self.traj_x.derivative(nu=1)
        self.traj_dy = self.traj_y.derivative(nu=1)
        self.traj_dz = self.traj_z.derivative(nu=1)

        self.traj_ddx = self.traj_x.derivative(nu=2)
        self.traj_ddy = self.traj_y.derivative(nu=2)
        self.traj_ddz = self.traj_z.derivative(nu=2)

    def plot_trajectory(self):
        t = np.arange(0, self.time[-1], 0.01)
        positions = np.array([0, 0, 0])

        for time in t:
            positions = np.vstack((positions, self.position_ref(time)))

        fix, axs = plt.subplots(3, 1)
        axs[0].plot(t, positions[:-1, 0])
        axs[0].scatter(self.time, self.waypoints[:, 0])

        axs[1].plot(t, positions[:-1, 1])
        axs[1].scatter(self.time, self.waypoints[:, 1])

        axs[2].plot(t, positions[:-1, 2])
        axs[2].scatter(self.time, self.waypoints[:, 2])

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
        ax.scatter(self.waypoints[:, 0], self.waypoints[:, 1], self.waypoints[:, 2])

        plt.show()


def test_trajectory():
    config = configs.config.Config()
    trajectory = Trajectory(config)
    trajectory.random_spline_trajectory()
    t = np.arange(0, trajectory.time[-1], 0.01)
    positions = np.array([0, 0, 0])
    velocities = np.array([0, 0, 0])
    accelerations = np.array([0, 0, 0])

    for time in t:
        positions = np.vstack((positions, trajectory.position_ref(time)))
        velocities = np.vstack((velocities, trajectory.velocity_ref(time)))
        accelerations = np.vstack((accelerations, trajectory.acceleration_ref(time)))

    fix, axs = plt.subplots(3, 3)
    axs[0, 0].plot(t, positions[:-1, 0])
    axs[0, 0].scatter(trajectory.time, trajectory.waypoints[:, 0])
    axs[0, 0].set_ylabel(r"$p_x$-position [m]")
    axs[0, 0].grid()

    axs[1, 0].plot(t, positions[:-1, 1])
    axs[1, 0].scatter(trajectory.time, trajectory.waypoints[:, 1])
    axs[1, 0].set_ylabel(r"$p_y$-position [m]")
    axs[1, 0].grid()

    axs[2, 0].plot(t, positions[:-1, 2])
    axs[2, 0].scatter(trajectory.time, trajectory.waypoints[:, 2])
    axs[2, 0].set_ylabel(r"$p_z$-position [m]")
    axs[2, 0].set_xlabel("Time [s]")
    axs[2, 0].grid()


    axs[0, 1].plot(t, velocities[:-1, 0])
    axs[0, 1].set_ylabel(r"$v_x$-velocity $[m/s]$")
    axs[0, 1].grid()

    axs[1, 1].plot(t, velocities[:-1, 1])
    axs[1, 1].set_ylabel(r"$v_y$-velocity $[m/s]$")
    axs[1, 1].grid()

    axs[2, 1].plot(t, velocities[:-1, 2])
    axs[2, 1].set_ylabel(r"$v_z$-velocity $[m/s]$")
    axs[2, 1].set_xlabel("Time [s]")
    axs[2, 1].grid()


    axs[0, 2].plot(t, accelerations[:-1, 0])
    axs[0, 2].set_ylabel(r"$a_x$-acceleration $[m/s^2]$")
    axs[0, 2].grid()

    axs[1, 2].plot(t, accelerations[:-1, 1])
    axs[1, 2].set_ylabel(r"$a_y$-acceleration $[m/s^2]$")
    axs[1, 2].grid()

    axs[2, 2].plot(t, accelerations[:-1, 2])
    axs[2, 2].set_ylabel(r"$a_z$-acceleration $[m/s^2]$")
    axs[2, 2].set_xlabel("Time [s]")
    axs[2, 2].grid()

    plt.tight_layout()

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
    ax.scatter(trajectory.waypoints[:, 0], trajectory.waypoints[:, 1], trajectory.waypoints[:, 2])
    ax.set_xlabel(r"$p_x$-position [m]")
    ax.set_ylabel(r"$p_y$-position [m]")
    ax.set_zlabel(r"$p_z$-position [m]")

    plt.show()

