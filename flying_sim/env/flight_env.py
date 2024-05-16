import numpy as np
import random
import torch

import gymnasium as gym
import matplotlib
from gymnasium import spaces
import matplotlib.pyplot as plt

from configs.config import Config
from flying_sim.drone import Drone
from flying_sim.controllers import ControlAllocation, AttitudeController, PositionController
from flying_sim.trajectory import Trajectory

matplotlib.use('TkAgg')


class FlightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, **kwargs):
        self.config: Config = Config()

        # Environment configuration
        self.num_envs = self.config.training.num_processes
        self.n_steps = self.config.ppo.num_steps

        self.action_space = spaces.Box(low=-5, high=5, shape=(3,), dtype=np.float32)    # Normalized PD gains
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64)

        self.train = kwargs['train']

        # Log environment variables
        self.error = [0.]
        self.reach_count = 0
        self.deviation_count = 0
        self.timeout_count = 0
        self.is_success = False

        self.time = []
        self.configure(self.config)

    def configure(self, config: Config):
        print("[INFO] Setting up Drone")
        self.drone: Drone = Drone(config)
        self.thrust_des = -self.drone.m * self.drone.g
        self.control_moment = np.zeros((3,))
        print("[INFO] Setting up Controller")
        self.pos_controller = PositionController(self.drone)
        self.att_controller = AttitudeController(self.drone)
        self.control_allocation = ControlAllocation(self.drone)
        print(f"[INFO] Setting up Trajectory")
        self.trajectory = Trajectory(config)
        if not self.train:
            self.trajectory.random_spline_trajectory()
        if self.config.env_config.seed is not None:
            self.set_seed(self.config.env_config.seed)

        self.states = self.drone.state
        self.aux_states = np.array([0, 0, 0, 0, 0, 0])
        self.inputs = np.array([0, 0, 0, 0])
        self.angular_vel_ref = np.array([0, 0, 0])
        self.attitude_ref = np.array([[0, 0, 0]])
        self.acceleration_des = np.array([0, 0, 0])
        self.acceleration_ref = np.array([[0, 0, 0]])
        self.velocity_ref = np.array([[0, 0, 0]])
        self.position_ref = np.array([[0, 0, 0]])

        self.time.append(config.env_config.t0)
        self.dt = config.env_config.dt
        print("[INFO] Finished setting up Environement")

    def set_seed(self, seed):
        self.trajectory.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _get_obs(self):
        # Observe the drone's velocity in body reference frame
        # np.hstack((self.drone.velocity_e, self.drone.lin_acc, self.drone.attitude))
        return np.hstack(((self.position_ref[-1, :] - self.drone.position),
                          (self.velocity_ref[-1, :] - self.drone.velocity_e),
                          (self.attitude_ref[-1, :] - self.drone.attitude)))

    def _get_info(self):
        return {'cur_state': self.drone.state,
                'cur_time': self.time[-1],
                'reference': self.trajectory.position_ref(self.time[-1]),
                'states': self.states,
                'time': self.time,
                'RMSE_pos': np.sqrt((np.array(self.error) ** 2).mean()),
                'pos_ref': self.position_ref,
                'trajectory': self.trajectory,
                'reach_count': self.reach_count,
                'deviation_count': self.deviation_count,
                'timeout_count': self.timeout_count,
                'is_success': self.is_success,
                'train': self.train,
                'log_interval': self.config.training.log_interval,
                'num_steps': self.config.ppo.num_steps
                }

    def reset(self, seed=None, options=None):
        observation = self._get_obs()

        info = self._get_info()

        # Reset environment
        super().reset(seed=seed)        # seed self.np_random

        self.drone.reset()
        self.trajectory.random_spline_trajectory()

        self.thrust_des = -self.drone.m * self.drone.g
        self.control_moment = np.array([0, 0, 0])

        self.states = self.drone.state
        self.aux_states = np.array([0, 0, 0, 0, 0, 0])
        self.inputs = np.array([0, 0, 0, 0])
        self.angular_vel_ref = np.array([0, 0, 0])
        self.attitude_ref = np.array([[0, 0, 0]])
        self.acceleration_des = np.array([0, 0, 0])
        self.acceleration_ref = np.array([[0, 0, 0]])
        self.velocity_ref = np.array([[0, 0, 0]])
        self.position_ref = np.array([[0, 0, 0]])

        self.time = [self.config.env_config.t0]
        self.error = [0.]

        self.is_success = False

        return observation, info

    def step(self, action):
        # Reference trajectory
        pos_ref = self.trajectory.position_ref(self.time[-1])
        vel_ref = self.trajectory.velocity_ref(self.time[-1])
        acc_ref = self.trajectory.acceleration_ref(self.time[-1])

        # Position controller
        des_lin_acc = self.pos_controller.get_desired_lin_acc(self.drone.position, self.drone.velocity_e, self.drone.lin_acc, pos_ref, vel_ref, acc_ref)
        att_des, self.thrust_des = self.pos_controller.get_desired_attitude(self.drone.attitude, self.thrust_des, self.drone.lin_acc, des_lin_acc, action)
        yaw_des = 0
        des_attitude = np.array([att_des[0], att_des[1], yaw_des])  # Desired attitude

        # Attitude controller
        des_angular_vel = self.att_controller.get_des_angular_vel(self.drone.attitude, des_attitude)
        self.control_moment = self.att_controller.get_control_moment(self.control_moment, self.drone.angular_velocity, des_angular_vel)
        control_input = self.control_allocation.get_control_input(self.control_moment, self.thrust_des)

        # Time step for drone
        self.drone.step(control_input)
        self.time.append(self.time[-1] + self.dt)

        # Log states
        self.states = np.vstack((self.states, self.drone.state))
        self.aux_states = np.vstack((self.aux_states, np.hstack((self.drone.velocity_e, self.drone.lin_acc))))
        self.inputs = np.vstack((self.inputs, control_input))
        self.angular_vel_ref = np.vstack((self.angular_vel_ref, des_angular_vel))
        self.attitude_ref = np.vstack((self.attitude_ref, des_attitude))
        self.acceleration_des = np.vstack((self.acceleration_des, des_lin_acc))
        self.acceleration_ref = np.vstack((self.acceleration_ref, acc_ref))
        self.velocity_ref = np.vstack((self.velocity_ref, vel_ref))
        self.position_ref = np.vstack((self.position_ref, pos_ref))

        # Check for terminal state
        pos_deviation = np.linalg.norm(self.drone.position - pos_ref)
        self.error.append(pos_deviation)

        reached = np.linalg.norm(self.trajectory.waypoints[-1, :] - self.drone.position) < 0.5 and \
                  self.time[-1] >= self.trajectory.traj_time
        deviated = pos_deviation >= 2
        terminated = reached or deviated or self.time[-1] > self.trajectory.traj_time * 2

        reward = 0
        if reached:
            # Drone reached its goal
            self.reach_count += 1
            self.is_success = True
            # print("Goal reached with mean error: {}".format(self.error / len(self.time)))
            print("Goal reached with mean squared error: {}".format(np.sqrt((np.array(self.error) ** 2).mean())))
        elif deviated:
            # reward += -10
            # Drone became unstable and deviated from path
            self.deviation_count += 1
            print("Drone deviated with: {}".format(pos_deviation))
        elif terminated:
            # Drone did not reach goal in time
            self.timeout_count += 1
            print("Simulation terminated!")

        reward += -np.linalg.norm(self.drone.lin_acc - des_lin_acc) * self.dt
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def plot(self):
        drone = self.drone
        trajectory = self.trajectory
        t = self.time

        # Plot states
        fig1, ax = plt.subplots(3, 2)

        ax[0, 0].plot(t, self.states[:, 0] * 180 / np.pi, label="state")
        ax[0, 0].plot(t, self.attitude_ref[:, 0] * 180 / np.pi, label="ref")
        ax[0, 0].set_ylabel("Roll Angle [deg]")
        ax[0, 0].grid()

        ax[1, 0].plot(t, self.states[:, 1] * 180 / np.pi, t, self.attitude_ref[:, 1] * 180 / np.pi)
        ax[1, 0].set_ylabel("Pitch Angle [deg]")
        ax[1, 0].grid()

        ax[2, 0].plot(t, self.states[:, 2] * 180 / np.pi, t, self.attitude_ref[:, 2] * 180 / np.pi)
        ax[2, 0].set_ylabel("Yaw Angle [deg]")
        ax[2, 0].grid()

        ax[0, 1].plot(t, self.states[:, 3] * 180 / np.pi, t, self.angular_vel_ref[:, 0] * 180 / np.pi)
        ax[0, 1].set_ylabel("Roll Rate [deg/s]")
        ax[0, 1].grid()

        ax[1, 1].plot(t, self.states[:, 4] * 180 / np.pi, t, self.angular_vel_ref[:, 1] * 180 / np.pi)
        ax[1, 1].set_ylabel("Pitch Rate [deg/s]")
        ax[1, 1].grid()

        ax[2, 1].plot(t, self.states[:, 5] * 180 / np.pi, t, self.angular_vel_ref[:, 2] * 180 / np.pi)
        ax[2, 1].set_ylabel("Yaw Rate [deg/s]")
        ax[2, 1].grid()

        ax[0, 0].legend()
        plt.tight_layout()

        fig2, ax = plt.subplots(3, 3)
        ax[0, 0].plot(t, self.states[:, 6], t, self.position_ref[:, 0])
        ax[0, 0].scatter(trajectory.time, trajectory.waypoints[:, 0])
        ax[0, 0].set_ylabel("X-position [m]")
        ax[0, 0].grid()

        ax[1, 0].plot(t, self.states[:, 7], t, self.position_ref[:, 1])
        ax[1, 0].scatter(trajectory.time, trajectory.waypoints[:, 1])
        ax[1, 0].set_ylabel("Y-position [m]")
        ax[1, 0].grid()

        ax[2, 0].plot(t, self.states[:, 8], t, self.position_ref[:, 2])
        ax[2, 0].scatter(trajectory.time, trajectory.waypoints[:, 2])
        ax[2, 0].set_ylabel("Z-position [m]")
        ax[2, 0].grid()

        ax[0, 1].plot(t, self.aux_states[:, 0], t, self.velocity_ref[:, 0])
        ax[0, 1].set_ylabel("X-velocity [m/s]")
        ax[0, 1].grid()

        ax[1, 1].plot(t, self.aux_states[:, 1], t, self.velocity_ref[:, 1])
        ax[1, 1].set_ylabel("Y-velocity [m/s]")
        ax[1, 1].grid()

        ax[2, 1].plot(t, self.aux_states[:, 2], t, self.velocity_ref[:, 2])
        ax[2, 1].set_ylabel("Z-velocity [m/s]")
        ax[2, 1].grid()

        ax[0, 2].plot(t, self.aux_states[:, 3], label="State")
        ax[0, 2].plot(t, self.acceleration_ref[:, 0], label="Ref")
        ax[0, 2].plot(t, self.acceleration_des[:, 0], label="Des")
        ax[0, 2].set_ylabel("X-acceleration [m/s2]")
        ax[0, 2].grid()

        ax[1, 2].plot(t, self.aux_states[:, 4], t, self.acceleration_ref[:, 1], t, self.acceleration_des[:, 1])
        ax[1, 2].set_ylabel("Y-acceleration [m/s2]")
        ax[1, 2].grid()

        ax[2, 2].plot(t, self.aux_states[:, 5], t, self.acceleration_ref[:, 2], t, self.acceleration_des[:, 2])
        ax[2, 2].set_ylabel("Z-acceleration [m/s2]")
        ax[2, 2].grid()
        ax[0, 2].legend()

        plt.tight_layout()

        fig3, ax = plt.subplots(4, 1)

        ax[0].plot(t, np.sqrt(np.abs(self.inputs[:, 0])) * 60 / (2 * np.pi))
        ax[0].plot(t, drone.max_rotor_speed * np.ones(len(t), ) * 60 / (2 * np.pi))
        ax[0].set_ylabel("rotational velocity 1 [rpm]")
        ax[0].grid()

        ax[1].plot(t, np.sqrt(np.abs(self.inputs[:, 1])) * 60 / (2 * np.pi))
        ax[1].set_ylabel("rotational velocity 2 [rad/s]")
        ax[1].grid()

        ax[2].plot(t, np.sqrt(np.abs(self.inputs[:, 2])) * 60 / (2 * np.pi))
        ax[2].set_ylabel("rotational velocity 3 [rad/s]")
        ax[2].grid()

        ax[3].plot(t, np.sqrt(np.abs(self.inputs[:, 3])) * 60 / (2 * np.pi))
        ax[3].set_ylabel("rotational velocity 4 [rad/s]")
        ax[3].grid()

        plt.tight_layout()

        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(self.states[:, 6], self.states[:, 7], self.states[:, 8])
        ax.plot(self.position_ref[:, 0], self.position_ref[:, 1], self.position_ref[:, 2])
        ax.plot(trajectory.waypoints[:, 0], trajectory.waypoints[:, 1], trajectory.waypoints[:, 2])

        plt.show()
