import numpy as np
import pandas as pd


class Drone:
    def __init__(self, config):
        self.config = config

        # State variables
        init_pos = config.traj_config.init_pos
        self.state = np.array([init_pos[0], init_pos[1], init_pos[2], 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.dt = config.env_config.dt
        self.lin_acc = np.array([0, 0, 0])

        self.wind = True
        self.drag = []
        self.thrust = []
        self.gravity = []

        # Configuration variables
        self.m = config.drone_config.m
        self.I = config.drone_config.I

        self.lx = config.drone_config.lx
        self.ly = config.drone_config.ly

        self.rho = config.env_config.rho
        self.g = config.env_config.g

        self.kf = -4e-4
        self.km = 1e-5

        self.A = config.drone_config.A
        self.Cd_v = config.drone_config.Cd_v
        self.Cd_om = config.drone_config.Cd_om

        self.max_rotor_speed = np.sqrt(np.abs(self.m * self.g / (self.kf * 4)) * 2)

        #
        self.x_dim = config.drone_config.x_dim
        self.u_dim = config.drone_config.u_dim

    @property
    def attitude(self):
        return self.state[0:3]

    @property
    def angular_velocity(self):
        return self.state[3:6]

    @property
    def position(self):
        return self.state[6:9]

    @property
    def velocity_b(self):
        return self.state[9:12]

    @property
    def velocity_e(self):
        return self.rotationBodytoEarth(self.attitude) @ self.velocity_b

    def step(self, input: np.ndarray, wind=False):
        """ Discrete-time dynamics (Runge-Kutta 4) of a planar quadrotor """
        assert self.state.shape == (self.x_dim, ), f"{self.state.shape} does not equal {(self.x_dim, )}"
        assert input.shape == (self.u_dim, ), f"{input.shape} does not equal {(self.u_dim, )}"
        # Update state
        k1 = self.dxdt(self.state, input)
        k2 = self.dxdt(self.state + self.dt / 2 * k1, input)
        k3 = self.dxdt(self.state + self.dt / 2 * k2, input)
        k4 = self.dxdt(self.state + self.dt * k3, input)
        self.state = self.state + self.dt * (1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4)
        if wind and self.wind:
            Rbe = np.linalg.inv(self.rotationBodytoEarth(self.attitude))
            wind = 3 * np.array([np.sqrt(3) / 3, np.sqrt(3) / 3, np.sqrt(3) / 3])
            self.state[9:12] += Rbe @ wind
            self.wind = False

    def reset(self):
        init_pos = self.config.traj_config.init_pos
        self.state = np.array([init_pos[0], init_pos[1], init_pos[2], 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.dt = self.config.env_config.dt
        self.lin_acc = np.array([0, 0, 0])

    def dxdt(self, state, input) -> np.ndarray:
        """
        state: [phi, theta, psi, p, q, r, x, y, z, vx, vy, vz]
        """
        dx = np.zeros((self.x_dim, ))

        force = self.get_force(state, input)
        moment = self.get_moment(input)

        # Derivatives of attitude angles (phi, theta, psi)
        kinematics = np.array([[1, np.tan(state[1]) * np.sin(state[0]), np.tan(state[1]) * np.cos(state[0])],
                               [0, np.cos(state[0]), -np.sin(state[0])],
                               [0, np.sin(state[0]) / np.cos(state[1]), np.cos(state[0]) / np.cos(state[1])]])
        dx[0:3] = kinematics @ state[3:6]                                            # dphi, dtheta, dpsi

        # Derivative of angular velocity (p, q, r)
        dx[3:6] = np.array([((self.I[1, 1] - self.I[2, 2]) * state[5] * state[4] + moment[0]) / self.I[0, 0],
                            ((self.I[2, 2] - self.I[0, 0]) * state[5] * state[3] + moment[1]) / self.I[1, 1],
                            ((self.I[0, 0] - self.I[1, 1]) * state[4] * state[3] + moment[2]) / self.I[2, 2]])

        # Derivative of position (x, y, z)
        Reb = self.rotationBodytoEarth(state[0:3])  # rotation matrix body-fixed to NED
        dx[6:9] = Reb @ state[9:12]                                                # dx, dy, dz

        # Derivative of velocity (u, v, w)
        dx[9:12] = np.cross(state[9:12], state[3:6]) + force / self.m              # du, dv, dw

        # Linear acceleration
        self.lin_acc = Reb @ force / self.m                                         # ax, ay, az
        return dx

    def get_force(self, state: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        """ Determine resulting force from inputs in body reference frame """
        # Thrust
        T = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [self.kf, self.kf, self.kf, self.kf]]) @ np.clip(control_input, -self.max_rotor_speed**2,
                                                                       self.max_rotor_speed**2)
        self.thrust.append(np.linalg.norm(T))
        # Gravity
        Rbe = np.linalg.inv(self.rotationBodytoEarth(state[0:3]))
        G = Rbe @ np.array([0, 0, self.m * self.g])
        self.gravity.append(np.linalg.norm(G))

        # Drag
        # D = np.random.normal(self.Cd_v, self.Cd_v/5) * 1 / 2 * self.rho * state[9:12] ** 2 * self.A
        D = self.Cd_v * 1 / 2 * self.rho * state[9:12] ** 2 * self.A
        # D = 0 * 1 / 2 * self.rho * state[9:12] ** 2 * self.A
        self.drag.append(np.linalg.norm(D))

        return T + G + D

    def get_moment(self, control_input: np.ndarray) -> np.ndarray:
        """ Determine resulting moment from inputs in body reference frame """
        return np.array([[-self.lx * self.kf, -self.lx * self.kf, self.lx * self.kf, self.lx * self.kf],
                      [self.ly * self.kf, -self.ly * self.kf, -self.ly * self.kf, self.ly * self.kf],
                      [self.km, -self.km, self.km, -self.km]]) @ control_input

    def rotationBodytoEarth(self, euler_angles: np.ndarray) -> np.ndarray:
        phi = euler_angles[0]
        theta = euler_angles[1]
        psi = euler_angles[2]

        Ryaw = np.array([[np.cos(psi), -np.sin(psi), 0],
                         [np.sin(psi), np.cos(psi), 0],
                         [0, 0, 1]])
        Rpitch = np.array([[np.cos(theta), 0, np.sin(theta)],
                           [0, 1, 0],
                           [-np.sin(theta), 0, np.cos(theta)]])
        Rroll = np.array([[1, 0, 0],
                          [0, np.cos(phi), -np.sin(phi)],
                          [0, np.sin(phi), np.cos(phi)]])
        Rbe = Ryaw @ Rpitch @ Rroll

        assert Rbe.shape == (3, 3), f"Rotation matrix does not have the correct shape"
        assert np.isclose(np.linalg.norm(Rbe @ np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])), 1), f"Rotation changes vector magnitude"
        return Rbe



def test_rotationBodyEarth():
    from configs.config import Config
    config = Config()
    drone = Drone(config)

    euler_angles = np.array([np.pi / 12, np.pi / 6, np.pi / 3])
    Rbe = drone.rotationBodytoEarth(euler_angles)

    # Zero euler angles
    euler_angles = np.array([0, 0, 0])
    Rbe = drone.rotationBodytoEarth(euler_angles)
    assert np.allclose(Rbe, np.array([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]]))

    # Deviation in roll by 30 degrees
    roll = np.pi / 6
    euler_angles = np.array([roll, 0, 0])
    Rbe = drone.rotationBodytoEarth(euler_angles)
    Rbe_expected = np.array([[1, 0, 0],
                             [0, np.cos(roll), -np.sin(roll)],
                             [0, np.sin(roll), np.cos(roll)]])
    assert np.allclose(Rbe, Rbe_expected)

    # Deviation in pitch by 30 degrees
    pitch = np.pi / 6
    euler_angles = np.array([0, pitch, 0])
    Rbe = drone.rotationBodytoEarth(euler_angles)
    Rbe_expected = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                             [0, 1, 0],
                             [-np.sin(pitch), 0, np.cos(pitch)]])
    assert np.allclose(Rbe, Rbe_expected)

    # Deviation in yaw by 30 degrees
    yaw = np.pi / 6
    euler_angles = np.array([0, 0, yaw])
    Rbe = drone.rotationBodytoEarth(euler_angles)
    Rbe_expected = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                             [np.sin(yaw), np.cos(yaw), 0],
                             [0, 0, 1]])
    assert np.allclose(Rbe, Rbe_expected)

    # Combination of roll, pitch, yaw
    roll = np.pi / 3
    pitch = np.pi / 6
    yaw = np.pi / 8
    euler_angles = np.array([roll, pitch, yaw])
    Rbe = drone.rotationBodytoEarth(euler_angles)
    Ryaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                     [np.sin(yaw), np.cos(yaw), 0],
                     [0, 0, 1]])
    Rpitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
    Rroll = np.array([[1, 0, 0],
                     [0, np.cos(roll), -np.sin(roll)],
                     [0, np.sin(roll), np.cos(roll)]])
    Rbe_expected = Ryaw @ Rpitch @ Rroll
    assert np.allclose(Rbe, Rbe_expected)


def test_step_z():
    """ Testing vertical dynamics implementation """
    drone = Drone()

    t = [0]
    initial_state = np.array([0, np.pi / 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    drone.state = initial_state

    while t[-1] < 10:
        # Input signal
        input = -np.array([3.705, 3.705, 3.705, 3.705]) / drone.kf

        # Step
        drone.step(input)
        t.append(t[-1] + drone.dt)

    assert np.isclose(np.pi / 15, drone.state[1])
    assert np.isclose(-0.5132, drone.lin_vel[2], atol=1e-1)
    assert np.isclose(-2.5662, drone.state[8], atol=1e-1)


def test_step_1():
    """ Testing against test input 1 of matlab script """

    drone = Drone()

    t = [0]
    initial_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    drone.state = initial_state

    attitude = np.array(pd.read_csv('../verification/attitude_1.csv'))
    angular_velocity = np.array(pd.read_csv('../verification/angular_velocity_1.csv'))
    position = np.array(pd.read_csv('../verification/position_1.csv'))
    velocity = np.array(pd.read_csv('../verification/velocity_1.csv'))

    i = 0
    j = 0
    while t[-1] < 10:
        # Input signal
        if t[-1] <= 1:
            input = np.array([4.905, 4.905, 5.005, 4.905])
        elif t[-1] <= 2:
            input = np.array([5.005, 4.905, 4.905, 4.905])
        else:
            input = np.array([4.905, 4.905, 4.905, 4.905])

        # Step
        drone.step(input)
        t.append(t[-1] + drone.dt)

        # Check states
        if i % 2 == 0 and j < 500:
            # Check position
            assert np.allclose(attitude[j, :], drone.state[0:3], atol=1e-2)
            assert np.allclose(angular_velocity[j, :], drone.state[3:6], atol=1e-2)
            assert np.allclose(position[j, :], drone.state[6:9], atol=5e-1)
            assert np.allclose(velocity[j, :], drone.state[9:12], atol=1e-1)
            j += 1
        i += 1

def test_step_2():
    """ Testing against test input 2 of matlab script """

    drone = Drone()

    t = [0]
    initial_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    drone.state = initial_state

    attitude = np.array(pd.read_csv('../verification/attitude_2.csv'))
    angular_velocity = np.array(pd.read_csv('../verification/angular_velocity_2.csv'))
    position = np.array(pd.read_csv('../verification/position_2.csv'))
    velocity = np.array(pd.read_csv('../verification/velocity_2.csv'))

    i = 0
    j = 0
    while t[-1] < 10:
        # Input signal
        if t[-1] <= 1:
            input = np.array([4.905, 5.005, 5.005, 4.905])
        elif t[-1] <= 2:
            input = np.array([5.005, 4.905, 4.905, 5.005])
        else:
            input = np.array([4.905, 4.905, 4.905, 4.905])

        # Step
        drone.step(input)
        t.append(t[-1] + drone.dt)

        # Check states
        if i % 2 == 0 and j < 500:
            # Check position
            assert np.allclose(attitude[j, :], drone.state[0:3], atol=1e-2)
            assert np.allclose(angular_velocity[j, :], drone.state[3:6], atol=1e-2)
            assert np.allclose(position[j, :], drone.state[6:9], atol=1)
            assert np.allclose(velocity[j, :], drone.state[9:12], atol=2e-1)
            j += 1
        i += 1

