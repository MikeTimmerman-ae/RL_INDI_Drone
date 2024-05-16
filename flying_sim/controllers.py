import numpy as np


class ControlAllocation:

    def __init__(self, drone):
        self.drone = drone
        self.km = drone.km
        self.kf = drone.kf
        self.lx = drone.lx
        self.ly = drone.ly

    def get_control_input(self, control_moment: np.ndarray, thrust: float) -> np.ndarray:
        """ Determine the trust levels for each propeller based on desired control moment and thrust """
        A = np.array([[-self.lx * self.kf, -self.lx * self.kf, self.lx * self.kf, self.lx * self.kf],
                      [self.ly * self.kf, -self.ly * self.kf, -self.ly * self.kf, self.ly * self.kf],
                      [self.km, -self.km, self.km, -self.km],
                      [self.kf, self.kf, self.kf, self.kf]])
        b = np.hstack((control_moment, np.array(thrust)))
        control_input = np.linalg.solve(A, b)
        return control_input


class AttitudeController:

    def __init__(self, drone):
        self.kp_ang_acc = np.diag([10, 10, 10])
        self.kp_att = np.diag([6, 6, 4])
        self.drone = drone
        self.prev_ang_vel = np.zeros((3,))

    def get_des_angular_vel(self, curr_attitude: np.ndarray, des_attitude: np.ndarray):
        """ Get desired angular velocity from current attitude using nonlinear dynamic inversion and proportional feedback """
        des_der_attitude = self.P_attitude(curr_attitude, des_attitude)
        des_angular_vel = self.inversion(curr_attitude, des_der_attitude)
        return des_angular_vel

    def P_attitude(self, curr_attitude: np.ndarray, des_attitude: np.ndarray) -> np.ndarray:
        return self.kp_att @ (des_attitude - curr_attitude)

    def inversion(self, attitude: np.ndarray, des_der_attitude: np.ndarray):
        """ Calculate desired angular velocity based on inverting kinematic equations using desired derivative of attitude """
        return np.array([[1, 0, -np.sin(attitude[1])],
                         [0, np.cos(attitude[0]), np.cos(attitude[1]) * np.sin(attitude[0])],
                         [0, -np.sin(attitude[1]), np.cos(attitude[1]) * np.cos(attitude[0])]]) @ des_der_attitude

    def get_control_moment(self, curr_moment: np.ndarray, curr_angular_vel: np.ndarray, des_angular_vel: np.ndarray):
        """ Get control moment from desired angular velocity """
        curr_angular_acc = self.angular_acc(curr_angular_vel)
        des_angular_acc = self.P_angular_velocity(curr_angular_vel, des_angular_vel)
        control_moment = self.iinversion(curr_moment, curr_angular_acc, des_angular_acc)
        return control_moment

    def angular_acc(self, angular_vel: np.ndarray):
        """ Determine angular acceleration by differentiating angular velocity """
        angular_acc = (angular_vel - self.prev_ang_vel) / self.drone.dt
        self.prev_ang_vel = angular_vel
        return angular_acc

    def P_angular_velocity(self, curr_angular_vel: np.ndarray, des_angular_vel: np.ndarray, ) -> np.ndarray:
        """ Calculate desired angular acceleration from desired angular velocity based on PD control """
        return self.kp_ang_acc @ (des_angular_vel - curr_angular_vel)

    def iinversion(self, curr_moment: np.ndarray, curr_angular_acc: np.ndarray, des_angular_acc: np.ndarray) -> np.ndarray:
        """ Calculate desired control moment based on previous moment, and desired and current angular acceleration """
        return self.drone.I @ (des_angular_acc - curr_angular_acc) + curr_moment


class PositionController:
    def __init__(self, drone):
        self.drone = drone

        self.kp_acc = np.diag([0.1, 0.1, 0.1])
        self.kp_vel = np.diag([0.9, 0.9, 0.3])
        self.kp_pos = np.diag([1.2, 1.2, 0.5])

        self.LPF_acc = LowPassFiter(drone.dt, 2 * np.pi * 2, 3)
        self.LPF_att = LowPassFiter(drone.dt, 2 * np.pi * 6, 2)

        self.a_max = 12

    def get_desired_lin_acc(self, curr_pos: np.ndarray, curr_vel: np.ndarray, cur_acc: np.ndarray, pos_ref: np.ndarray, vel_ref: np.ndarray, acc_ref: np.ndarray):
        des_lin_acc = self.kp_pos @ (pos_ref - curr_pos) + self.kp_vel @ (vel_ref - curr_vel) + self.kp_acc @ (acc_ref - cur_acc) + acc_ref
        return np.clip(des_lin_acc, -self.a_max, self.a_max)

    def P_position(self, curr_position: np.ndarray, des_position: np.ndarray) -> np.ndarray:
        return self.kp_pos @ (des_position - curr_position)

    def P_lin_velocity(self, curr_velocity: np.ndarray, des_velocity: np.ndarray) -> np.ndarray:
        return self.kp_vel @ (des_velocity - curr_velocity)

    def get_desired_attitude(self, curr_attitude: np.ndarray, curr_thrust: float, curr_lin_acc: np.ndarray, des_lin_acc: np.ndarray, beta=np.zeros((3,))):
        """ Get desired attitude from reference position """
        des_attitude = self.iinversion(curr_attitude, curr_thrust, curr_lin_acc, des_lin_acc, beta=beta)
        return np.array([des_attitude[0], des_attitude[1]]), des_attitude[2]

    def iinversion(self, curr_attitude: np.ndarray, curr_thrust: float, curr_lin_acc: np.ndarray, des_lin_acc: np.ndarray, beta) -> np.ndarray:
        """ Calculate desired control moment based on previous moment, and desired and current angular acceleration """
        phi = curr_attitude[0]
        theta = curr_attitude[1]
        psi = curr_attitude[2]

        G = 1 / self.drone.m * np.array([[(np.sin(psi) * np.cos(phi) - np.cos(psi) * np.sin(theta) * np.sin(phi)) * curr_thrust,
                                                (np.cos(psi) * np.cos(theta) * np.cos(phi)) * curr_thrust,
                                                np.sin(psi) * np.sin(phi) + np.cos(psi) * np.sin(theta) * np.cos(phi)],
                                               [(-np.cos(psi) * np.cos(phi) - np.sin(psi) * np.sin(theta) * np.sin(phi)) * curr_thrust,
                                                (np.sin(psi) * np.cos(theta) * np.cos(phi)) * curr_thrust,
                                                -np.cos(psi) * np.sin(phi) + np.sin(psi) * np.sin(theta) * np.sin(phi)],
                                               [(-np.cos(theta) * np.sin(phi)) * curr_thrust,
                                                (-np.sin(theta) * np.cos(phi)) * curr_thrust,
                                                np.cos(theta) * np.cos(phi)]])
        Ginv = np.linalg.inv(G)

        des_attitude = Ginv @ (des_lin_acc - curr_lin_acc - beta) + np.array([phi, theta, curr_thrust])
        return des_attitude


class LowPassFiter:

    def __init__(self, dt, om_c, n):
        self.om_c = om_c
        self.K = 2 / dt
        self.prev_signal = np.zeros((n,))
        self.prev_output = np.zeros((n,))

    def filter(self, curr_signal):
        output = (self.om_c * (curr_signal + self.prev_signal) + (self.K - self.om_c) * self.prev_output) / (self.K + self.om_c)
        self.prev_signal = curr_signal
        self.prev_output = output
        return output