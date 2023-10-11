
import os
import time
from pyproj import Proj
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scripts.sensors import Sensors
from scripts.vision import Vision
from scripts.functions import euclidean, quaternion_to_euler


class Pure_Pursuit:
    def __init__(self, path):
        self.error = False
        self.get_route(path)
        plt.ion()  # Inicializa matplotlib
        self.figure, self.ax = plt.subplots(figsize=(5, 5))

        self.proj_UTM = Proj(proj='utm', zone=52,
                             ellps='WGS84')

        # [20,10,30,9,36.3]
        # [20, 10, 30, 10,36.3]
        # [20, 10, 30, 10,40]
        # [20, 3, 25, 5, 36.3]
        # [20, 3, 25, 10, 36.3]
        # [15, 3, 25, 7, 36.3] - Ultimo intento con yaw
        self.p_ini, self.increment, self.velocity, self.ld,  self.max_timon = [
            20, 3, 30, 6.5, 36.3]

    def get_route(self, route):
        path = os.path.join(os.getcwd(), ("routes"))
        txt = os.path.join(path, (route))
        df = pd.read_csv(txt, sep="\t", header=None, names=["X", "Y"])
        self.traj_x, self.traj_y = df.X, df.Y

    def gps_to_utm(self, lat, lon):
        x, y = self.proj_UTM(lon, lat)
        return x - 302459.942, y - 4122635.537

    def calc_steering(self, p1, p2, h):

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angulo_objetivo = np.arctan2(dy, dx)
        theta = np.rad2deg(angulo_objetivo)

        if theta < 0:
            theta = 360 + theta

        print(theta)
        s1 = h - theta
        if h > theta:
            a = h
            b = theta
        else:
            a = theta
            b = h
        s2 = (360 - a) + b

        if abs(s1) > abs(s2):
            alpha = s2
        else:
            alpha = s1

        deg_steering = max(-self.max_timon, min(alpha,
                           self.max_timon))
        steering = deg_steering/self.max_timon

        return theta, alpha, deg_steering, steering

    def autonomous(self, lat, lon, imu, speed):
        self.x, self.y = self.gps_to_utm(lat, lon)  # Obtenemos XY
        quaternion = imu[0]  # quaternion
        euler_angles = quaternion_to_euler(quaternion)
        heading = 180 - euler_angles[0]  # Current yaw

        px, py = self.traj_x[self.p_ini], self.traj_y[self.p_ini]
        distance = euclidean(self.x, self.y, px, py)

        if distance < self.ld:
            self.p_ini += self.increment

        _, _, _, self.steering = self.calc_steering(
            (self.x, self.y), (px, py), heading)

        plt.scatter(self.traj_x, self.traj_y, c='g', s=0.1)
        plt.scatter(self.x, self.y, c='r')
        plt.scatter(px, py, c='b')

        self.figure.canvas.flush_events()
        self.ax.clear()


if __name__ == '__main__':
    pp = Pure_Pursuit("beginner_path.txt")
    sensors = Sensors(cam=False)

    while True:
        if sensors.gps() and sensors.imu():
            imu = sensors.imu()
            lat, lon = sensors.gps()
            current_velocity = sensors.ego_velocity
            pp.autonomous(lat, lon, imu, current_velocity)

            sensors.cmd.send_data([pp.velocity, 0, pp.steering])

        time.sleep(0.01)
