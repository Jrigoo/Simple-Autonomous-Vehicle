import os
import json
import numpy as np
from scripts.utils.gps_util import UDP_GPS_Parser
from scripts.utils.imu_util import UDP_IMU_Parser
from scripts.utils.cam_util import UDP_CAM_Parser
from network.sender.ctrl_cmd_sender import CtrlCmdSender


class Sensors:
    def __init__(self, cam=False):
        """
        1. Obtiene el ip y puerto de cada sensor
        2. Inicializa cada sensor
        """
        path = os.getcwd()
        with open(os.path.join(path, ("params.json")), 'r') as fp:
            params = json.load(fp)

        params = params["params"]
        self.user_ip = params["user_ip"]
        self.host_ip = params["host_ip"]

        if params["is_local"] == "True":
            self.user_ip = params["local_user_ip"]
            self.host_ip = params["local_host_ip"]

        self.gps_port = params["gps_dst_port"]
        self.imu_port = params["imu_dst_port"]
        self.cam_port = params["cam_dst_port"]
        self.cmd_port = params["ctrl_cmd_host_port"]

        self.gps_parser = UDP_GPS_Parser(self.user_ip, self.gps_port, 'GPRMC')
        self.imu_parser = UDP_IMU_Parser(self.user_ip, self.imu_port, 'imu')
        self.cmd = CtrlCmdSender(self.host_ip, self.cmd_port)

        params_cam = {
            "localIP": self.user_ip,
            "localPort": self.cam_port,
            "Block_SIZE": int(65000)
        }
        if cam:
            self.udp_cam = UDP_CAM_Parser(
                ip=params_cam["localIP"], port=params_cam["localPort"], params_cam=params_cam)

    def gps(self):
        if self.gps_parser.parsed_data != None:
            latitude = self.gps_parser.parsed_data[0]
            longitude = self.gps_parser.parsed_data[1]
            return latitude, longitude

    def imu(self):
        if len(self.imu_parser.parsed_data) == 10:
            quaternion = np.array([round(self.imu_parser.parsed_data[0], 2), round(self.imu_parser.parsed_data[1], 2), round(
                self.imu_parser.parsed_data[2], 2), round(self.imu_parser.parsed_data[3], 2)])
            ang_vel_XYZ = np.array([round(self.imu_parser.parsed_data[4], 2), round(
                self.imu_parser.parsed_data[5], 2), round(self.imu_parser.parsed_data[6], 2)])
            lin_acc_XYZ = np.array([round(self.imu_parser.parsed_data[7], 2), round(
                self.imu_parser.parsed_data[8], 2), round(self.imu_parser.parsed_data[9], 2)])
            return [quaternion, ang_vel_XYZ, lin_acc_XYZ]

    def camara(self):
        if self.udp_cam.is_img == True:
            img = self.udp_cam.raw_img
            return img
