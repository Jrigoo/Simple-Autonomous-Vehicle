import math
import numpy as np
from scipy.spatial.transform import Rotation


def euclidean(x1, y1, x2, y2):
    return (((x1-x2)**2)+((y1-y2)**2))**(0.5)


def quaternion_to_euler(quaternion):
    """
    Convertir de Quaternarios a Angulos de Euler
    :quaternion -> Un array con la posición en w,x,y,z
    :return -> yaw,roll,pitch en grados
    """
    r = Rotation.from_quat(quaternion)
    euler_angles = r.as_euler('XYZ')
    return np.degrees(euler_angles)


def quaternion_to_euler_2(quaternion):
    """
    Convertir de Quaternarios a Angulos de Euler
    :quaternion -> Un array con la posición en w,x,y,z
    :return -> yaw,roll,pitch en grados
    """
    w, x, y, z = quaternion
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return np.degrees(yaw), roll, pitch
