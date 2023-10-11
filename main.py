import time
from scripts.sensors import Sensors
from scripts.waypoint import Waypoint


if __name__ == '__main__':
    wp = Waypoint("beginner_path.txt")
    sensors = Sensors(cam=False)

    while True:
        if sensors.gps() and sensors.imu():
            imu = sensors.imu()
            lat, lon = sensors.gps()
            wp.autonomous(lat, lon, imu)

            sensors.cmd.send_data([wp.velocity, 0, wp.steering])

        time.sleep(0.01)
