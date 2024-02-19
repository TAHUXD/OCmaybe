#!/usr/bin/env python

import queue
from turtle import position
from typing import Dict
import sys
import rclpy  # ROS 2 equivalent for rospy
from rclpy.node import Node  # ROS 2 way to create a node
import cv2
import numpy as np
import json
from geometry_msgs.msg import Point  # ROS 2 standard message type for points
from image_processor import ImageProcessor  # Assuming Image_processor is a custom class
from std_msgs.msg import Float64
import copy

class GPSSimulator(Node):
    def __init__(self):
        super().__init__('GPSVideo_Processor')
        self.LAT_MIN = 0.0
        self.LAT_MAX = 1.2631578947
        self.LONG_MIN = 0.0
        self.LONG_MAX = 1.0
        self.IMAGE_Y = 950
        self.IMAGE_X = 1200

        self.robotPosition = None
        self._processor = ImageProcessor()

        self.pos_pub = self.create_publisher(Point, "/robot_position", 10)
        self.bench2_pub = self.create_publisher(Point, '/bench2_position_longlat', 10)
        self.bench3_pub = self.create_publisher(Point, '/bench3_position_longlat', 10)
        self.pos_pub_longlat = self.create_publisher(Point, '/robot_position_longlat', 10)

        self.cap = cv2.VideoCapture(0)
        self.timer = self.create_timer(1/30, self.process_video)

    def process_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to capture video frame')
            return

        frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        cv2.imwrite('camera_input.jpg', frame)

        robot_position, angle, position_yellow, position_green = self._processor.runProcessor(frame)
        position_in_point_robot = Point(x=float(robot_position[1]), y=float(robot_position[0]), z=angle)
        position_in_point_bench2 = Point(x=float(position_yellow[1]), y=float(position_yellow[0]), z=0)
        position_in_point_bench3 = Point(x=float(position_green[1]), y=float(position_green[0]), z=0)

        self.pos_pub.publish(position_in_point_robot)
        self.pos_pub_longlat.publish(self.convert_to_longlat(position_in_point_robot))
        self.bench2_pub.publish(self.convert_to_longlat(position_in_point_bench2))
        self.bench3_pub.publish(self.convert_to_longlat(position_in_point_bench3))

    def convert_to_longlat(self, position_in_point):
        change_in_Long = self.LONG_MAX - self.LONG_MIN
        change_in_lat = self.LAT_MAX - self.LAT_MIN
        long_conversion_constant = change_in_Long / self.IMAGE_Y
        lat_conversion_constant = change_in_lat / self.IMAGE_X

        point_to_convert = copy.deepcopy(position_in_point)
        point_to_convert.x = long_conversion_constant * point_to_convert.x
        point_to_convert.y = lat_conversion_constant * point_to_convert.y

        return point_to_convert

def main(args=None):
    rclpy.init(args=args)
    gps_simulator = GPSSimulator()

    try:
        rclpy.spin(gps_simulator)
    except KeyboardInterrupt:
        gps_simulator.get_logger().info('GPS Simulator stopped cleanly')
    except BaseException:
        gps_simulator.get_logger().info('Exception in GPS Simulator', exc_info=True)
    finally:
        gps_simulator.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()