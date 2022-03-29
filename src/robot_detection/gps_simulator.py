#!/usr/bin/env python

import queue
from typing import Dict
import roslib
import sys
import rospy
import cv2
import numpy as np
import json
from parked_custom_msgs.msg import Point
from image_processor import Image_processor
from std_msgs.msg import Float64
import copy

# Provides an entry into the gps simulator.
# Process flow:
#   Fetches merged camera stream from the local network.
#   Divides the stream into respective camera streams.
#   Corrects for distortion.
#   stitches the feeds together to form a comprehensive image of the space.
#   Detects the robot and reports its position and orientation.
class GPS_SIMULATOR:

  # Defines publisher and subscriber
  def __init__(self):
    # constants for conversions between the world frame and the system's frame.
    self.LAT_MIN = 0.0
    self.LAT_MAX = 1.2631578947
    self.LONG_MIN = 0.0
    self.LONG_MAX = 1.0
    self.IMAGE_Y = 950
    self.IMAGE_X = 1200

    self.robotPosition = None
    # initialize a Processor to perform the required process flow.
    self._processor = Image_processor()
    # initialize the node named image_processing
    rospy.init_node('GPSVideo_Processor', anonymous=True)
    # initialize a publisher to send robot coordinates in the system's frame.
    self.pos_pub = rospy.Publisher("/robot_position", Point ,queue_size = 1)
    # initialize a publisher to send robot coordinates in the world frame.
    self.pos_pub_longlat = rospy.Publisher('/robot_position_longlat', Point, queue_size=1)

    self.cap = cv2.VideoCapture(0)
    
    rate = rospy.Rate(4)
   
    while not rospy.is_shutdown():
        
        rate.sleep()

        ret, frame = self.cap.read()
        frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        cv2.imwrite('camera_input.jpg', frame)

        robot_position, angle = self._processor.runProcessor(frame)
        position_in_point = Point(float(robot_position[1]), float(robot_position[0]), angle)
        # print(position_in_point)

        change_in_Long = self.LONG_MAX - self.LONG_MIN
        change_in_lat = self.LAT_MAX - self.LAT_MIN
        long_conversion_constant = change_in_Long / self.IMAGE_Y
        lat_conversion_constant = change_in_lat / self.IMAGE_X

        point_to_convert = copy.deepcopy(position_in_point)
        
        point_to_convert.long = long_conversion_constant * point_to_convert.long
        point_to_convert.lat = lat_conversion_constant * point_to_convert.lat

        #print(position_in_point)
        self.pos_pub.publish(position_in_point)
        self.pos_pub_longlat.publish(point_to_convert)
        print(point_to_convert)

    cap.release()
    cv2.destroyAllWindows()

        

# call the class
def main(args):
  gs = GPS_SIMULATOR()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)