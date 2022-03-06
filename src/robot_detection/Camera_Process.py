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
from image_manipulation import Image_processes
from std_msgs.msg import Float64

class image_converter:

  # Defines publisher and subscriber
  def __init__(self):

    self._processor = Image_processes()
      
    self.robotPosition = None
    # initialize the node named image_processing
    rospy.init_node('GPSVideo_Processor', anonymous=True)
    # initialize a publisher to send xz coordinates
    self.pos_pub = rospy.Publisher("/robot_position", Point ,queue_size = 1)
    self.pos_pub_longlat = rospy.Publisher('/robot_position_longlat', Point, queue_size=1) #TODO: impletement this
    
    self.cap = cv2.VideoCapture(0)
    
    rate = rospy.Rate(50)  # 5hz
    # record the beginning time
    while not rospy.is_shutdown():
        
        rate.sleep()

        ret, frame = self.cap.read()
        frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        robot_position = self._processor.runProcessor(frame)
        # print(robot_position)
        position_in_point = Point(float(robot_position[0]), float(robot_position[1]), -999)
        print(position_in_point)
        self.pos_pub.publish(position_in_point)
        # cv2.imshow('Input', frame)

        # c = cv2.waitKey(1)
        # if c == 27:
        #     break



        
        # # Publish the results
        # try: 
        #     self.robotPosition = imP.runProcessor(frame)
        #     self.pos_pub.publish(json.dumps({'bench1': self.robotPosition}))
          
        # except e:
        #   print(e)

    cap.release()
    cv2.destroyAllWindows()

        

# call the class
def main(args):
  ic = image_converter()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)