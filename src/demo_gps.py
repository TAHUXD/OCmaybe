#!/usr/bin/env python
# license removed for brevity
import rospy
from parked_custom_msgs.msg import Point

def talker():
    bench1 = rospy.Publisher('bench1/gps_pos', Point, queue_size=10)
    bench2 = rospy.Publisher('bench2/gps_pos', Point, queue_size=10)
    rospy.init_node('cv_gps_system', anonymous=True)
    rate = rospy.Rate(1) # 1hz
    pos = Point(1.002001231456, 1.0000)
    while not rospy.is_shutdown():
        pos = Point(pos.long + 1, pos.lat + 1)
        rospy.loginfo(pos)
        bench1.publish(pos)
        bench2.publish(pos)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass