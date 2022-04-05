#!/usr/bin/env python

import rospy
import math
import pandas as pd
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage
from tf.transformations import euler_from_quaternion, quaternion_from_euler

def laser_callback(msg):
    global laser
    laser = msg.ranges;
    print(len(laser))
    print("0",laser[0])
    print("89",laser[89])
    print("179",laser[179])
    print("269",laser[269])
#####################

def deg2rad(degVal):
    return (degVal* math.pi / 180.)

if __name__ == '__main__':
    
    try:
        rospy.init_node("navigation")
        pub = rospy.Publisher('turtle1//cmd_vel', Twist, queue_size=1)
        rospy.loginfo("/cmd_vel ready !!")

        laser_pub = rospy.Subscriber('/scan', LaserScan, laser_callback)
        rospy.loginfo("/range ready !!")
        while(1):
            
            mv = Twist()
            mv.linear.x = 0.25
            mv.angular.z = deg2rad(30)
            pub.publish(mv)

    except rospy.ROSInterruptException:
        pass
    # Allow ROS to go to all callbacks.
    rospy.spin()