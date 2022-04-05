#!/usr/bin/env python
# imports
import rospy
import sys
sys.path.append(rospy.get_param("/pfe/path")+"/src")


import rospkg
import robot_envirenment as rbe
import q_learning_class as ql

if __name__ == '__main__':
    # init node
    try:
        rospy.init_node("navigation")
        print('node init')
        

        # init robot_envirenment/robot_monitoring(if made)
        env = rbe.RobotNameEnv()
        
        # init and start q_learning/deep_q_learning with envirenment as parameter
        q_learning = ql.QLearning()

        choice = input('Chose 0 for learning and 1 for testing: ')
        
        if int(choice) == 0:
            # start learning
            q_learning.start_learning(env)
        else:
            # start testing
            q_learning.start_testing(env)
        
    except rospy.ROSInterruptException:
        pass
    # Allow ROS to go to all callbacks.