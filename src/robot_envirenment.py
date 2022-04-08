#!/usr/bin/env python
# imports
import rospy
import math
import time

from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

# creating an envirenement class (not sure about using classes)
class RobotNameEnv:
    _cmd_vel_pub = ""
    _laser = []
    _reward = 0
    _number_of_steps = 0
    # initializing the class variables and subscribing to topics
    def __init__(self):
        
        self._number_of_steps = 0
        self._reward = 0
        #subscribers
        rospy.Subscriber('/scan', LaserScan, self._laser_scan_callback)
        rospy.loginfo("/scan ready !!")

        #publisher
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.loginfo("/cmd_vel ready !!")

        self._calculat_state()
        time.sleep(0.2)
    
    # the callback function for laser_scan subscription 
    def _laser_scan_callback(self, msg):
        self._laser = msg.ranges
    
    # taking only a finite number of laser scans
    def _discretize_scan_observation(self, mod):
        disc_scan = []
        #for i in range(len(self._laser)):
        #    if i % mod == 0:
        #        if self._laser[i+45] != float('+inf'):
        #            disc_scan.append(self._laser[i+45])
        #        else:
        #            disc_scan.append(3)
        #if self._laser[0] != float('+inf'):
        #    disc_scan.append(self._laser[0])
        #else:
        #    disc_scan.append(3)
        for i in range(mod):
            temp = self._laser[-40+i*25]
            if temp != float('+inf'):
                disc_scan.append(temp)
            else:
                disc_scan.append(3)
        
        return disc_scan

    # takes curent laser scans and return the current state
    def _calculat_state(self):
        mod = rospy.get_param("/pfe/laser_to_skip")
        if len(self._laser) == 360:
            return self._discretize_scan_observation(mod)
        else:
            return 0

    # takes a state and return the reward
    def _calculat_reward(self, action, done):
        if done:
            self._reward
        elif action == 0:
            self._reward = rospy.get_param("/pfe/forwards_reward")
        elif action == 1:
            self._reward = rospy.get_param("/pfe/turn_reward")
        else:
            self._reward = rospy.get_param("/pfe/turn_reward")
        return self._reward
    
    # checks if the episode is done (hit an opstacle for example)
    def _is_done(self, state):
        for i in state:
            if i < 0.19:
                self._reward = -rospy.get_param("/pfe/end_episode_reward")
                return True
        if self._number_of_steps == 1000:
            self._reward = rospy.get_param("/pfe/end_episode_reward")
            return True
        else:
            return False
    
    ### THE FOLLOWING TWO FUNCTIONS ARE NOT USED BUT THEY STILL WORK IT IS NOT RECOMMENDED TO USE
    ### _deleting_robot FUNCTION FOR MEMORY LEAK REASONS IN THE GAZEBO SIMULATOR

    # spawning robot
    def _spawning_robot(self):
        spawn_robot_client = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
        path = rospy.get_param("/pfe/path")
        pos = Pose()
        pos.position.x = float(-0.7)
        pos.position.y = float(0.0)
        pos.position.z = float(0.0)
        spawn_robot_client(
            model_name = 'turtlebot3_burger',
            model_xml = open(path+'/robot_description/turtlebot3_burger.urdf', 'r').read(),
            robot_namespace = '/',
            initial_pose = pos,
            reference_frame = 'world'
        )

    # deleting robot
    def _deleting_robot(self):
        delete_robot_client = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        delete_robot_client('turtlebot3_burger')

    # resetting the envirenment (robot position and parameters) and return the initial state
    def reset(self):

        rospy.wait_for_service('/gazebo/reset_world')
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_world()
        rospy.loginfo("World Reset done.")  
        
        print('number of steps: ', self._number_of_steps)
        self._number_of_steps = 0
        self._reward = 0

        return self._calculat_state(), False
        

    # takes an action and execut it while returning the new state, the reward of the previous state
    def step(self, action):
        mv = Twist()
        if action == 0:
            mv.linear.x = 0.10
        elif action == 1:
            mv.linear.x = 0.035
            mv.angular.z =(10* math.pi / 180.)
        else:
            mv.linear.x = 0.035
            mv.angular.z =(-10* math.pi / 180.)
        self._cmd_vel_pub.publish(mv)
        time.sleep(0.2)
        self._number_of_steps = self._number_of_steps + 1

        state = self._calculat_state() 
        done = self._is_done(state)
        print(state)
        reward = self._calculat_reward(action, done)
        return state, reward, done

    # getters and setters depending on variables needed