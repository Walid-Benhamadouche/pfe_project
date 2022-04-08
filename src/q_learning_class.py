#!/usr/bin/env python
# imports
import rospy
import random
import numpy as np
import matplotlib.pyplot as plt

from numpy import asarray
from numpy import savetxt

class QLearning:

    # initializing the q_table and parameters using the config file
    def __init__(self):
        
        # parameters
        #self._env = env
        self._EPOCHS = rospy.get_param("/pfe/nepisodes")
        self._alpha = rospy.get_param("/pfe/alpha")
        self._epsilon = rospy.get_param("/pfe/epsilon")
        self._epsilon_discount = rospy.get_param("/pfe/epsilon_discount")
        self._gamma = rospy.get_param("/pfe/gamma")
        self._naction = rospy.get_param("/pfe/naction")

        self._reset_params()
        self._ep_rewards = []

        self._q_table = np.zeros([33334, self._naction])
        self._aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
        self._STEP = 50

    # reseting q learning parameters
    def _reset_params(self):
        self._epochs, self._penalties, self._reward, self._ep_reward = 0, 0, 0, 0
        #
        self._done = False

    # conveting observation to state
    def _obs_to_state(self,obs):
        for i in range(5):
            obs[i] = int(obs[i])
        state = int(''.join(map(str, obs)))
        #print('state: ',state)
        return state

    # printing aggr ep rewards
    def _show_aggr_ep_rewards(self,episode):
        print('\n', self._epsilon)
        if not episode % self._STEP:
            average_reward = sum(self._ep_rewards[-self._STEP:])/self._STEP
            self._aggr_ep_rewards['ep'].append(episode)
            self._aggr_ep_rewards['avg'].append(average_reward)
            self._aggr_ep_rewards['max'].append(max(self._ep_rewards[-self._STEP:]))
            self._aggr_ep_rewards['min'].append(min(self._ep_rewards[-self._STEP:]))
            print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {self._epsilon:>1.2f}')
    
    # saving q table results
    def _q_table_save(self, episode, array):
        np.savetxt(rospy.get_param("/pfe/path")+"/training_results/Q_"+episode+".csv", array, delimiter=",")
    
    # starting the learning phase
    def start_learning(self, env):
        fig=plt.figure()
        plt.ion()
        plt.show()
        for episode in range (self._EPOCHS+1):
            temp = []
            print("\n ", episode, "\n")

            # reset envirenment
            observation, self._done = env.reset()
            state = self._obs_to_state(observation)
            self._reset_params()

            while not self._done:
                # chosing action
                if random.uniform(0, 1) < self._epsilon:
                    action = random.randint(0, self._naction-1)
                else:
                    action = np.argmax(self._q_table[state,:])
                
                # executing action
                next_observation, self._reward, self._done = env.step(action)
                next_state = self._obs_to_state(next_observation)

                # updating q_table
                self._q_table[state, action] = self._q_table[state, action] + self._alpha*(self._reward + self._gamma*np.max(self._q_table[next_state, :]) - self._q_table[state, action])

                # updating state and ep reward
                state = next_state
                self._ep_reward = self._ep_reward + self._reward
            self._ep_rewards.append(self._ep_reward)

                # updating epsilon
            if self._epsilon > 0.05:
                self._epsilon = self._epsilon * self._epsilon_discount

            
            ax=fig.add_subplot(111, label="1")
            if not episode % self._STEP:
                temp.append(sum(self._ep_rewards[-self._STEP:])/self._STEP)
                ax2=fig.add_subplot(111, label="2", frame_on=False)
                ax2.plot(len(temp), temp, c = 'b')
                ax2.set_xticks([])
                ax2.set_yticks([])


            ax.plot(range(0, episode+1), self._ep_rewards, c = 'k')
            plt.pause(0.05)
            
            self._show_aggr_ep_rewards(episode)
            
            self._q_table_save(str(episode), self._q_table)
        
        self._q_table_save("Final", self._q_table)
        self._q_table_save("Rewards", self._ep_rewards)
        plt.savefig("traning rewards")

    # starting the testing phase
    def start_testing(self, env):
        
        self._q_table = np.loadtxt(rospy.get_param("/pfe/path")+"/training_results/Q_950.csv", delimiter=",")

        observation, self._done = env.reset()
        state = self._obs_to_state(observation)
        
        i=0
        while not self._done:
            action = np.argmax(self._q_table[state,:])

            next_observation, self._reward, self._done = env.step(action)
            next_state = self._obs_to_state(next_observation)

            state = next_state
            i = i +1
        print(i)