#!/usr/bin/env python
#imports
import numpy as np
from numpy import asarray
from numpy import savetxt
import random

#initialising variables
def Q_learn(env, alpha, epsilon, gamma, EPOCHS, epsilon_discount):
    q_table = np.zeros([11112, 3])
    aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}
    STEP = 100

    #Q-learning algorithm
    for episode in range (EPOCHS):
        print("\n ", episode, "\n")
        observation = env.reset()
        print(observation)
        for i in range(5):
            observation[i] = int(observation[i])
            if observation[i] > 0:
                observation[i] = 1
        print(observation)
        state = int(''.join(map(str, observation)))
        print('state: ',state)
        epochs, penalties, reward, ep_reward = 0, 0, 0, 0
        ep_rewards = []
        done = False
        while not done:
            if random.uniform(0, 1) > epsilon:
                action = np.argmax(q_table[state])
            else:
                action = random.randrange(0, 2)
            next_observation, reward, done = env.step(action)
            for i in range(5):
                next_observation[i] = int(next_observation[i])
                if next_observation[i] > 0:
                    next_observation[i] = 1
            next_state = int(''.join(map(str, next_observation)))
            ep_reward = ep_reward + reward
            q_table[state, action] = (1-alpha)*q_table[state, action]+alpha*(reward+gamma*np.max(q_table[next_state]))
            if reward == -200:
                break
            state = next_state
        if epsilon > 0.05:
            epsilon = epsilon * epsilon_discount
        ep_rewards.append(ep_reward)
        if not episode % STEP:
            average_reward = sum(ep_rewards[-STEP:])/STEP
            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['max'].append(max(ep_rewards[-STEP:]))
            aggr_ep_rewards['min'].append(min(ep_rewards[-STEP:]))
            print(f'Episode: {episode:>5d}, average reward: {average_reward:>4.1f}, current epsilon: {epsilon:>1.2f}')
    print("start saving")
    data = asarray(q_table)
    savetxt('q_table.csv', data, delimiter=',')
    print("Training finished.\n")


    #testing the results after learning is done
    total_epochs, total_penalties = 0, 0
    episodes = 100

    for _ in range(episodes):
        observation = env.reset()
        state = int(''.join(map(str, observation)))
        epochs, penalties, reward = 0, 0, 0
        
        done = False
        
        while not done:
            action = np.argmax(q_table[state])
            next_observation, reward, done, info = env.step(action)
            #for jupyter
            #clear_output(wait=True)
            #for CMD Windows
    #        system('cls')
    #        env.render()
    #        sleep(.1)
            if reward == -200:
                penalties += 1
            epochs += 1
        total_penalties += penalties
        total_epochs += epochs
    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")