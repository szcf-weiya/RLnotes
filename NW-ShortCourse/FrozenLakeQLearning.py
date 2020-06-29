#random walk in Frozen Lake, plus detection of end of episode

import gym
import random
import numpy

env = gym.make('FrozenLake-v0')

epsilon=0.1
gamma=0.9
alpha=0.01

Q = numpy.zeros([env.observation_space.n, env.action_space.n])

nbEpisodes=100000
for i in range(0, nbEpisodes):
    if i%100 == 0:
        print(i)
    state=env.reset()
    
    firstState=state
    endOfEpisode = False
    nbSteps=0
    while not endOfEpisode:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = numpy.argmax(Q[state])

        next_state, reward, endOfEpisode, info = env.step(action) 
        
        old_value = Q[state, action]
        next_max = numpy.max(Q[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        Q[state, action] = new_value

        state = next_state
        nbSteps=nbSteps+1

print(Q)

#evaluation
averageNumberSuccesses=0
for i in range(0, nbEpisodes):
    state=env.reset()
    
    endOfEpisode = False
    while not endOfEpisode:
        action = numpy.argmax(Q[state])

        next_state, reward, endOfEpisode, info = env.step(action) 

        state = next_state

    if reward==1:
        averageNumberSuccesses=averageNumberSuccesses+1
averageNumberSuccesses=averageNumberSuccesses/nbEpisodes

print(averageNumberSuccesses)



#evaluation random strategy
averageNumberSuccesses=0
for i in range(0, nbEpisodes):
    state=env.reset()
    
    endOfEpisode = False
    while not endOfEpisode:
        next_state, reward, endOfEpisode, info = env.step(env.action_space.sample()) 

        state = next_state

    if reward==1:
        averageNumberSuccesses=averageNumberSuccesses+1
averageNumberSuccesses=averageNumberSuccesses/nbEpisodes

print(averageNumberSuccesses)



