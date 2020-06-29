#random walk in Frozen Lake, plus detection of end of episode

import gym
import random
import numpy
import math

env = gym.make('FrozenLake-v0')

epsilon=0.1
gamma=0.9
alpha=0.01

Q = numpy.zeros([env.observation_space.n, env.action_space.n])

#21% en 50000 iterations et n=5 contre 100000 pour n=1
#sampling size for pSarsa
n=5

nbEpisodes=30000
for i in range(0, nbEpisodes):
    #epsilon=0.1-(0.1-0.01)*i/nbEpisodes
    #alpha=0.02-(0.02-0.001)*i/nbEpisodes

    statesHistory=[]
    rewardsHistory=[]
    actionsHistory=[]
    
    state=env.reset()
    if i%100 == 0:
        print(i)
    
    firstState=state
    statesHistory.append(state)
    rewardsHistory.append(0)
    
    endOfEpisode = False
    t=0
    tau=0
    T=math.inf

    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = numpy.argmax(Q[state])
    actionsHistory.append(action)
        
    while tau<(T-1):
        if t<T:
            next_state, reward, endOfEpisode, info = env.step(action) 
            statesHistory.append(next_state)
            rewardsHistory.append(reward)

            if endOfEpisode:
#                if reward>0:
 #                   input("trouve")
                T=t+1
            else:
                if random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = numpy.argmax(Q[state])
                actionsHistory.append(action)
        tau=t-n+1

        if tau>=0:
            #print("#####################################")
            #print(T)
            #print(tau)
            #print(len(actionsHistory))
            G=0
            for j in range(tau+1,min(tau+n,T)+1):
                G=G+math.pow(gamma,j-tau-1)*rewardsHistory[j]
            if (tau+n)<T:
#                print("toto")
                G=G+math.pow(gamma,n)*Q[statesHistory[tau+n],actionsHistory[tau+n]]

            stateTau=statesHistory[tau]
            actionTau=actionsHistory[tau]
            Q[stateTau,actionTau] = (1 - alpha) * Q[stateTau,actionTau] + alpha * G

        state = next_state
        t=t+1
   
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



