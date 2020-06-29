#random walk in Frozen Lake, plus detection of end of episode

import gym
import random
import numpy

env = gym.make('FrozenLake-v0')
#env = gym.make('FrozenLake8x8-v0')


epsilon=0.1
gamma=0.9
alpha=0.01

Q = numpy.zeros([env.observation_space.n, env.action_space.n])
model=[]
for i in range(0,env.observation_space.n):
    model.append(0)
encounteredStates=[]

#sampling size for DynaQ
nDynaQ=5
nbEpisodes=50000
for i in range(0, nbEpisodes):
    state=env.reset()
    if i%100 == 0:
        print(i)
    
    firstState=state
    endOfEpisode = False
    nbSteps=0
    while not endOfEpisode:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = numpy.argmax(Q[state])

        next_state, reward, endOfEpisode, info = env.step(action) 
        model[state]=[action,reward,next_state]
        if encounteredStates.count(state)==0:
            encounteredStates.append(state)
        #print(encounteredStates)

        #if reward>0:
         #   print(reward)
          #  input("coucou")
        next_max = numpy.max(Q[next_state])

        new_value = (1 - alpha) * Q[state,action] + alpha * (reward + gamma * next_max)
        Q[state, action] = new_value

        for j in range(0,nDynaQ):
            selectedState=random.sample(encounteredStates,1)[0]
            #print(selectedState)
            [action,R,Sprime]=model[selectedState]
            #print([action,R,Sprime])
            Q[selectedState,action]=(1-alpha)*Q[selectedState,action]+alpha*(R+gamma*numpy.max(Q[Sprime]))
            
        state = next_state
        nbSteps=nbSteps+1
   

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



