#FrozenLake with "Deep (not so deep) Q-learning"

import gym
import random
import numpy
import math

env = gym.make('FrozenLake-v0')

epsilon=0.1
gamma=0.9
alpha=0.5

Q = numpy.zeros([env.observation_space.n, env.action_space.n])

def sigmoid(x):
    if x>100:
        returnValue=1
    elif x<-100:
        returnValue=-1
    else:
        returnValue=(math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
    return returnValue
        
class NN:
    def __init__(self,sizeInput,sizeHiddenLayer,sizeOutput):
        self.sizeInput=sizeInput
        self.sizeHiddenLayer=sizeHiddenLayer
        self.sizeOutput=sizeOutput

        #below are the weights
        self.HiddenLayerEntryWeights=numpy.zeros([sizeHiddenLayer,sizeInput])
        self.LastLayerEntryWeights=numpy.zeros([sizeOutput,sizeHiddenLayer])

        #random initialization
        for i in range(0,sizeHiddenLayer):
            for j in range(0,sizeInput):
                self.HiddenLayerEntryWeights[i,j]=random.uniform(-0.1,0.1)
                
        for i in range(0,sizeOutput):
            for j in range(0,sizeHiddenLayer):
                self.LastLayerEntryWeights[i,j]=random.uniform(-0.1,0.1)

        self.HiddenLayerEntryDeltas=numpy.zeros(sizeHiddenLayer)
        self.LastLayerEntryDeltas=numpy.zeros(sizeOutput)

        self.HiddenLayerOutput=numpy.zeros(sizeHiddenLayer)
        self.LastLayerOutput=numpy.zeros(sizeOutput)

    def output(self,x):
        for i in range(0, self.sizeHiddenLayer):
            self.HiddenLayerOutput[i]=sigmoid(numpy.dot(self.HiddenLayerEntryWeights[i],x))
        for i in range(0, self.sizeOutput):
            self.LastLayerOutput[i]= \
            sigmoid(numpy.dot(self.LastLayerEntryWeights[i],self.HiddenLayerOutput))

    def retropropagation(self,x,y,actionIndex):
        self.output(x)

        #deltas computation
        self.LastLayerEntryDeltas[actionIndex]=2*(self.LastLayerOutput[actionIndex]-y)* \
            (1+self.LastLayerOutput[actionIndex])*(1-self.LastLayerOutput[actionIndex])

        for i in range(0,self.sizeHiddenLayer):
            #here usually you need a sum
            self.HiddenLayerEntryDeltas[i]=self.LastLayerEntryDeltas[actionIndex]* \
            (1+self.HiddenLayerOutput[i])*(1-self.HiddenLayerOutput[i])*self.LastLayerEntryWeights[actionIndex,i]

        #weights update
        for i in range(0,self.sizeHiddenLayer):
            self.LastLayerEntryWeights[actionIndex,i]-=alpha*self.LastLayerEntryDeltas[actionIndex]* \
            self.HiddenLayerOutput[i]

        for i in range(0,self.sizeHiddenLayer):
            for j in range(0,self.sizeInput):
                self.HiddenLayerEntryWeights[i,j]-=alpha*self.HiddenLayerEntryDeltas[i]*x[j]

        
nbCells=16
sizeInput=nbCells
sizeHiddenLayer=10
sizeOutput=4
myNN = NN(sizeInput, sizeHiddenLayer, sizeOutput)

nbEpisodes=50000
for i in range(0, nbEpisodes):
    if i>5000:
        alpha=0.1
    if i>10000:
        alpha=0.05
    if i>20000:
        alpha=0.02
    if i>30000:
        alpha=0.01
    
    if i%100==0:
        print("episode: "+str(i))
        successesInARow=0

    state=env.reset()
    
    firstState=state
    endOfEpisode = False
    nbSteps=0

    while not endOfEpisode:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            x=numpy.zeros(sizeInput)
            x[state-1]=1
            myNN.output(x)
            action = numpy.argmax(myNN.LastLayerOutput)
            
        next_state, reward, endOfEpisode, info = env.step(action) 
      
        x=numpy.zeros(sizeInput)
        x[next_state-1]=1
        myNN.output(x)
        next_max = numpy.max(myNN.LastLayerOutput)

        target = reward+gamma*next_max

        x=numpy.zeros(sizeInput)
        x[state-1]=1

        if reward==1:
            successesInARow=successesInARow+1
            print("successesInARow: "+str(successesInARow))

        myNN.retropropagation(x,target,action)
        
        state = next_state
        nbSteps=nbSteps+1
print("end of learning period")

#evaluation
averageNumberSuccesses=0

nbEpisodes=10000
for i in range(0, nbEpisodes):
    if i%100==0:
        print("episode: "+str(i))
    state=env.reset()
    
    endOfEpisode = False
    while not endOfEpisode:
        x=numpy.zeros(sizeInput)
        x[state-1]=1
        myNN.output(x)
        action = numpy.argmax(myNN.LastLayerOutput)
        
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



