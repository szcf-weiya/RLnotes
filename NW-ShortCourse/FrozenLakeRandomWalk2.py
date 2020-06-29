#random walk in Frozen Lake, plus detection of end of episode

import gym
env = gym.make('FrozenLake-v0')
env.reset()
endOfEpisode = False
while not endOfEpisode:
    env.render()
    input("#############################################")
    next_state, reward, endOfEpisode, info = env.step(env.action_space.sample())
    print(reward)
    
env.close()
