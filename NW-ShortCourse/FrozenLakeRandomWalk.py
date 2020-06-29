#random walk in Frozen Lake

import gym
env = gym.make('FrozenLake8x8-v0')

print(env.observation_space)
state=env.reset()
print(state)
for _ in range(10):
    env.render()
    input("#############################################")
    next_state, reward, endOfEpisode, info=env.step(env.action_space.sample())
    # take a random action
    
    print(next_state)
env.close()

