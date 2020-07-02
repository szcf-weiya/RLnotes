import numpy as np

import gym

# source: https://github.com/MeepMoop/tilecoding/blob/master/tilecoding.py
class TileCoder:
  def __init__(self, tiles_per_dim, value_limits, tilings, offset=lambda n: 2 * np.arange(n) + 1):
    tiling_dims = np.array(np.ceil(tiles_per_dim), dtype=np.int) + 1
    self._offsets = offset(len(tiles_per_dim)) * \
      np.repeat([np.arange(tilings)], len(tiles_per_dim), 0).T / float(tilings) % 1
    self._limits = np.array(value_limits)
    self._norm_dims = np.array(tiles_per_dim) / (self._limits[:, 1] - self._limits[:, 0])
    self._tile_base_ind = np.prod(tiling_dims) * np.arange(tilings)
    self._hash_vec = np.array([np.prod(tiling_dims[0:i]) for i in range(len(tiles_per_dim))])
    self._n_tiles = tilings * np.prod(tiling_dims)

  def __getitem__(self, x):
    off_coords = ((x - self._limits[:, 0]) * self._norm_dims + self._offsets).astype(int)
    return self._tile_base_ind + np.dot(off_coords, self._hash_vec)

  @property
  def n_tiles(self):
    return self._n_tiles

iter_max = 5000

gamma = 1.0
t_max = 10000
eps = 0.1
alpha = 0.05

def run_episode(env, T = None, w=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(t_max):
        if render:
            env.render()
        if T is None or w is None:
            action = env.action_space.sample()
        else:
            action = np.argmax(np.sum(w[T[obs], :], axis = 0))
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(0)
    np.random.seed(0)

    env_low = env.observation_space.low
    env_high = env.observation_space.high

    # tile coder
    T = TileCoder([9, 9], [(env_low[0], env_high[0]), (env_low[1], env_high[1])], 10)
    w = np.zeros((T.n_tiles, 3))
    for i in range(iter_max):
        obs = env.reset()
        total_reward = 0
        if np.random.uniform(0, 1) < eps:
            action = np.random.choice(env.action_space.n)
        else:
            action = np.argmax(np.sum(w[T[obs], :], axis=0))
        for j in range(t_max):
            obs_prime, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                w[T[obs], action] += alpha * (reward - np.sum(w[T[obs], action]))
                break
            if np.random.uniform(0, 1) < eps:
                action_prime = np.random.choice(env.action_space.n)
            else:
                action_prime = np.argmax(np.sum(w[T[obs_prime], :], axis=0))
            # update q via w
            # the index in the left hand side includes the partial of q
            # print(np.sum(w[T[obs_prime], action_prime]))
            w[T[obs], action] += alpha * (reward + gamma * np.sum(w[T[obs_prime], action_prime]) - np.sum(w[T[obs], action]))
            obs = obs_prime.copy()
            action = action_prime
        if i % 200 == 0:
            print('Iteration #%d -- Total reward = %d.' %(i+1, total_reward))
    solution_policy_scores = [run_episode(env, T, w, False) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    # Animate it
    for _ in range(2):
        run_episode(env, T, w, True)
    env.close()
