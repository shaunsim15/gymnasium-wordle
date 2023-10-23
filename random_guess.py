import gymnasium as gym
import gym_wordle
from gym_wordle.exceptions import InvalidWordException

env = gym.make('Wordle-v0')

obs = env.reset()
done = False
while not done:
    while True:
        try:
            # make a random guess
            act = env.action_space.sample()

            # take a step
            obs, reward, done, _ = env.step(act)
            break
        except InvalidWordException:
            pass

    env.render()