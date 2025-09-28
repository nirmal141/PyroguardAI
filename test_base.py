import time
from envs.wildfire_gym import WildfireGymEnv

env = WildfireGymEnv(grid_size=25)

obs, info = env.reset()

while env.running:
    obs, reward, terminated, truncated, info = env.step()
    env.render()
    time.sleep(0.15)

env.close()



