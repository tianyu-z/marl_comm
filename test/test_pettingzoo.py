# pass test with newest version of pettingzoo and tianshou
import pettingzoo.butterfly.pistonball_v6 as pistonball_v6
import pettingzoo.mpe.simple_push_v2 as simple_push_v2
import supersuit as ss
from tianshou.env import SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv

# oenv = pistonball_v6.env(continuous=False, n_pistons=5)

env = PettingZooEnv(pistonball_v6.env(continuous=False, n_pistons=5))

env.reset()
for i in range(10000):
    x = env.step(env.action_space.sample())
    if len(x) == 4:
        obs, _, done, info = env.step(env.action_space.sample())
    elif len(x) == 5:
        obs, _, done, trunc, info = env.step(env.action_space.sample())
    if done:
        print(info)
        break

# def get_env():
#     return PettingZooEnv(pistonball_v6.env(continuous=False, n_pistons=2))

# venv = SubprocVectorEnv([get_env for _ in range(2)])

# env = PettingZooEnv(
#     ss.pad_observations_v0(ss.pad_observations_v0(simple_push_v2.env())))

# def get_env():
#     return PettingZooEnv(
#         ss.pad_observations_v0(ss.pad_observations_v0(simple_push_v2.env())))

# venv = SubprocVectorEnv([get_env for _ in range(2)])
