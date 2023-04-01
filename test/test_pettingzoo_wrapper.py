# pass test with newest version of pettingzoo and tianshou

import pettingzoo.butterfly.pistonball_v6 as pistonball_v6
from tianshou.env import BaseVectorEnv, SubprocVectorEnv
import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root)
from marl_comm.env import MAEnvWrapper, get_MA_VectorEnv_cls

# single env
env = MAEnvWrapper(pistonball_v6.env(continuous=False, n_pistons=2))
print("env len", len(env))  # 2
obs = env.reset()
# print("init obs:", obs)
act = env.action_space.sample()
print("act:", act)
x = env.step(act)
if len(x) == 4:
    obs, rew, term, info = env.step(act)
elif len(x) == 5:
    obs, rew, term, trunc, info = env.step(act)
    print("trunc:", trunc)
# print("next_obs:", obs)
print("reward:", rew)
print("term:", term)

print("info:", info)
print("action space", env.action_space)
print("observation space", env.observation_space)

# multiple envs


def get_env():
    return MAEnvWrapper(pistonball_v6.env(continuous=False, n_pistons=2))


ma_venv_cls = get_MA_VectorEnv_cls(SubprocVectorEnv)

venv = ma_venv_cls([get_env for _ in range(3)])

print("venv len", len(venv))  # 6
obs = venv.reset()
# print("init obs:", obs)
act = [act_sp.sample() for act_sp in venv.get_env_attr("action_space")]
print("act:", act)
if len(venv.step(act)) == 4:
    obs, rew, term, info = venv.step(act)
elif len(venv.step(act)) == 5:
    obs, rew, term, trunc, info = venv.step(act)
    print("trunc:", trunc)
# print("next_obs:", obs)
print("reward:", rew)
print("term:", term)

print("info:", info)
print("action space", venv.action_space)
print("observation space", venv.observation_space)
