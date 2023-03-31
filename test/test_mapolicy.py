import argparse
import os
from typing import List, Optional, Tuple

import gym
import numpy as np
import pettingzoo.butterfly.pistonball_v6 as pistonball_v6
import torch
from tianshou.data import VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root)
from marl_comm.data import MACollector, MAReplayBuffer
from marl_comm.env import MAEnvWrapper, get_MA_VectorEnv
from marl_comm.ma_policy import MAPolicyManager


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--eps-test", type=float, default=0.05)
    parser.add_argument("--eps-train", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma",
                        type=float,
                        default=0.9,
                        help="a smaller gamma favors earlier win")
    parser.add_argument("--n-pistons",
                        type=int,
                        default=10,
                        help="Number of pistons(agents) in the env")
    parser.add_argument("--n-step", type=int, default=100)
    parser.add_argument("--target-update-freq", type=int, default=320)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--step-per-epoch", type=int, default=500)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--hidden-sizes",
                        type=int,
                        nargs="*",
                        default=[64, 64])
    parser.add_argument("--training-num", type=int, default=5)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=0.0)

    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="no training, "
        "watch the play of pre-trained models",
    )
    parser.add_argument("--device",
                        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


def get_env(args: argparse.Namespace = get_args()):
    return MAEnvWrapper(
        pistonball_v6.env(continuous=False, n_pistons=args.n_pistons))


def get_agents(
    args: argparse.Namespace = get_args(),
    agents: Optional[List[BasePolicy]] = None,
    optims: Optional[List[torch.optim.Optimizer]] = None,
) -> Tuple[BasePolicy, List[torch.optim.Optimizer], List]:

    env = get_env()
    observation_space = (env.observation_space["observation"] if isinstance(
        env.observation_space, gym.spaces.Dict) else env.observation_space)
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n

    if agents is None:
        agents = []
        optims = []
        for _ in range(args.n_pistons):
            # model
            net = Net(
                args.state_shape,
                args.action_shape,
                hidden_sizes=args.hidden_sizes,
                device=args.device,
            ).to(args.device)
            optim = torch.optim.Adam(net.parameters(), lr=args.lr)
            agent = DQNPolicy(
                net,
                optim,
                args.gamma,
                args.n_step,
                target_update_freq=args.target_update_freq,
            )
            agents.append(agent)
            optims.append(optim)

    policy = MAPolicyManager(agents, env, train_scheme="FD")
    return policy, optims, env.agents


def get_buffer(args: argparse.Namespace = get_args()):
    env = get_env()
    return MAReplayBuffer(args.buffer_size, env.agents, VectorReplayBuffer,
                          args.training_num)


def train_agent(
    args: argparse.Namespace = get_args(),
    agents: Optional[List[BasePolicy]] = None,
    optims: Optional[List[torch.optim.Optimizer]] = None,
) -> Tuple[dict, BasePolicy]:
    train_envs = get_MA_VectorEnv(SubprocVectorEnv,
                                  [get_env for _ in range(args.training_num)])
    test_envs = get_MA_VectorEnv(SubprocVectorEnv,
                                 [get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    policy, optim, agents = get_agents(args, agents=agents, optims=optims)

    # collector
    train_collector = MACollector(policy,
                                  train_envs,
                                  get_buffer(args),
                                  exploration_noise=True)
    test_collector = MACollector(policy, test_envs)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, "pistonball", "dqn")
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy):
        pass

    def stop_fn(mean_rewards):
        return False

    def train_fn(epoch, env_step):
        [agent.set_eps(args.eps_train) for agent in policy.policies.values()]

    def test_fn(epoch, env_step):
        [agent.set_eps(args.eps_test) for agent in policy.policies.values()]

    def reward_metric(rews):
        return rews[0]

    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=args.update_per_step,
        logger=logger,
        test_in_train=False,
        reward_metric=reward_metric,
    )

    return result, policy


def watch(args: argparse.Namespace = get_args(),
          policy: Optional[BasePolicy] = None) -> None:
    env = get_MA_VectorEnv(DummyVectorEnv, [get_env])
    policy.eval()
    [agent.set_eps(args.eps_test) for agent in policy.policies.values()]
    collector = MACollector(policy, env)
    result = collector.collect(n_episode=1, render=args.render)
    # print(collector.data)
    rews, lens = result["rews"], result["lens"]
    print(f"Final reward: {rews[0].mean()}, length: {lens.mean()}")


def test_piston_ball(args=get_args()):
    import pprint

    if args.watch:
        watch(args)
        return

    result, agent = train_agent(args)
    # assert result["best_reward"] >= args.win_rate

    if __name__ == "__main__":
        pprint.pprint(result)
        # Let's watch its performance!
        watch(args, agent)


if __name__ == "__main__":
    test_piston_ball(get_args())
