from typing import List, Any

from marl_comm.ma_policy import MAPolicyManager
from tianshou.env import PettingZooEnv
from tianshou.policy import BasePolicy


class QMIXPolicy(MAPolicyManager):
    def __init__(
        self,
        policies: List[BasePolicy],
        env: PettingZooEnv,
        train_scheme: str = "CTDE",
        parameter_mode: str = "Indvd",
        critic_mode: str = "IC",
        comm: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(
            policies, env, train_scheme, parameter_mode, critic_mode, comm, **kwargs
        )
