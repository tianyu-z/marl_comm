# pass test with newest version of pettingzoo and tianshou
import numpy as np
from tianshou.data import Batch, ReplayBuffer, VectorReplayBuffer


def test_replaybuffer_old():
    obs = np.arange(12)
    next_obs = np.arange(1, 13)
    buffer = ReplayBuffer(100)
    for i in range(100):
        batch = Batch(
            {
                "obs": {"obs": obs, "agent_id": f"agent_{i % 10}"},
                "next_obs": next_obs,
                "act": i,
                "rew": i**2,
                "done": i % 2,
            }
        )
        buffer.add(batch)
    print(buffer.sample(2))

    vbuffer = VectorReplayBuffer(100, 5)
    for i in range(100):
        batch = Batch(
            {
                "obs": {"obs": [obs], "agent_id": [f"agent_{i % 10}"]},
                "next_obs": [next_obs],
                "act": [i],
                "rew": [i**2],
                "done": [i % 2],
            }
        )
        vbuffer.add(batch, [i % 5])
    print(vbuffer.sample(2))


def test_replaybuffer_new():
    obs = np.arange(12)
    next_obs = np.arange(1, 13)
    buffer = ReplayBuffer(100)
    for i in range(100):
        batch = Batch(
            {
                "obs": {"obs": obs, "agent_id": f"agent_{i % 10}"},
                "next_obs": next_obs,
                "act": i,
                "rew": i**2,
                "terminated": i % 2,
                "truncated": i % 2
                # When an agent's episode is truncated, the agent will also no longer take any actions or receive any rewards in the current episode, but it didn't reach a natural endpoint.
            }
        )
        buffer.add(batch)
    print(buffer.sample(2))

    vbuffer = VectorReplayBuffer(100, 5)
    for i in range(100):
        batch = Batch(
            {
                "obs": {"obs": [obs], "agent_id": [f"agent_{i % 10}"]},
                "next_obs": [next_obs],
                "act": [i],
                "rew": [i**2],
                "terminated": [i % 2],
                "truncated": [i % 2],
                # When an agent's episode is truncated, the agent will also no longer take any actions or receive any rewards in the current episode, but it didn't reach a natural endpoint.
            }
        )
        vbuffer.add(batch, [i % 5])
    print(vbuffer.sample(2))


if __name__ == "__main__":
    test_replaybuffer_new()
