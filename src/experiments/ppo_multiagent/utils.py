import gym
from ray.rllib import MultiAgentEnv
import soccer_twos
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from wrapper import CustomRewardWrapper

# get_agent_and_policy imports
import os
import pickle
import ray
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env import BaseEnv
from ray.tune.registry import get_trainable_cls

class RLLibWrapper(gym.core.Wrapper, MultiAgentEnv):
    """
    A RLLib wrapper so our env can inherit from MultiAgentEnv.
    """

    pass


def create_rllib_env(env_config: dict = {}):
    """
    Creates a RLLib environment and prepares it to be instantiated by Ray workers.
    Args:
        env_config: configuration for the environment.
            You may specify the following keys:
            - variation: one of soccer_twos.EnvType. Defaults to EnvType.multiagent_player.
            - opponent_policy: a Callable for your agent to train against. Defaults to a random policy.
    """
    if hasattr(env_config, "worker_index"):
        env_config["worker_id"] = (
            env_config.worker_index * env_config.get("num_envs_per_worker", 1)
            + env_config.vector_index
        )
    env = soccer_twos.make(**env_config)
    if "multiagent" in env_config and not env_config["multiagent"]:
        # is multiagent by default, is only disabled if explicitly set to False
        return env
    return RLLibWrapper(env)


def create_custom_env(env_config: dict = {}):
    env = create_rllib_env(env_config)
    return CustomRewardWrapper(env)


def get_agent_and_policy(algorithm, checkpoint_path, policy_name):
    ray.init(ignore_reinit_error=True)
    config_path = ""
    if checkpoint_path:
        config_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(config_dir, "params.pkl")
        # Try parent directory.
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

    # Load the config from pickled.
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            cfg = pickle.load(f)
    else:
        # If no config in given checkpoint -> Error.
        raise ValueError(
            "Could not find params.pkl in either the checkpoint dir or "
            "its parent directory!"
        )

    # no need for parallelism on evaluation
    cfg["num_workers"] = 0
    cfg["num_gpus"] = 0
    cfg['evaluation_interval'] = 0

    # create a dummy env since it's required but we only care about the policy
    tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
    cfg["env"] = "DummyEnv"

    # create the Trainer from config
    # cls = get_trainable_cls(algorithm) if isinstance(algorithm, str) else algorithm
    trainer = PPOTrainer(config=cfg, env=cfg["env"])
    # agent = cls(env=cfg["env"], config=cfg)
    # load state from checkpoint
    trainer.restore(checkpoint_path)
    # get policy for evaluation
    policy = trainer.get_policy(policy_name)

    return trainer, policy


# baseline_trainer, baseline_policy = get_agent_and_policy("PPO",
#                                                      "/home/bruno/Workspace/soccer-tows-player/src/ray_results/PPO_selfplay_twos/PPO_Soccer_f475e_00000_0_2021-09-19_15-54-02/checkpoint_002449/checkpoint-2449",
#                                                      "default",
#                                                      )

# bajai_trainer, bajai_policy = get_agent_and_policy("PPO",
#                                                "/home/bruno/Workspace/soccer-tows-player/src/ray_results/PPO_multiagent_player_custom_rewards/PPO_Soccer_491c2_00000_0_2021-11-30_01-16-26/checkpoint_002617/checkpoint-2617",
#                                                "default")
