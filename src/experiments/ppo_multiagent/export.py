from ray.tune import Analysis
import os
import shutil

this_path = os.path.dirname(os.path.abspath(__file__))
print('this_path', this_path)

def export_agent(agent_file: str, TRIAL, agent_name="my_ray_soccer_agent", makeZip=False):
    agent_path = os.path.join(f'{this_path}/agents', agent_name)
    os.makedirs(agent_path, exist_ok=True)

    shutil.rmtree(agent_path)
    os.makedirs(agent_path)

    # salva a classe do agente
    with open(os.path.join(agent_path, "agent.py"), "w") as f:
        f.write(agent_file)

    # salva um __init__ para criar o módulo Python
    with open(os.path.join(agent_path, "__init__.py"), "w") as f:
        f.write("from .agent import MyRaySoccerAgent")

    # copia o trial inteiro, incluindo os arquivos de configuração do experimento
    print(f"TRIALLL {TRIAL}")
    shutil.copytree(TRIAL, os.path.join(agent_path, TRIAL.split("ray_results/")[1]), )

    # empacota tudo num arquivo .zip
    if makeZip:
        shutil.make_archive(os.path.join(agent_path, agent_name),
                            "zip", os.path.join(agent_path, agent_name))

def get_agent_file_str(ALGORITHM, CHECKPOINT, POLICY_NAME="default"):
    return f"""
import pickle
import os
from typing import Dict

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.env.base_env import BaseEnv
from ray.tune.registry import get_trainable_cls

from soccer_twos import AgentInterface

ALGORITHM = "{ALGORITHM}"
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "{CHECKPOINT.split("ray_results/")[1]}"
)
POLICY_NAME = "{POLICY_NAME}"


class MyRaySoccerAgent(AgentInterface):
    def __init__(self, env: gym.Env):
        super().__init__()
        ray.init(ignore_reinit_error=True)

        # Load configuration from checkpoint file.
        config_path = ""
        if CHECKPOINT_PATH:
            config_dir = os.path.dirname(CHECKPOINT_PATH)
            config_path = os.path.join(config_dir, "params.pkl")
            # Try parent directory.
            if not os.path.exists(config_path):
                config_path = os.path.join(config_dir, "../params.pkl")

        # Load the config from pickled.
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
        else:
            # If no config in given checkpoint -> Error.
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory!"
            )

        # no need for parallelism on evaluation
        config["num_workers"] = 0
        config["num_gpus"] = 0

        # create a dummy env since it's required but we only care about the policy
        tune.registry.register_env("DummyEnv", lambda *_: BaseEnv())
        config["env"] = "DummyEnv"

        # create the Trainer from config
        cls = get_trainable_cls(ALGORITHM)
        agent = cls(env=config["env"], config=config)
        # load state from checkpoint
        agent.restore(CHECKPOINT_PATH)
        # get policy for evaluation
        self.policy = agent.get_policy(POLICY_NAME)

    def act(self, observation: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        actions = {{}}
        for player_id in observation:
            # compute_single_action returns a tuple of (action, action_info, ...)
            # as we only need the action, we discard the other elements
            actions[player_id], *_ = self.policy.compute_single_action(
                observation[player_id]
            )
        return actions

"""

def getAnalysis(experiment: str):
    return Analysis(experiment)

def export():
    # PPO_Soccer_18d23_00000
    # /home/bruno/Workspace/soccer-tows-player/src/ray_results/Testing_env/PPO_Soccer_18d23_00000_0_2021-11-24_20-34-41/checkpoint_000500/checkpoint-500
    analysis = getAnalysis("/home/bruno/Workspace/soccer-tows-player/src/ray_results/PPO_multiagent_player_custom_rewards")

    
    ALGORITHM = "PPO"
    TRIAL = analysis.get_best_logdir("training_iteration", "max")
    CHECKPOINT = analysis.get_best_checkpoint(
        TRIAL,
        "training_iteration",
        "max",
    )

    print(TRIAL, CHECKPOINT)
    agent_file = get_agent_file_str(ALGORITHM, CHECKPOINT)
    export_agent(agent_file, TRIAL)

if __name__ == "__main__":
    export()