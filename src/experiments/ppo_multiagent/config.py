from soccer_twos import EnvType
from ray import tune
from callback import Callback
from utils import create_custom_env, create_rllib_env

# NUM_ENVS_PER_WORKER = 1
NUM_ENVS_PER_WORKER = 4
ENVIRONMENT_ID = "Soccer"

ENVIRONMENT_CONFIG = {
    "num_envs_per_worker": NUM_ENVS_PER_WORKER,
    "variation": EnvType.multiagent_player,
}


temp_env = create_custom_env(ENVIRONMENT_CONFIG)
obs_space = temp_env.observation_space
act_space = temp_env.action_space
temp_env.close()


config = {
    # system settings
    "num_gpus": 1,
    # "num_workers": 3,
    "num_workers": 0,
    "num_envs_per_worker": NUM_ENVS_PER_WORKER,
    "num_cpus_for_driver": 8,
    "num_cpus_per_worker": 1,
    "num_gpus_per_worker": 1,
    "log_level": "INFO",
    "framework": "torch",
    # RL setup
    "multiagent": {
        "policies": {
            "default": (None, obs_space, act_space, {}),
        },
        "policy_mapping_fn": tune.function(lambda _: "default"),
        "policies_to_train": ["default"],
    },
    "env": ENVIRONMENT_ID,
    "env_config": ENVIRONMENT_CONFIG,
    "callbacks": Callback,
}
