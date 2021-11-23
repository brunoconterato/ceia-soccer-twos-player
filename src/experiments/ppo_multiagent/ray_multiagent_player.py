import ray
from ray import tune

from src.utils import create_custom_env

from src.experiments.ppo_multiagent.config import ENVIRONMENT_ID, config
from src.experiments.ppo_multiagent.stop import stop
from src.experiments.ppo_multiagent.callback import Callback


def run_experiment():
    ray.init(num_cpus=8, include_dashboard=False)

    tune.registry.register_env(ENVIRONMENT_ID, create_custom_env)

    analysis = tune.run(
        "PPO",
        num_samples=1,
        name="PPO_multiagent_player",
        config=config,
        stop=stop,
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir="./src/ray_results",
        # callbacks=[Callback()]
        # restore="./src/ray_results/PPO_selfplay_1/PPO_Soccer_ID/checkpoint_00X/checkpoint-X",
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial("episode_reward_mean", mode="max")
    print(best_trial)
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        trial=best_trial, metric="episode_reward_mean", mode="max"
    )
    print(best_checkpoint)
    print("Done training")


if __name__ == "__main__":
    run_experiment()
