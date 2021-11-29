import ray
from ray import tune

from utils import create_custom_env
from config import ENVIRONMENT_ID, config
from stop import stop

 
def run_experiment():
    ray.init(num_cpus=8, include_dashboard=False)

    tune.registry.register_env(ENVIRONMENT_ID, create_custom_env)

    analysis = tune.run(
        "PPO",
        num_samples=1,
        # name="PPO_multiagent_player",
        name="Testing_pickle",
        config=config,
        stop=stop,
        checkpoint_freq=1,
        checkpoint_at_end=True,
        local_dir="./src/ray_results",
        # restore="./src/ray_results/PPO_selfplay_1/PPO_Soccer_ID/checkpoint_00X/checkpoint-X",
        # resume=True
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
    return analysis, best_trial, best_checkpoint


if __name__ == "__main__":
    analysis, best_trial, best_checkpoint = run_experiment()