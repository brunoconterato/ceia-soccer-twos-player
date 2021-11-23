from typing import List, Dict, Optional
# import ray
# from ray import tune
from ray.tune import Callback as Cb
from ray.tune.trial import Trial
from ray.rllib.env import BaseEnv
# from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.agents.callbacks import DefaultCallbacks



class Callback(DefaultCallbacks):
    def on_episode_step(self,
                        *,
                        worker: "RolloutWorker",
                        base_env: BaseEnv,
                        episode: MultiAgentEpisode,
                        env_index: Optional[int] = None,
                        **kwargs
                        ):
        pass
        # print(f"Last reward: {episode.prev_reward_for(0)}")
        # print(f"Total reward: {episode.total_reward}")

    def on_step_end(self, iteration: int, trials: List[Trial], **info):
        pass
