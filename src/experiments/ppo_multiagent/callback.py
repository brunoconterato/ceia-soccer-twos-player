from typing import Dict, Optional
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.utils.typing import AgentID, PolicyID
import numpy as np

MAX_STEPS = 1000
MATCH_STEPS = 4000

class Callback(DefaultCallbacks):
    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs):
        total_timesteps = episode.last_info_for(
            0)["ep_metrics"]["total_timesteps"]
        total_goals = float(episode.last_info_for(0)["ep_metrics"]["total_goals"])
        estimated_goals_in_match = total_goals * MATCH_STEPS / \
            float(total_timesteps) if total_goals > 0 else 0.0
        timesteps_to_goal = float(total_timesteps) if total_goals > 0 else 9999.0
        episode.custom_metrics = {
            "total_timesteps": total_timesteps,
            "timesteps_to_goal": timesteps_to_goal,
            "estimated_goals_in_match": estimated_goals_in_match,
            "team_0_goals": episode.last_info_for(0)["ep_metrics"]["team_0_goals"],
            "team_1_goals": episode.last_info_for(0)["ep_metrics"]["team_1_goals"],
            "have_goals": episode.last_info_for(0)["ep_metrics"]["have_goals"]
        }
