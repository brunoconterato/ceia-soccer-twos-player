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
    def on_episode_step(self,
                        *,
                        worker: "RolloutWorker",
                        base_env: BaseEnv,
                        episode: MultiAgentEpisode,
                        env_index: Optional[int] = None,
                        **kwargs) -> None:
        total_timesteps = episode.last_info_for(
            0)["ep_metrics"]["total_timesteps"]
        total_goals = float(episode.last_info_for(0)[
                            "ep_metrics"]["total_goals"])
        estimated_goals_in_match = total_goals * MATCH_STEPS / \
            float(total_timesteps) if total_goals > 0 else 0.0
        timesteps_to_goal = float(
            total_timesteps) if total_goals > 0 else 9999.0

        if not episode.user_data:
            episode.user_data = {
                0: {
                    "total_env_reward": 0.0,
                    "total_ball_to_goal_speed_reward": 0.0,
                    "total_agent_position_to_ball_reward": 0.0,
                },
                1: {
                    "total_env_reward": 0.0,
                    "total_ball_to_goal_speed_reward": 0.0,
                    "total_agent_position_to_ball_reward": 0.0,
                },
                2: {
                    "total_env_reward": 0.0,
                    "total_ball_to_goal_speed_reward": 0.0,
                    "total_agent_position_to_ball_reward": 0.0,
                },
                3: {
                    "total_env_reward": 0.0,
                    "total_ball_to_goal_speed_reward": 0.0,
                    "total_agent_position_to_ball_reward": 0.0,
                }
            }

        episode.user_data = {
            **episode.user_data,
            0: {
                "total_env_reward": episode.user_data[0]["total_env_reward"] + episode.last_info_for(0)["ep_metrics"]["env_reward"],
                "total_ball_to_goal_speed_reward": episode.user_data[0]["total_ball_to_goal_speed_reward"] + episode.last_info_for(0)["ep_metrics"]["ball_to_goal_speed_reward"],
                "total_agent_position_to_ball_reward": episode.user_data[0]["total_agent_position_to_ball_reward"] + episode.last_info_for(0)["ep_metrics"]["agent_position_to_ball_reward"],
            },
            1: {
                "total_env_reward": episode.user_data[1]["total_env_reward"] + episode.last_info_for(1)["ep_metrics"]["env_reward"],
                "total_ball_to_goal_speed_reward": episode.user_data[1]["total_ball_to_goal_speed_reward"] + episode.last_info_for(1)["ep_metrics"]["ball_to_goal_speed_reward"],
                "total_agent_position_to_ball_reward": episode.user_data[1]["total_agent_position_to_ball_reward"] + episode.last_info_for(1)["ep_metrics"]["agent_position_to_ball_reward"],
            },
            2: {
                "total_env_reward": episode.user_data[2]["total_env_reward"] + episode.last_info_for(2)["ep_metrics"]["env_reward"],
                "total_ball_to_goal_speed_reward": episode.user_data[2]["total_ball_to_goal_speed_reward"] + episode.last_info_for(2)["ep_metrics"]["ball_to_goal_speed_reward"],
                "total_agent_position_to_ball_reward": episode.user_data[2]["total_agent_position_to_ball_reward"] + episode.last_info_for(2)["ep_metrics"]["agent_position_to_ball_reward"],
            },
            3: {
                "total_env_reward": episode.user_data[3]["total_env_reward"] + episode.last_info_for(3)["ep_metrics"]["env_reward"],
                "total_ball_to_goal_speed_reward": episode.user_data[3]["total_ball_to_goal_speed_reward"] + episode.last_info_for(3)["ep_metrics"]["ball_to_goal_speed_reward"],
                "total_agent_position_to_ball_reward": episode.user_data[3]["total_agent_position_to_ball_reward"] + episode.last_info_for(3)["ep_metrics"]["agent_position_to_ball_reward"],
            }
        }

        episode.custom_metrics = {
            # "total_timesteps": total_timesteps,
            # "timesteps_to_goal": timesteps_to_goal,
            # "estimated_goals_in_match": estimated_goals_in_match,
            # "team_0_goals": episode.last_info_for(0)["ep_metrics"]["team_0_goals"],
            # "team_1_goals": episode.last_info_for(0)["ep_metrics"]["team_1_goals"],
            # "have_goals": episode.last_info_for(0)["ep_metrics"]["have_goals"],
            "agent_0_total_env_reward": episode.user_data[0]["total_env_reward"],
            "agent_0_total_ball_to_goal_speed_reward": episode.user_data[0]["total_ball_to_goal_speed_reward"],
            "agent_0_total_agent_position_to_ball_reward": episode.user_data[0]["total_agent_position_to_ball_reward"],
        }

    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:
        total_timesteps = episode.last_info_for(
            0)["ep_metrics"]["total_timesteps"]
        total_goals = float(episode.last_info_for(0)[
                            "ep_metrics"]["total_goals"])
        estimated_goals_in_match = total_goals * MATCH_STEPS / \
            float(total_timesteps) if total_goals > 0 else 0.0
        timesteps_to_goal = float(
            total_timesteps) if total_goals > 0 else 9999.0

        if not episode.user_data:
            episode.user_data = {
                0: {
                    "total_env_reward": 0.0,
                    "total_ball_to_goal_speed_reward": 0.0,
                    "total_agent_position_to_ball_reward": 0.0,
                },
                1: {
                    "total_env_reward": 0.0,
                    "total_ball_to_goal_speed_reward": 0.0,
                    "total_agent_position_to_ball_reward": 0.0,
                },
                2: {
                    "total_env_reward": 0.0,
                    "total_ball_to_goal_speed_reward": 0.0,
                    "total_agent_position_to_ball_reward": 0.0,
                },
                3: {
                    "total_env_reward": 0.0,
                    "total_ball_to_goal_speed_reward": 0.0,
                    "total_agent_position_to_ball_reward": 0.0,
                }
            }

        episode.user_data = {
            **episode.user_data,
            0: {
                "total_env_reward": episode.user_data[0]["total_env_reward"] + episode.last_info_for(0)["ep_metrics"]["env_reward"],
                "total_ball_to_goal_speed_reward": episode.user_data[0]["total_ball_to_goal_speed_reward"] + episode.last_info_for(0)["ep_metrics"]["ball_to_goal_speed_reward"],
                "total_agent_position_to_ball_reward": episode.user_data[0]["total_agent_position_to_ball_reward"] + episode.last_info_for(0)["ep_metrics"]["agent_position_to_ball_reward"],
            },
            1: {
                "total_env_reward": episode.user_data[1]["total_env_reward"] + episode.last_info_for(1)["ep_metrics"]["env_reward"],
                "total_ball_to_goal_speed_reward": episode.user_data[1]["total_ball_to_goal_speed_reward"] + episode.last_info_for(1)["ep_metrics"]["ball_to_goal_speed_reward"],
                "total_agent_position_to_ball_reward": episode.user_data[1]["total_agent_position_to_ball_reward"] + episode.last_info_for(1)["ep_metrics"]["agent_position_to_ball_reward"],
            },
            2: {
                "total_env_reward": episode.user_data[2]["total_env_reward"] + episode.last_info_for(2)["ep_metrics"]["env_reward"],
                "total_ball_to_goal_speed_reward": episode.user_data[2]["total_ball_to_goal_speed_reward"] + episode.last_info_for(2)["ep_metrics"]["ball_to_goal_speed_reward"],
                "total_agent_position_to_ball_reward": episode.user_data[2]["total_agent_position_to_ball_reward"] + episode.last_info_for(2)["ep_metrics"]["agent_position_to_ball_reward"],
            },
            3: {
                "total_env_reward": episode.user_data[3]["total_env_reward"] + episode.last_info_for(3)["ep_metrics"]["env_reward"],
                "total_ball_to_goal_speed_reward": episode.user_data[3]["total_ball_to_goal_speed_reward"] + episode.last_info_for(3)["ep_metrics"]["ball_to_goal_speed_reward"],
                "total_agent_position_to_ball_reward": episode.user_data[3]["total_agent_position_to_ball_reward"] + episode.last_info_for(3)["ep_metrics"]["agent_position_to_ball_reward"],
            }
        }

        episode.custom_metrics = {
            # "total_timesteps": total_timesteps,
            # "timesteps_to_goal": timesteps_to_goal,
            # "estimated_goals_in_match": estimated_goals_in_match,
            # "team_0_goals": episode.last_info_for(0)["ep_metrics"]["team_0_goals"],
            # "team_1_goals": episode.last_info_for(0)["ep_metrics"]["team_1_goals"],
            # "have_goals": episode.last_info_for(0)["ep_metrics"]["have_goals"],
            "agent_0_total_env_reward": episode.user_data[0]["total_env_reward"],
            "agent_0_total_ball_to_goal_speed_reward": episode.user_data[0]["total_ball_to_goal_speed_reward"],
            "agent_0_total_agent_position_to_ball_reward": episode.user_data[0]["total_agent_position_to_ball_reward"],
        }
