import gym
from typing import Any, Dict, List, Union

from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from collections import deque

from callback import MAX_STEPS


def get_scalar_projection(x, y):
    return np.dot(x, y) / np.linalg.norm(y)


# Os seguintes valores foram obtidos experimentalmente executando pré-experimentos
# A partir desses valores vamops derivar vários outros como posições ddos gols etc
min_ball_position_x, max_ball_position_x = - \
    15.563264846801758, 15.682827949523926
min_ball_position_y, max_ball_position_y = -7.08929967880249, 7.223850250244141
min_player_position_x, max_player_position_x = - \
    17.26804542541504, 17.16301727294922
min_player_position_y, max_player_position_y = - \
    7.399587631225586, 7.406457424163818
min_ball_to_goal_avg_velocity, max_ball_to_goal_avg_velocity = - \
    -23.366606239568615, 23.749571761530724

max_ball_abs_velocity = 78.25721740722656
max_goals_one_team = -9999999
max_goals_one_match = -9999999
max_steps = -999999

max_diff_reward = -np.inf

# Infered
max_ball_abs_avg_velocity = max(
    abs(min_ball_to_goal_avg_velocity), abs(max_ball_to_goal_avg_velocity))


SPEED_IMPORTANCE = 1.0 / (14.0)
CLIP_SPEED_REWARD_BY_SPEED_IMPORTANCE = True

AFTER_BALL_STEP_PENALTY = 1 / MAX_STEPS #0.001

# OBS.: Este hyperparâmetro não pode ser modificado sem fazer novos testes em
# min_ball_to_goal_avg_velocity e
# max_ball_to_goal_avg_velocity:
AVG_SPEED_TIMESTEPS_WINDOW = 1


def is_after_the_ball(player_id: int, player_pos: np.array, ball_pos: np.array):
    if player_id in range(2):
        return player_pos[0] > ball_pos[0]
    elif player_id in [2, 3]:
        return player_pos[0] < ball_pos[0]


def get_center_of_goal_pos(player_id):
    global min_ball_position_x, max_ball_position_x, \
        min_ball_position_y, max_ball_position_y, \
        min_player_position_x, max_player_position_x, \
        min_player_position_y, max_player_position_y
    if player_id in [0, 1]:
        return np.array([max_ball_position_x, 0.0])
    elif player_id in [2, 3]:
        return np.array([min_ball_position_x, 0.0])


def calculate_ball_to_goal_scalar_velocity(player_id: int, info: Dict):
    goal_pos = get_center_of_goal_pos(player_id)
    # print(f"goal_pos: {goal_pos}")
    ball_pos = info["ball_info"]["position"]
    # print(f"ball_pos: {ball_pos}")
    direction_to_center_of_goal = goal_pos - ball_pos
    # print(f"direction_to_center_of_goal: {direction_to_center_of_goal}")

    ball_velocity = info["ball_info"]["velocity"]

    # global max_ball_abs_velocity
    # if np.linalg.norm(ball_velocity) > max_ball_abs_velocity:
    #     max_ball_abs_velocity = np.linalg.norm(ball_velocity)

    # print(f"ball_velocity: {ball_velocity}")
    ball_velocity_to_center_of_goal = get_scalar_projection(
        ball_velocity, direction_to_center_of_goal)
    # print(f"ball_velocity_to_center_of_goal: {ball_velocity_to_center_of_goal}")
    return ball_velocity_to_center_of_goal

# print('ball_velocity_to_center_of_goal', calculate_ball_to_goal_scalar_velocity(0, { "ball_info": { "position": np.array([3.0, 2.0]), "velocity": np.array([0.0, 0.0]) }}))


class CustomRewardWrapper(gym.core.Wrapper, MultiAgentEnv):
    # def __init__(self, env):
    #     gym.Wrapper.__init__(self, env)

    def step(self, action: Union[Dict[int, List[Any]], List[Any]]):
        obs, rewards, done, info = super().step(action)

        # print(info)
        # if rewards[0] > 0.0:
        #     assert False

        if type(action) is dict:
            new_rewards = {k: self._calculate_reward(
                rewards[k], k, info[k]) for k in info.keys()}
        else:
            raise NotImplementedError('Necessário implementar!')

        if type(action) is dict:
            splitted_rets = {k: self._calculate_reward(
                rewards[k], k, info[k], splitted_returns=True) for k in info.keys()}
        else:
            raise NotImplementedError('Necessário implementar!')


        info = {
            i: {
                **info[i],
                "ep_metrics": {
                    # "total_timesteps": np.array([0.0008], dtype=np.float32)
                    "total_timesteps": self.n_step + 1,
                    "total_goals": self.scoreboard["team_0"] + self.scoreboard["team_1"],
                    "goals_opponent": self.scoreboard["team_1"] if i in range(2) else self.scoreboard["team_0"],
                    "goals_in_favor": self.scoreboard["team_0"] if i in range(2) else self.scoreboard["team_1"],
                    "team_0_goals": self.scoreboard["team_0"],
                    "team_1_goals": self.scoreboard["team_1"],
                    "episode_ended": done["__all__"],
                    "have_goals": self.scoreboard["team_0"] + self.scoreboard["team_1"] > 0,
                    "env_reward": splitted_rets[i][0],
                    "ball_to_goal_speed_reward": splitted_rets[i][1],
                    "agent_position_to_ball_reward": splitted_rets[i][2],
                }
            } for i in info.keys()
        }

        # global min_ball_position_x, max_ball_position_x, \
        #     min_ball_position_y, max_ball_position_y, \
        #     min_player_position_x, max_player_position_x, \
        #     min_player_position_y, max_player_position_y, \
        #     max_goals_one_team, max_goals_one_match
        # if done:
        #     print(f'min_ball_position_x: {min_ball_position_x}')
        #     print(f'max_ball_position_x: {max_ball_position_x}')
        #     print(f'min_ball_position_y: {min_ball_position_y}')
        #     print(f'max_ball_position_y: {max_ball_position_y}')
        #     print(f'min_player_position_x: {min_player_position_x}')
        #     print(f'max_player_position_x: {max_player_position_x}')
        #     print(f'min_player_position_y: {min_player_position_y}')
        #     print(f'max_player_position_y: {max_player_position_y}')
        #     print(f'min_ball_to_goal_avg_velocity: {min_ball_to_goal_avg_velocity}')
        #     print(f'max_ball_to_goal_avg_velocity: {max_ball_to_goal_avg_velocity}')
        #     print(f'max_goals_one_team: {max_goals_one_team}')
        #     print(f'max_goals_one_match: {max_goals_one_match}')
        #     print(self.scoreboard)
        #     print(f'Done... last n_step: {self.n_step}')
        #     if self.scoreboard["team_0"] > 0 or self.scoreboard["team_1"] > 0:
        #         input("Press Enter to continue...")

        # global max_steps
        # if done:
        #     if self.n_step + 1 > max_steps:
        #         max_steps = self.n_step + 1
        #     print('max_steps', max_steps)

        # global max_diff_reward
        # if done:
        #     print(f'max_diff_reward: {max_diff_reward}')
        #     print(f'min_ball_to_goal_avg_velocity: {min_ball_to_goal_avg_velocity}')
        #     print(f'max_ball_to_goal_avg_velocity: {max_ball_to_goal_avg_velocity}')

        # if done:
        #     print(f'max_ball_abs_velocity: {max_ball_abs_velocity}')

        self.n_step += 1
        return obs, new_rewards, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.n_step = 0
        self.last_ball_speed_mean_per_player = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.ball_speed_deque_per_player = {0: deque(maxlen=AVG_SPEED_TIMESTEPS_WINDOW),
                                            1: deque(maxlen=AVG_SPEED_TIMESTEPS_WINDOW),
                                            2: deque(maxlen=AVG_SPEED_TIMESTEPS_WINDOW),
                                            3: deque(maxlen=AVG_SPEED_TIMESTEPS_WINDOW)}
        self.scoreboard = {"team_0": 0, "team_1": 0}
        self.await_press = False
        # print(f'min_ball_to_goal_avg_velocity: {min_ball_to_goal_avg_velocity}')
        # print(f'max_ball_to_goal_avg_velocity: {max_ball_to_goal_avg_velocity}')
        return obs

    def _calculate_reward(self, reward: float, player_id: int, info: Dict, splitted_returns=False) -> float:
        # print('calculating reward')
        if reward != 0.0:
            # print('Goal was made!', reward, info)
            self._update_scoreboard(player_id, reward)
        # global min_ball_position_x, max_ball_position_x, \
        #     min_ball_position_y, max_ball_position_y, \
        #     min_player_position_x, max_player_position_x, \
        #     min_player_position_y, max_player_position_y
        # print(f"info: {info}")
        # if info["ball_info"]["position"][0] < min_ball_position_x:
        #     min_ball_position_x = info["ball_info"]["position"][0]
        # if info["ball_info"]["position"][0] > max_ball_position_x:
        #     max_ball_position_x = info["ball_info"]["position"][0]
        # if info["ball_info"]["position"][1] < min_ball_position_y:
        #     min_ball_position_y = info["ball_info"]["position"][1]
        # if info["ball_info"]["position"][1] > max_ball_position_y:
        #     max_ball_position_y = info["ball_info"]["position"][1]
        # if info["player_info"]["position"][0] < min_player_position_x:
        #     min_player_position_x = info["player_info"]["position"][0]
        # if info["player_info"]["position"][0] > max_player_position_x:
        #     max_player_position_x = info["player_info"]["position"][0]
        # if info["player_info"]["position"][1] < min_player_position_y:
        #     min_player_position_y = info["player_info"]["position"][1]
        # if info["player_info"]["position"][1] > max_player_position_y:
        #     max_player_position_y = info["player_info"]["position"][1]

        self._update_avg_ball_speed_to_goal(
            player_id, calculate_ball_to_goal_scalar_velocity(player_id, info))
        # global max_diff_reward
        # if (np.abs(SPEED_IMPORTANCE * self.last_ball_speed_mean_per_player[player_id] / max_ball_abs_avg_velocity) > max_diff_reward):
        #     max_diff_reward = SPEED_IMPORTANCE * \
        #         self.last_ball_speed_mean_per_player[player_id] / \
        #         max_ball_abs_avg_velocity

        ball_pos = info["ball_info"]["position"]
        player_pos = info["player_info"]["position"]

        env_reward = reward
        ball_to_goal_speed_reward = np.clip(SPEED_IMPORTANCE * self.last_ball_speed_mean_per_player[player_id] / max_ball_abs_avg_velocity, -SPEED_IMPORTANCE,
                               SPEED_IMPORTANCE) if CLIP_SPEED_REWARD_BY_SPEED_IMPORTANCE else SPEED_IMPORTANCE * self.last_ball_speed_mean_per_player[player_id] / max_ball_abs_avg_velocity
        agent_position_to_ball_reward = is_after_the_ball(player_id, player_pos,
                                  ball_pos) * (-AFTER_BALL_STEP_PENALTY)

        if splitted_returns:
            return (env_reward, ball_to_goal_speed_reward, agent_position_to_ball_reward)
        return env_reward + ball_to_goal_speed_reward + agent_position_to_ball_reward
        if CLIP_SPEED_REWARD_BY_SPEED_IMPORTANCE:
            # print(reward + np.clip(SPEED_IMPORTANCE * self.last_ball_speed_mean_per_player[player_id] / max_ball_abs_avg_velocity, -SPEED_IMPORTANCE, SPEED_IMPORTANCE))
            return reward + \
                np.clip(SPEED_IMPORTANCE * self.last_ball_speed_mean_per_player[player_id] / max_ball_abs_avg_velocity, -SPEED_IMPORTANCE, SPEED_IMPORTANCE) + \
                is_after_the_ball(player_id, player_pos,
                                  ball_pos) * AFTER_BALL_STEP_PENALTY
        return reward + \
            SPEED_IMPORTANCE * self.last_ball_speed_mean_per_player[player_id] / max_ball_abs_avg_velocity + \
            is_after_the_ball(player_id, player_pos,
                              ball_pos) * AFTER_BALL_STEP_PENALTY

    def _update_avg_ball_speed_to_goal(self, player_id: int, ball_speed: float):
        assert player_id in [0, 1, 2, 3]
        global min_ball_to_goal_avg_velocity, max_ball_to_goal_avg_velocity

        # Getting min/max ball to goal speed forr normalization
        # print(f'player_id: {player_id}')
        # print(f'self.last_ball_speed_mean_per_player: {self.last_ball_speed_mean_per_player}')
        # print(f'self.n_step: {self.n_step}')
        # print(f'ball_speed: {ball_speed}')

        self.ball_speed_deque_per_player[player_id].append(ball_speed)
        avg = np.mean(self.ball_speed_deque_per_player[player_id])
        # if avg < min_ball_to_goal_avg_velocity:
        #     min_ball_to_goal_avg_velocity = avg
        # elif avg > max_ball_to_goal_avg_velocity:
        #     max_ball_to_goal_avg_velocity = avg

        self.last_ball_speed_mean_per_player[player_id] = avg

    def _update_scoreboard(self, player_id, reward):
        global max_goals_one_team, max_goals_one_match

        if player_id == 0 and reward == -1.0:
            self.scoreboard["team_1"] += 1
            # print(self.scoreboard)

            # if self.scoreboard["team_1"] > max_goals_one_team:
            #     max_goals_one_team = self.scoreboard["team_1"]
            # if self.scoreboard["team_0"] + self.scoreboard["team_1"] > max_goals_one_match:
            #     max_goals_one_match = self.scoreboard["team_0"] + \
            #         self.scoreboard["team_1"]
            # if max_goals_one_match > 0:
            #     if not self.await_press:
            #         input("Press Enter to continue...")
            #         self.await_press = True
            #     else:
            #         self.await_press = False
        elif player_id == 2 and reward == -1.0:
            self.scoreboard["team_0"] += 1
            # print(self.scoreboard)

            # if self.scoreboard["team_0"] > max_goals_one_team:
            #     max_goals_one_team = self.scoreboard["team_0"]
            # if self.scoreboard["team_0"] + self.scoreboard["team_1"] > max_goals_one_match:
            #     max_goals_one_match = self.scoreboard["team_0"] + \
            #         self.scoreboard["team_1"]
            # if max_goals_one_match > 0:
            #     if not self.await_press:
            #         input("Press Enter to continue...")
            #         self.await_press = True
            #     else:
            #         self.await_press = False
