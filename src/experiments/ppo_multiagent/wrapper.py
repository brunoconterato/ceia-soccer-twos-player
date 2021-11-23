import gym
from typing import Any, Dict, List, Union

from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from collections import deque


def get_scalar_projection(x, y):
    return np.dot(x, y) / np.linalg.norm(y)


# Registered by experience
min_ball_position_x, max_ball_position_x = - \
    15.563264846801758, 15.682827949523926
min_ball_position_y, max_ball_position_y = -7.08929967880249, 7.223850250244141
min_player_position_x, max_player_position_x = - \
    17.26804542541504, 17.16301727294922
min_player_position_y, max_player_position_y = - \
    7.399587631225586, 7.406457424163818
min_ball_to_goal_avg_velocity, max_ball_to_goal_avg_velocity = - \
    10.90982112794721, 12.360231160743304
max_goals_one_team = -999999
max_goals_one_match = -9999999

# Infered
max_ball_abs_velocity = max(
    abs(min_ball_to_goal_avg_velocity), abs(max_ball_to_goal_avg_velocity))


def get_center_of_goal_pos(player_id):
    global min_ball_position_x, max_ball_position_x, \
        min_ball_position_y, max_ball_position_y, \
        min_player_position_x, max_player_position_x, \
        min_player_position_y, max_player_position_y
    if player_id in [0, 1]:
        return np.array([max_ball_position_x, 0.0])
    elif player_id in [2, 3]:
        return np.array([min_ball_position_x, 0.0])


def calculate_ball_to_goal_speed(player_id: int, info: Dict):
    goal_pos = get_center_of_goal_pos(player_id)
    # print(f"goal_pos: {goal_pos}")
    ball_pos = info["ball_info"]["position"]
    # print(f"ball_pos: {ball_pos}")
    direction_to_center_of_goal = ball_pos - goal_pos
    # print(f"direction_to_center_of_goal: {direction_to_center_of_goal}")

    ball_velocity = info["ball_info"]["velocity"]
    # print(f"ball_velocity: {ball_velocity}")
    ball_velocity_to_center_of_goal = get_scalar_projection(
        ball_velocity, direction_to_center_of_goal)
    # print(f"ball_velocity_to_center_of_goal: {ball_velocity_to_center_of_goal}")
    return ball_velocity_to_center_of_goal


class CustomRewardWrapper(gym.core.Wrapper, MultiAgentEnv):
    # def __init__(self, env):
    #     gym.Wrapper.__init__(self, env)

    def step(self, action: Union[Dict[int, List[Any]], List[Any]]):
        obs, rewards, done, info = super().step(action)

        if type(action) is dict:
            rewards = {k: self._calculate_reward(
                rewards[k], k, info[k]) for k in info.keys()}
        else:
            raise NotImplementedError('Necessário implementar!')

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

        self.n_step += 1
        return obs, rewards, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.n_step = 0
        self.last_ball_speed_mean_per_player = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        self.ball_speed_deque_per_player = {0: deque(maxlen=400), 1: deque(
            maxlen=400), 2: deque(maxlen=400), 3: deque(maxlen=400)}
        self.scoreboard = {"team_0": 0, "team_1": 0}
        self.await_press = False
        # print(f'min_ball_to_goal_avg_velocity: {min_ball_to_goal_avg_velocity}')
        # print(f'max_ball_to_goal_avg_velocity: {max_ball_to_goal_avg_velocity}')
        return obs

    def _calculate_reward(self, reward: float, player_id: int, info: Dict) -> float:
        # print('calculating reward')
        if reward != 0.0:
            # print('Goal was made!', reward, info)
            self._update_scoreboard(player_id, reward)
        global min_ball_position_x, max_ball_position_x, \
            min_ball_position_y, max_ball_position_y, \
            min_player_position_x, max_player_position_x, \
            min_player_position_y, max_player_position_y
        # print(f"info: {info}")
        if info["ball_info"]["position"][0] < min_ball_position_x:
            min_ball_position_x = info["ball_info"]["position"][0]
        if info["ball_info"]["position"][0] > max_ball_position_x:
            max_ball_position_x = info["ball_info"]["position"][0]
        if info["ball_info"]["position"][1] < min_ball_position_y:
            min_ball_position_y = info["ball_info"]["position"][1]
        if info["ball_info"]["position"][1] > max_ball_position_y:
            max_ball_position_y = info["ball_info"]["position"][1]
        if info["player_info"]["position"][0] < min_player_position_x:
            min_player_position_x = info["player_info"]["position"][0]
        if info["player_info"]["position"][0] > max_player_position_x:
            max_player_position_x = info["player_info"]["position"][0]
        if info["player_info"]["position"][1] < min_player_position_y:
            min_player_position_y = info["player_info"]["position"][1]
        if info["player_info"]["position"][1] > max_player_position_y:
            max_player_position_y = info["player_info"]["position"][1]

        self._update_avg_ball_speed_to_goal(
            player_id, calculate_ball_to_goal_speed(player_id, info))
        return reward + self.last_ball_speed_mean_per_player[player_id]

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
        if avg < min_ball_to_goal_avg_velocity:
            # print(f'new min_ball_to_goal_avg_velocity: {min_ball_to_goal_avg_velocity}')
            min_ball_to_goal_avg_velocity = avg
        elif avg > max_ball_to_goal_avg_velocity:
            # print(f'new max_ball_to_goal_avg_velocity: {max_ball_to_goal_avg_velocity}')
            max_ball_to_goal_avg_velocity = avg

        self.last_ball_speed_mean_per_player[player_id] = avg

    def _update_scoreboard(self, player_id, reward):
        global max_goals_one_team, max_goals_one_match

        if player_id == 0 and reward == -1.0:
            self.scoreboard["team_1"] += 1
            # print(self.scoreboard)

            if self.scoreboard["team_1"] > max_goals_one_team:
                max_goals_one_team = self.scoreboard["team_1"]
            if self.scoreboard["team_0"] + self.scoreboard["team_1"] > max_goals_one_match:
                max_goals_one_match = self.scoreboard["team_0"] + \
                    self.scoreboard["team_1"]
            # if max_goals_one_match > 0:
            #     if not self.await_press:
            #         input("Press Enter to continue...")
            #         self.await_press = True
            #     else:
            #         self.await_press = False
        elif player_id == 2 and reward == -1.0:
            self.scoreboard["team_0"] += 1
            # print(self.scoreboard)

            if self.scoreboard["team_0"] > max_goals_one_team:
                max_goals_one_team = self.scoreboard["team_0"]
            if self.scoreboard["team_0"] + self.scoreboard["team_1"] > max_goals_one_match:
                max_goals_one_match = self.scoreboard["team_0"] + \
                    self.scoreboard["team_1"]
            # if max_goals_one_match > 0:
            #     if not self.await_press:
            #         input("Press Enter to continue...")
            #         self.await_press = True
            #     else:
            #         self.await_press = False
