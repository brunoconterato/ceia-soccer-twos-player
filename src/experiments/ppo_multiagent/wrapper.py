import gym
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, overload

from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np


def get_scalar_projection(x, y):
    return np.dot(x, y) / np.linalg.norm(y)


min_ball_position_x, max_ball_position_x = -15.491240501403809, 15.682827949523926
min_ball_position_y, max_ball_position_y = -6.946798324584961, 7.0257463455200195

min_player_position_x, max_player_position_x = -17.14044189453125, 17.060199737548828
min_player_position_y, max_player_position_y = -7.3328938484191895, 7.272202014923096

min_ball_to_goal_speed, max_ball_to_goal_speed = np.inf, -np.inf


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
    ball_velocity_to_center_of_goal = get_scalar_projection(ball_velocity, direction_to_center_of_goal)
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
            print(f'new rewards: {rewards}')
        else:
            raise NotImplementedError('Necessário implementar!')

        global min_ball_position_x, max_ball_position_x, \
            min_ball_position_y, max_ball_position_y, \
            min_player_position_x, max_player_position_x, \
            min_player_position_y, max_player_position_y
        if done:
            print(f'min_ball_position_x: {min_ball_position_x}')
            print(f'max_ball_position_x: {max_ball_position_x}')
            print(f'min_ball_position_y: {min_ball_position_y}')
            print(f'max_ball_position_y: {max_ball_position_y}')
            print(f'min_player_position_x: {min_player_position_x}')
            print(f'max_player_position_x: {max_player_position_x}')
            print(f'min_player_position_y: {min_player_position_y}')
            print(f'max_player_position_y: {max_player_position_y}')
            print(f'min_ball_to_goal_speed: {min_ball_to_goal_speed}')
            print(f'max_ball_to_goal_speed: {max_ball_to_goal_speed}')

        self.n_step += 1
        return obs, rewards, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.n_step = 0
        self.last_ball_speed_mean_per_player = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        # print(f'min_ball_to_goal_speed: {min_ball_to_goal_speed}')
        # print(f'max_ball_to_goal_speed: {max_ball_to_goal_speed}')
        return obs

    def _calculate_reward(self, reward: float, player_id: int, info: Dict) -> float:
        # print('calculating reward')
        if reward != 0.0:
            print('Goal was made!', reward, info)
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

        self._update_mean_ball_speed_to_goal(player_id, calculate_ball_to_goal_speed(player_id, info))
        return reward + self.last_ball_speed_mean_per_player[player_id]

    def _update_mean_ball_speed_to_goal(self, player_id: int, ball_speed: float):
        assert player_id in [0, 1, 2, 3]
        global min_ball_to_goal_speed, max_ball_to_goal_speed

        # Getting min/max ball to goal speed forr normalization
        # print(f'player_id: {player_id}')
        # print(f'self.last_ball_speed_mean_per_player: {self.last_ball_speed_mean_per_player}')
        # print(f'self.n_step: {self.n_step}')
        # print(f'ball_speed: {ball_speed}')
        if (self.last_ball_speed_mean_per_player[player_id] * self.n_step + ball_speed) / ( self.n_step + 1 ) < min_ball_to_goal_speed:
            # print(f'new min_ball_to_goal_speed: {min_ball_to_goal_speed}')
            min_ball_to_goal_speed = (self.last_ball_speed_mean_per_player[player_id] * self.n_step + ball_speed) / ( self.n_step + 1 )
        elif (self.last_ball_speed_mean_per_player[player_id] * self.n_step + ball_speed) / ( self.n_step + 1 ) > max_ball_to_goal_speed:
            # print(f'new max_ball_to_goal_speed: {max_ball_to_goal_speed}')
            max_ball_to_goal_speed = (self.last_ball_speed_mean_per_player[player_id] * self.n_step + ball_speed) / ( self.n_step + 1 )

        self.last_ball_speed_mean_per_player[player_id] = \
            (self.last_ball_speed_mean_per_player[player_id] * self.n_step + ball_speed) / ( self.n_step + 1 )
                
