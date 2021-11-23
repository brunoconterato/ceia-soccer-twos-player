import gym
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, overload

from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np

def get_scalar_projection(x, y):
    np.dot(x, y) / np.linalg.norm(y)

def calculate_ball_to_goal_speed(player_key: int, info: Dict):
    return 99.0


min_ball_position_x, max_ball_position_x = np.inf, -np.inf
min_ball_position_y, max_ball_position_y = np.inf, -np.inf

min_player_position_x, max_player_position_x = np.inf, -np.inf
min_player_position_y, max_player_position_y = np.inf, -np.inf


def calculate_reward(reward: float, player_key: int, info: Dict) -> float:
    global min_ball_position_x, max_ball_position_x, \
        min_ball_position_y, max_ball_position_y, \
        min_player_position_x, max_player_position_x, \
        min_player_position_y, max_player_position_y
    if reward != 0.0:
        print('Goal was made!', reward, info)
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
    return reward + calculate_ball_to_goal_speed(player_key, info)


class CustomRewardWrapper(gym.core.Wrapper, MultiAgentEnv):
    # def __init__(self, env):
    #     gym.Wrapper.__init__(self, env)

    def step(self, action: Union[Dict[int, List[Any]], List[Any]]):
        obs, rewards, done, info = super().step(action)

        if type(action) is dict:
            rewards = {k: calculate_reward(
                rewards[k], k, info[k]) for k in info.keys()}
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

        return obs, rewards, done, info
