import gymnasium as gym
# import gym
import highway_env
from highway_env.envs import ControlledVehicle
from matplotlib import pyplot as plt
from pprint import pprint
import numpy as np
import os
import time
import rl_utils
np.set_printoptions(suppress=True)

"""
@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/eleurent/highway-env}},
}
"""
def state_pretreat(state):
    for i in range(1, state.shape[0]):
        state[i][3] += state[0][3]

def dis_to_con(discrete_aciton, env, action_dim):
    action_lowbound = env.action_space.low[0]
    action_upbound = env.action_space.high[0]
    return action_lowbound + (discrete_aciton / (action_dim - 1)) * (action_upbound - action_lowbound)

# print(gym.__version__)
# print(gym.__version__)

# Kinematics config
"""
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 10,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "sorted",
        "flatten": False,
        "normalize": False
    },
    "action": {
        "type": "DiscreteMetaAction"
        # "type": "ContinuousAction"
    },
    "centering_position": [0.3, 0.5],
    "duration": 15, #[s]
    "lanes_count": 4,
    "vehicles_count": 50,
    "controlled_vehicles": 1,
    "initial_lane_id": None,
    "ego_spacing": 2,
    "vehicles_density": 1,
    "collision_reward": -1,    # The reward received when colliding with a vehicle.
    "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                               # zero for other lanes.
    "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                               # lower speeds according to config["reward_speed_range"].
    "lane_change_reward": 0,   # The reward received at each lane change action.
    "reward_speed_range": [20, 30],
    "normalize_reward": False,
    "offroad_terminal": False,
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "scaling": 5.5,
    "show_trajectories": False,
    "render_agent": True,
    "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
    "manual_control": False,
    "real_time_rendering": False
}

"""

"""
# GrayscaleObservation config
# config = {
#        "observation": {
#            "type": "GrayscaleObservation",
#            "observation_shape": (128, 64),
#            "stack_size": 4,
#            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
#            "scaling": 1.75,
#        },
#        "policy_frequency": 2
#    }

# OccupancyGrid config
# config = {
#         "observation": {
#             "type": "OccupancyGrid",
#             "vehicles_count": 15,
#             "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
#             "features_range": {
#                 "x": [-100, 100],
#                 "y": [-100, 100],
#                 "vx": [-20, 20],
#                 "vy": [-20, 20]
#             },
#             "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
#             "grid_step": [5, 5],
#             "absolute": False
#         }
# }

# TimeCollision config
# config = {
#     "observation": {
#         "type": "TimeToCollision",
#         "horizon": 10
#     }
# }
"""

config_path = 'yaml_config/customized_config.yaml'
config = rl_utils.read_config(config_path)
action_dim = 20
dis_actions = gym.spaces.MultiDiscrete([action_dim, 2])
# print(type(dis_actions.sample()))

env = gym.make('tyc-highway-v0', render_mode='rgb_array')
env.unwrapped.configure(config)
# env.unwrapped.config["observation"]["normalize"] = False
pprint(env.unwrapped.config)
# env.configure({
#     "manual_control": True
# })
obs, info = env.reset(seed=2)
# fig, axes = plt.subplots(ncols=4, figsize=(12, 5))
# for i, ax in enumerate(axes.flat):
#     ax.imshow(obs[i, ...].T, cmap=plt.get_cmap('gray'))
# plt.show()
print(env.observation_space.shape)
# print(env.action_space.low[0])
print(env.action_space.n)
print(env.unwrapped.vehicle.lane_index)
print(isinstance(env.unwrapped.vehicle, ControlledVehicle))
done = truncated = False
T1 = time.perf_counter()
print(env.PERCEPTION_DISTANCE)
while not (done or truncated):
    action = env.action_space.sample()
    # action = 0
    # print(action)
    # continue_action = dis_to_con(action, env, action_dim)
    # print(continue_action)
    obs, reward, done, truncated, info = env.step(action)
    print(" next step ".center(100, '-'))
    print(obs)
    # print(f"On road:{env.unwrapped.vehicle.on_road}")
    # print(f'lane index: {env.unwrapped.vehicle.lane_index[2]}, reward: {reward}')
    plt.imshow(env.render())
    # plt.show()
    # for i in range(np.shape(obs)[0]):
    #     print("第{}辆车的碰撞时间预测:\n{}".format(i, obs[i, ...]))
    # env.render()
T2 = time.perf_counter()
print(f'time : {(T2 - T1) * 1000} ms')
# done = False
# while not done:
#     env.step(env.action_space.sample())
# plt.imshow(env.render())
# plt.show()
env.close()