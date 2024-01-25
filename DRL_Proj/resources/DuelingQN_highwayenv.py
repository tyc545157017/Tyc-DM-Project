# -*- coding: utf-8 -*-
# @Time     : 2024/1/2 22:04
# @Author   : TangYaochen
# @File     : DuelingQN_highwayenv.py
# @Software : PyCharm

import torch
import torch.nn as nn
import numpy as np
import collections
import matplotlib.pyplot as plt
import random
import gymnasium as gym
from tqdm import tqdm
import rl_utils
import highway_env
from pprint import pprint
import yaml
import os
from matplotlib import animation
import gif
import pandas as pd
from DQN_highwayenv import *

if __name__ == "__main__":
    actions_all = {
        0: 'Lane_left',
        1: 'Keeping',
        2: 'Lane_right',
        3: 'Faster',
        4: 'Slower'
    }
    """
     config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 6,
            "features": ["presence", "x", "y", "vx", "vy"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20]
            },
            "absolute": False,
            "order": "sorted",
            "normalize": True
        },
        "duration": 15,
        "render_agent": True
    }

    """
    random_seed = 0

    retrain = False
    train = False
    # dqn_type = "DuelingDQN"
    # env_name = 'highway-v0'
    env_name = 'tyc-highway-v0'
    dqn_type = "DuelingDQN"
    drl_param = {
        "gamma": 0.9,
        "epsilon": 0.01,
        "target_update": 50,
        "buffer_size": 15000,
        "minimal_size": 200,
        "batch_size": 64

    }
    lr = {
        "lr": 5e-4
    }
    num_episodes = 2000
    dqn_pm = ParamManager(env_name, dqn_type, drl_param, lr, num_episodes, train=train)

    model_path = os.path.join(dqn_pm.model_dir, 'model.pkl')
    config_path = os.path.join(dqn_pm.config_dir, 'config.yaml')
    logs_path = os.path.join(dqn_pm.logs_dir, 'train_log.csv')
    config = dqn_pm.config

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    env = gym.make(env_name, render_mode="rgb_array") #
    env.configure(config)
    pprint(env.config)
    obs, info = env.reset(seed=random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    # env.render()
    replay_buffer = ReplayBuffer(drl_param["buffer_size"])
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n

    if train:
        if torch.cuda.is_available():
            print("Use GPU train")
        T1 = time.perf_counter()
        return_lists = dqn_train(env, replay_buffer, num_episodes, lr["lr"],
                                 drl_param["minimal_size"], drl_param["batch_size"],
                                 drl_param["gamma"], drl_param["epsilon"], drl_param["target_update"],
                                 device, dqn_type, model_path=model_path, retrain=retrain)
        T2 = time.perf_counter()
        print(f"trian time: {(T2-T1)/3600} hours")
        dqn_pm.save_config_file()
        if not retrain:
            dqn_pm.update_version_files()
        save_train_log(logs_path, **{"reward": return_lists[0], "episode_len": return_lists[1], "loss": return_lists[2]})
        episodes_list = list(range(len(return_lists[0])))
        mv_return = rl_utils.moving_average(return_lists[0], 9)
        plt.plot(episodes_list, mv_return, label='{}'.format(return_lists[1]))
        plt.legend()
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DQN on {}'.format(env_name))
        plt.show()
    else:
        frames = []
        env.config["duration"] = 300
        env.config["vehicles_count"] = 80
        # env.configure()
        pprint(env.config)
        # env.render_mode = "human"
        state_shape = env.observation_space.shape
        if len(state_shape) == 2:
            state_dim = state_shape[0] * state_shape[1]
        else:
            raise ValueError
        action_dim = env.action_space.n
        print("\n开始{}测试".format(dqn_type))
        print("状态维度：{}, 动作维度：{}".format(state_dim, action_dim))
        agent = DQN(state_dim, action_dim, lr["lr"],
                    drl_param["gamma"], drl_param["epsilon"], drl_param["target_update"], device, dqn_type)
        print(model_path)
        agent.target_q_net.load_state_dict(torch.load(model_path)) # final_model_path
        state, _ = env.reset(seed=3)
        state = state.flatten()
        done = False
        count = 0
        r_episode = 0
        ego_vehicle = env.unwrapped.vehicle
        while not done:
            count += 1
            frame = plot_show(env.render())
            if 50 > count > 10:
                frames.append(frame)
            action = agent.best_action(state)
            # print("action: {}, step: {}".format(actions_all[action], count))
            # action_continuous = dis_to_con(action, env, action_dim)
            s_, r, ter_, trunc_, _ = env.step(action)
            if ter_ or trunc_:
                done = True
            s_ = s_.flatten()
            r_episode += r
            state = s_
            print("action: {0:<10}, forward speed: {1:<5.3f}, step: {2:<5}, target lane index: {3}".format(actions_all[action], 
                                                                                                        ego_vehicle.speed * np.cos(ego_vehicle.heading), 
                                                                                                        count, 
                                                                                                        ego_vehicle.target_lane_index[2]))
            # print("high_speed_reward:{}".format(env.config["high_speed_reward"]))
        gif_filename = os.path.join(dqn_pm.logs_dir, 'train_log.gif')
        # display_frames_as_gif(frames[10:20], gif_filename)
        gif.save(frames, gif_filename, duration=10, unit='s', between='startend')
        print("%d轮后停止" % count)
        print("总收益为%d" % r_episode)
    env.close()
