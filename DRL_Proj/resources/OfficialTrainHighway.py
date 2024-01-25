# -*- coding: utf-8 -*-
# @Time     : 2023/12/26 16:25
# @Author   : TangYaochen
# @File     : OfficialTrainHighway.py
# @Software : PyCharm

import gymnasium as gym
from matplotlib import pyplot as plt
from pprint import pprint
import highway_env
import torch
import numpy as np
from stable_baselines3 import DQN, PPO
import time
np.set_printoptions(suppress=True)

if __name__ == '__main__':
    actions_all = {
		0: 'Lane_left',
		1: 'Keeping',
		2: 'Lane_right',
		3: 'Faster',
		4: 'Slower'
	}
    env = gym.make("highway-v0", render_mode='rgb_array') #
    # pprint(env.config)
    model = PPO("MlpPolicy", env, verbose=1)
    # model = DQN('MlpPolicy', env,
    #               policy_kwargs=dict(net_arch=[256, 256]),
    #               learning_rate=5e-4,
    #               buffer_size=15000,
    #               learning_starts=200,
    #               batch_size=32,
    #               gamma=0.8,
    #               train_freq=1,
    #               gradient_steps=1,
    #               target_update_interval=50,
    #               verbose=1,
    #               tensorboard_log="highway_dqn/")
    # print(model.get_parameters()['policy'])
    # env1 = model.get_env()
    # print(env1.observation_space.shape)
    # layer_param = model.get_parameters()['policy']
    # for key, value in layer_param.items():
    #     print("{0} shape is {1}".format(key, value.shape))
    ''''''
    # T1 = time.perf_counter()
    # # model.learn(int(2e4))
    # model.learn(total_timesteps=25000)
    # T2 = time.perf_counter()
    # model.save("highway_ppo/model")
    # print(f"Official trian time: {(T2 - T1)/3600} hours")

    # del model


    # Load and test saved model
    env.config["duration"] = 180
    pprint(env.config)
    model = PPO.load("highway_ppo/model")
    done = truncated = False
    obs, info = env.reset(seed=3)
    # print(obs)
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        # print('next step'.center(60, '-'))
        print("action: {0:<10}".format(actions_all[int(action)])) 
        # print(obs)
        env.render()
    env.close()

