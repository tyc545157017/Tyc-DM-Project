# -*- coding: utf-8 -*-
# @Time     : 2023/12/25 19:57
# @Author   : TangYaochen
# @File     : DQN_highwayenv.py
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
import time
from MyParamManager import ParamManager

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Qnet, self).__init__()
        hidden1_dim = 1024
        hidden2_dim = 512
        self.fc1 = nn.Linear(state_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, action_dim)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)

class VAnet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(VAnet, self).__init__()
        hidden1_dim = 1024
        hidden2_dim = 512
        self.fc1 = nn.Linear(state_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc_A = nn.Linear(hidden2_dim, action_dim)
        self.fc_V = nn.Linear(hidden2_dim, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        A = self.fc_A(x)
        V = self.fc_V(x)
        # print(A)
        # print(V)
        Q = A + V - A.mean(-1).view(-1, 1)
        return Q

class DQN:
    def __init__(self, state_dim, action_dim,
                 learning_rate, gamma, epsilon, target_update, device, dqn_type='NormalDQN'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        if dqn_type == 'DuelingDQN':
            self.q_net = VAnet(state_dim, self.action_dim).to(device)
            self.target_q_net = VAnet(state_dim, self.action_dim).to(device)
        else:
            self.q_net = Qnet(state_dim, self.action_dim).to(device)
            self.target_q_net = Qnet(state_dim, self.action_dim).to(device)
        # self.learning_rate = learning_rate

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        self.count = 0
        self.dqn_type = dqn_type
        self.loss = 0

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            # action = np.argmax(self.q_net(state).data.numpy())
            action = self.q_net(state).argmax().item()
        return action

    def best_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        action = self.target_q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transitions_dict):
        states = torch.tensor(np.array(transitions_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transitions_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transitions_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transitions_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transitions_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        # print("Update...")
        # print(np.shape(states))
        q_values = self.q_net(states).gather(1, actions)
        if self.dqn_type == "DoubleDQN":
            next_actions = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, next_actions)
        else:
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_target = rewards + self.gamma * max_next_q_values * (1 - dones)

        dqn_loss = torch.mean(nn.functional.mse_loss(q_values, q_target))
        self.loss = dqn_loss.item()

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.count += 1

def dqn_train(env, replay_buffer, num_episodes,
              lr, minimal_size, batch_size,
              gamma, epsilon, target_update, device, dqn_type,
              model_path, retrain=False):
    state_shape = env.observation_space.shape
    if len(state_shape) == 2:
        state_dim = state_shape[0] * state_shape[1]
    else:
        raise ValueError
    action_dim = env.action_space.n
    print("\n开始{}训练".format(dqn_type))
    print("状态维度：{}, 动作维度：{}".format(state_dim, action_dim))
    agent = DQN(state_dim, action_dim, lr, gamma, epsilon, target_update, device, dqn_type)
    if retrain:
        agent.q_net.load_state_dict(torch.load(model_path))
        agent.target_q_net.load_state_dict(torch.load(model_path))
        print('load model to retrian')

    episode_len = []
    return_list = []
    loss_list = []
    max_q_value_list = []
    max_q_value = 0
    best_mr = 0
    save_step = int(num_episodes / 10)
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                state = state.flatten()
                # print(state)
                # print(np.shape(state))
                done = False
                count = 0
                loss = 0
                # print("i am here")
                while not done:
                    count += 1
                    # print(done)
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(state)*0.005 + max_q_value*0.995
                    max_q_value_list.append(max_q_value)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    # plt.imshow(env.render())
                    if terminated or truncated:
                        done = True
                    # print("next_state:{}, reward:{}".format(next_state, reward))
                    next_state = next_state.flatten()
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                        transitions_dict = {
                            "states": states,
                            "actions": actions,
                            "rewards": rewards,
                            "next_states": next_states,
                            "dones": dones
                        }
                        agent.update(transitions_dict)
                    loss += agent.loss
                episode_len.append(count)
                return_list.append(episode_return)
                loss_list.append(loss/count)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
        mr = np.mean(return_list[-save_step:])
        if mr > best_mr:
            print("save best reward model:(now_mr={0}, last_mr={1}) per {2} episodes".format(mr, best_mr, save_step))
            torch.save(agent.target_q_net.state_dict(), model_path.rsplit('.pkl')[0] + '_best.pkl')
            best_mr = mr
    torch.save(agent.target_q_net.state_dict(), model_path)
    return (return_list, episode_len, loss_list)


def save_train_log(log_name, **datas):
    try:
        data_list = []
        colums = datas.keys()
        df = pd.DataFrame(columns=colums)
        df.to_csv(log_name, index=False)
        for key in colums:
            data_list.append(datas[key])
        datas = pd.DataFrame(data_list).T
        datas.to_csv(log_name, mode='a', header=False, index=False)
    except Exception as e:
        print(type(e))
        raise KeyError("Can't find args")


def display_frames_as_gif(frames, filename):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=4)
    anim.save('./{}'.format(filename), writer='pillow', fps=120)

@gif.frame
def plot_show(i):
    plt.imshow(i)


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
        "duration": 300,
        "render_agent": True
    }

    """

    random_seed = 0

    retrain = False
    train = True
    train_version = -1
    # dqn_type = "DuelingDQN"
    # env_name = 'highway-v0'
    env_name = 'tyc-highway-v0'
    dqn_type = "DoubleDQN"
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
    num_episodes = 3000
    dqn_pm = ParamManager(env_name, dqn_type, drl_param, lr, num_episodes, train_version=train_version, train=train)
    # train_version = dqn_pm.train_version

    model_path = os.path.join(dqn_pm.model_dir, 'model.pkl')
    config_path = os.path.join(dqn_pm.config_dir, 'config.yaml')
    logs_path = os.path.join(dqn_pm.logs_dir, 'train_log.csv')
    config = dqn_pm.config

    # lr = 5e-4
    # gamma = 0.9
    # epsilon = 0.01
    # target_update = 50
    # buffer_size = 15000
    # minimal_size = 200
    # batch_size = 64
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
        # env.render_mode = None
        if os.path.exists(model_path) and not retrain:
            raise ValueError("model path has existed, maybe override, please check the name!")
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
        env.config["vehicles_count"] = 50
        env.config["vehicles_density"] = 1
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
        print("状态维度：{0:<10}, 动作维度：{1:<10}".format(state_dim, action_dim))
        agent = DQN(state_dim, action_dim, lr["lr"],
                    drl_param["gamma"], drl_param["epsilon"], drl_param["target_update"], device, dqn_type)
        print(f"load model: {model_path}")
        agent.target_q_net.load_state_dict(torch.load(model_path)) # final_model_path
        state, _ = env.reset(seed=8)
        state = state.flatten()
        done = False
        count = 0
        r_episode = 0
        env.render()
        # while input('Please enter y for testing recording: ') != 'y':
        #     pass
        T1 = time.perf_counter()
        while not done:
            count += 1
            frame = plot_show(env.render())
            if 50 > count > 10:
                frames.append(frame)
            action = agent.best_action(state)
            # action_continuous = dis_to_con(action, env, action_dim)
            s_, r_, ter_, trunc_, _ = env.step(action)
            if ter_ or trunc_:
                done = True
            s_ = s_.flatten()
            r_episode += r_
            state = s_
            # print("action: {0:<10}, step: {1:<5}, target lane index: {2}".format(actions_all[action], count, 
            #                                                               env.unwrapped.vehicle.target_lane_index[2]))
            print("action: {0:<10}, step: {1:<5}, reward: {2:<10}, speed: {3}".format(actions_all[action], count, r_, 
                                                                            env.unwrapped.vehicle.speed*np.cos(env.unwrapped.vehicle.heading)))
            # print("high_speed_reward:{}".format(env.config["high_speed_reward"]))
        # gif_filename = os.path.join(dqn_pm.logs_dir, 'train_log.gif')
        # display_frames_as_gif(frames[10:20], gif_filename)
        # gif.save(frames, gif_filename, duration=10, unit='s', between='startend')
        T2 = time.perf_counter()
        print(f"{count}轮后停止, 总收益为{r_episode}")
        print(f"程序运行时间: {(T2-T1)} s")
    env.close()

