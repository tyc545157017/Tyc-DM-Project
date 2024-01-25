
import torch
import torch.nn.functional as F
from torch.distributions import Normal
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
from DQN_highwayenv import *
from MyParamManager import ParamManager


class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        hidden1_dim = 1024
        hidden2_dim = 512
        self.fc1 = torch.nn.Linear(state_dim, hidden1_dim)
        self.fc2 = torch.nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = torch.nn.Linear(hidden2_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)


class QValueNet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, action_dim):
        super(QValueNet, self).__init__()
        hidden1_dim = 1024
        hidden2_dim = 512
        self.fc1 = torch.nn.Linear(state_dim, hidden1_dim)
        self.fc2 = torch.nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = torch.nn.Linear(hidden2_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class SAC:
    ''' 处理离散动作的SAC算法 '''
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, action_dim).to(device)
        # 第一个Q网络
        self.critic_1 = QValueNet(state_dim, action_dim).to(device)
        # 第二个Q网络
        self.critic_2 = QValueNet(state_dim, action_dim).to(device)
        self.target_critic_1 = QValueNet(state_dim,
                                         action_dim).to(device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNet(state_dim,
                                         action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.act_loss = 0
        self.crit1_loss = 0
        self.crit2_loss = 0
        self.alp_loss = 0
    """
    def best_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        print(f'probs:{probs}, type:{type(probs)}')
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        print(f'action:{action}')
        return action.item()
    """
    

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)  # 动作不再是float类型
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        self.act_loss = actor_loss.item()
        self.crit1_loss = critic_1_loss.item()
        self.crit2_loss = critic_2_loss.item()
        self.alp_loss = alpha_loss.item()


class PolicyNetContinuous(torch.nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
		super(PolicyNetContinuous, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
		self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
		self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
		self.action_bound = action_dim

	def forward(self, x):
		x = F.relu(self.fc1(x))
		mu = self.fc_mu(x)
		std = F.softplus(self.fc_std(x))
		dist = Normal(mu, std)
		normal_sample = dist.rsample()
		log_prob = dist.log_prob(normal_sample)
		action = torch.tanh(normal_sample)
		log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
		action = action * self.action_bound
		return action, log_prob

class QValueNetContinuous(torch.nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim):
		super(QValueNetContinuous, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
		self.fc_out = torch.nn.Linear(hidden_dim, 1)

	def forward(self, x, a):
		cat = torch.cat([x, a], dim=1)
		x = F.relu(self.fc1(cat))
		x = F.relu(self.fc2(x))
		return self.fc_out(x)


class SACContinuous:
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
    
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        return [action.item()]

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # 和之前章节一样,对倒立摆环境的奖励进行重塑以便训练
        rewards = (rewards + 8.0) / 8.0

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)



def sac_train(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, 
              drl_type, model_path, retrain=False):
    print("\n开始{}训练".format(drl_type))
    print("状态维度：{}, 动作维度：{}".format(state_dim, action_dim))
    actor_model_path = model_path.rsplit('.', 1)[0] + '_actor.pkl'
    critic1_model_path = model_path.rsplit('.', 1)[0] + '_critic1.pkl'
    critic2_model_path = model_path.rsplit('.', 1)[0] + '_critic2.pkl'
    print(model_path)
    if retrain:
        agent.actor.load_state_dict(torch.load(actor_model_path))
        agent.critic_1.load_state_dict(torch.load(critic1_model_path))
        agent.critic_2.load_state_dict(torch.load(critic2_model_path))
        agent.soft_update(agent.critic_1, agent.target_critic_1)
        agent.soft_update(agent.critic_2, agent.target_critic_2)
        print('load model to retrian')
    
    episode_len = []
    return_list = []
    loss_dict = {"act_loss": [], "crit1_loss": [], "crit2_loss": [], "alp_loss": []}
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, _ = env.reset()
                state = state.flatten()
                done = False
                count = 0
                act_loss = crit1_loss = crit2_loss = alp_loss = 0
                while not done:
                    count += 1
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state = next_state.flatten()
                    if terminated or truncated:
                        done = True
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                    act_loss += agent.act_loss
                    crit1_loss += agent.crit1_loss
                    crit2_loss += agent.crit2_loss
                    alp_loss += agent.alp_loss
                episode_len.append(count)
                return_list.append(episode_return)
                loss_dict["act_loss"].append(act_loss/count)
                loss_dict["crit1_loss"].append(crit1_loss/count)
                loss_dict["crit2_loss"].append(crit2_loss/count)
                loss_dict["alp_loss"].append(alp_loss/count)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    torch.save(agent.actor.state_dict(), actor_model_path)
    torch.save(agent.critic_1.state_dict(), critic1_model_path)
    torch.save(agent.critic_2.state_dict(), critic2_model_path)
    return (return_list, episode_len, loss_dict)


if __name__ == '__main__':
    """
	# env_name = 'Pendulum-v0'
	# env = gym.make(env_name)
	# state_dim = env.observation_space.shape[0]
	# action_dim = env.action_space.shape[0]
	# action_bound = env.action_space.high[0]  # 动作最大值
	# random.seed(0)
	# np.random.seed(0)
	# env.seed(0)
	# torch.manual_seed(0)
	#
	# actor_lr = 3e-4
	# critic_lr = 3e-3
	# alpha_lr = 3e-4
	# num_episodes = 100
	# hidden_dim = 128
	# gamma = 0.99
	# tau = 0.005  # 软更新参数
	# buffer_size = 100000
	# minimal_size = 1000
	# batch_size = 64
	# target_entropy = -env.action_space.shape[0]
	# device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
	# 	"cpu")
	#
	# replay_buffer = rl_utils.ReplayBuffer(buffer_size)
	# agent = SACContinuous(state_dim, hidden_dim, action_dim, action_bound,
	# 					  actor_lr, critic_lr, alpha_lr, target_entropy, tau,
	# 					  gamma, device)
	#
	# return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes,
	# 											  replay_buffer, minimal_size,
	# 											  batch_size)
	#
     
    actor_lr = 1e-3
	critic_lr = 1e-2
	alpha_lr = 1e-2
	num_episodes = 200
	hidden_dim = 128
	gamma = 0.98
	tau = 0.005  # 软更新参数
	buffer_size = 10000
	minimal_size = 500
	batch_size = 64
	target_entropy = -1
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
		"cpu")

	env_name = 'CartPole-v0'
	env = gym.make(env_name)
	random.seed(0)
	np.random.seed(0)
	env.seed(0)
	torch.manual_seed(0)
	replay_buffer = rl_utils.ReplayBuffer(buffer_size)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n
	agent = SAC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr,
				target_entropy, tau, gamma, device)

	return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes,
												  replay_buffer, minimal_size,
												  batch_size)

	episodes_list = list(range(len(return_list)))
	plt.plot(episodes_list, return_list)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('SAC on {}'.format(env_name))
	plt.show()

	mv_return = rl_utils.moving_average(return_list, 9)
	plt.plot(episodes_list, mv_return)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('SAC on {}'.format(env_name))
	plt.show()
    """

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
        "duration": 60,
        "render_agent": True, 
        "offroad_terminal": True, 
        "normalize_reward": False
    }
    """

    retrain = False
    train = True
    # env_name = 'highway-v0'
    env_name = 'tyc-highway-v0'
    drl_type = "SAC"
    train_version = -1
    lr = {
        'actor_lr': 5e-4,
        'critic_lr': 5e-3,
        'alpha_lr': 5e-3
    }
    drl_param = {
        'gamma': 0.9,
        'tau': 0.005,
        'buffer_size': 15000,
        'minimal_size': 200,
        'batch_size': 64,
        'target_entropy': -1

    }
    num_episodes = 10000
    sac_pm = ParamManager(env_name, drl_type, drl_param, lr, num_episodes, train_version=train_version, train=train)
    model_path = os.path.join(sac_pm.model_dir, 'model.pkl')
    config_path = os.path.join(sac_pm.config_dir, 'config.yaml')
    logs_path = os.path.join(sac_pm.logs_dir, 'train_log.csv')
    config = sac_pm.config

    random_seed = 0

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env = gym.make(env_name, render_mode="rgb_array") #
    env.configure(config)
    pprint(env.config)
    obs, info = env.reset(seed=random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    # env.render()
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n

    if train:
        if torch.cuda.is_available():
            print("Use GPU train")
        # env.render_mode = None
        if os.path.exists(model_path) and not retrain:
            raise ValueError("model path has existed, maybe override, please check the name!")
        replay_buffer = ReplayBuffer(drl_param['buffer_size'])
        T1 = time.perf_counter()
        state_shape = env.observation_space.shape
        if len(state_shape) == 2:
            state_dim = state_shape[0] * state_shape[1]
        else:
            raise ValueError
        action_dim = env.action_space.n
        agent = SAC(state_dim, action_dim, lr['actor_lr'], lr['critic_lr'], lr['alpha_lr'],
                drl_param['target_entropy'], drl_param['tau'], drl_param['gamma'], device)
        return_lists = sac_train(env, agent, num_episodes, replay_buffer,
                                 drl_param['minimal_size'], drl_param['batch_size'],
                                 drl_type, model_path=model_path, retrain=retrain)
        T2 = time.perf_counter()
        print(f"trian time: {(T2-T1)/3600} hours")
        sac_pm.save_config_file()
        if not retrain:
            sac_pm.update_version_files()
        save_train_log(logs_path, **{"reward": return_lists[0], "episode_len": return_lists[1], **return_lists[2]})
        episodes_list = list(range(len(return_lists[0])))
        mv_return = rl_utils.moving_average(return_lists[0], 9)
        plt.plot(episodes_list, mv_return, label='{}'.format(return_lists[1]))
        plt.legend()
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('SAC on {}'.format(env_name))
        plt.show()
    else:
        actor_model_path = model_path.rsplit('.', 1)[0] + '_actor.pkl'
        critic1_model_path = model_path.rsplit('.', 1)[0] + '_critic1.pkl'
        critic2_model_path = model_path.rsplit('.', 1)[0] + '_critic2.pkl'
        print(model_path)
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
        print("\n开始{}测试".format(drl_type))
        print("状态维度：{0:<10}, 动作维度：{1:<10}".format(state_dim, action_dim))
        agent = SAC(state_dim, action_dim, lr['actor_lr'], lr['critic_lr'], lr['alpha_lr'],
                drl_param['target_entropy'], drl_param['tau'], drl_param['gamma'], device)
        print(f"load model: {model_path}")
        agent.actor.load_state_dict(torch.load(actor_model_path))
        agent.critic_1.load_state_dict(torch.load(critic1_model_path))
        agent.critic_2.load_state_dict(torch.load(critic2_model_path))
        agent.soft_update(agent.critic_1, agent.target_critic_1)
        agent.soft_update(agent.critic_2, agent.target_critic_2)
        state, _ = env.reset(seed=3)
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
            action = agent.take_action(state)
            # action_continuous = dis_to_con(action, env, action_dim)
            s_, r_, ter_, trunc_, _ = env.step(action)
            if ter_ or trunc_:
                done = True
            s_ = s_.flatten()
            r_episode += r_
            state = s_
            print("action: {0:<10}, step: {1:<5}, reward: {2:<10}, speed: {3}".format(actions_all[action], count, r_, 
                                                                                      env.unwrapped.vehicle.speed*np.cos(env.unwrapped.vehicle.heading)))
            # print("high_speed_reward:{}".format(env.config["high_speed_reward"]))
        gif_filename = os.path.join(sac_pm.logs_dir, 'train_log.gif')
        display_frames_as_gif(frames[10:20], gif_filename)
        gif.save(frames, gif_filename, duration=10, unit='s', between='startend')
        T2 = time.perf_counter()
        print(f"{count}轮后停止, 总收益为{r_episode}")
        print(f"程序运行时间: {(T2-T1)} s")
    env.close()
