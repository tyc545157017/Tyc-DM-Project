
import torch
import torch.nn.functional as F
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

class ValueNet(torch.nn.Module):
	def __init__(self, state_dim):
		super(ValueNet, self).__init__()
		hidden1_dim = 1024
		hidden2_dim = 512
		self.fc1 = torch.nn.Linear(state_dim, hidden1_dim)
		self.fc2 = torch.nn.Linear(hidden1_dim, hidden2_dim)
		self.fc3 = torch.nn.Linear(hidden2_dim, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return self.fc3(x)

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = 2.0 * torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std


class PPOContinuous:
    ''' 处理连续动作的PPO算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim,
                                         action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample()
        return [action.item()]

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
        rewards = (rewards + 8.0) / 8.0  # 和TRPO一样,对奖励进行修改,方便训练
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        # 动作是正态分布
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()


class PPO:
	def __init__(self, state_dim, action_dim, actor_lr,
				 critic_lr, lmbda, epochs, eps, gamma, device):
		self.actor = PolicyNet(state_dim, action_dim).to(device)
		self.critic = ValueNet(state_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

		self.gamma = gamma
		self.lmbda = lmbda
		self.epochs = epochs
		self.eps = eps
		self.device = device
		self.act_loss = 0
		self.crit_loss = 0

	def take_action(self, state):
		state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
		probs = self.actor(state)
		# print(f"probs: {probs}")
		try:
			action_dist = torch.distributions.Categorical(probs)
		except Exception as e:
			print(f'probs: {probs}')
			print(f'act_loss:{self.act_loss}')
			raise e
		action = action_dist.sample()
		return action.item()

	def update(self, transition_dict):
		self.act_loss = 0
		self.crit_loss = 0
		states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
		actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
		rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
		next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
		dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

		td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
		td_delta = td_target - self.critic(states)

		advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
		old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

		for _ in range(self.epochs):
			log_probs = torch.log(self.actor(states).gather(1, actions))
			ratio = torch.exp(log_probs - old_log_probs)
			surr1 = ratio * advantage
			surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
			actor_loss = torch.mean(-torch.min(surr1, surr2))
			critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

			self.act_loss += actor_loss.item()
			self.crit_loss += critic_loss.item()

			self.actor_optimizer.zero_grad()
			self.critic_optimizer.zero_grad()

			actor_loss.backward()
			critic_loss.backward()

			self.actor_optimizer.step()
			self.critic_optimizer.step()
		self.act_loss = self.act_loss/self.epochs
		self.crit_loss = self.crit_loss/self.epochs

def ppo_train(env, agent, num_episodes, 
			  drl_type, model_path, retrain=False):
	print("\n开始{}训练".format(drl_type))
	print("状态维度：{}, 动作维度：{}".format(state_dim, action_dim))
	actor_model_path = model_path.rsplit('.', 1)[0] + '_actor.pkl'
	critic_model_path = model_path.rsplit('.', 1)[0] + '_critic.pkl'
	print(model_path)
	if retrain:
		agent.actor.load_state_dict(torch.load(actor_model_path))
		agent.critic.load_state_dict(torch.load(critic_model_path))
		print('load model to retrian')
	
	episode_len = []
	return_list = []
	act_loss_list = []
	crit_loss_list = []
	for i in range(10):
		with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
			for i_episode in range(int(num_episodes/10)):
				episode_return = 0
				transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
				state, _ = env.reset()
				state = state.flatten()
				done = False
				count = 0
				while not done:
					count += 1
					action = agent.take_action(state)
					next_state, reward, terminated, truncated, _ = env.step(action)
					next_state = next_state.flatten()
					if terminated or truncated:
						done = True
					transition_dict['states'].append(state)
					transition_dict['actions'].append(action)
					transition_dict['next_states'].append(next_state)
					transition_dict['rewards'].append(reward)
					transition_dict['dones'].append(done)
					state = next_state
					episode_return += reward
				episode_len.append(count)
				return_list.append(episode_return)
				agent.update(transition_dict)
				act_loss_list.append(agent.act_loss)
				crit_loss_list.append(agent.crit_loss)
				if (i_episode+1) % 10 == 0:
					pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
				pbar.update(1)
	torch.save(agent.actor.state_dict(), actor_model_path)
	torch.save(agent.critic.state_dict(), critic_model_path)
	return (return_list, episode_len, act_loss_list, crit_loss_list)

if __name__ == '__main__':
	""" original
	actor_lr = 5e-5
	critic_lr = 5e-4
	num_episodes = 500
	hidden_dim = 128
	gamma = 0.98
	lmbda = 0.95
	epochs = 10
	eps = 0.2
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
		"cpu")

	env_name = 'CartPole-v0'
	env = gym.make(env_name)
	env.seed(0)
	torch.manual_seed(0)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n
	agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
				epochs, eps, gamma, device)

	return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

	# actor_lr = 1e-4
	# critic_lr = 5e-3
	# num_episodes = 2000
	# hidden_dim = 128
	# gamma = 0.9
	# lmbda = 0.9
	# epochs = 10
	# eps = 0.2
	# device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
	# 	"cpu")
	#
	# env_name = 'Pendulum-v0'
	# env = gym.make(env_name)
	# env.seed(0)
	# torch.manual_seed(0)
	# state_dim = env.observation_space.shape[0]
	# action_dim = env.action_space.shape[0]  # 连续动作空间
	# agent = PPOContinuous(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
	# 					  lmbda, epochs, eps, gamma, device)
	#
	# return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
	#
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
		"render_agent": True
	}

	"""
	
	retrain = False
	train = True
	train_version = -1
	drl_type = "PPO"
	# env_name = 'highway-v0'
	env_name = 'tyc-highway-v0'
	lr = {
		'actor_lr': 1e-4,
		'critic_lr': 5e-3
	}
	drl_param = {
		'gamma': 0.9,
		'lmbda': 0.95,
		'epochs': 100,
		'eps': 0.2
	}
	num_episodes = 6000
	ppo_pm = ParamManager(env_name, drl_type, drl_param, lr, num_episodes, train_version=train_version, train=train)
	model_path = os.path.join(ppo_pm.model_dir, 'model.pkl')
	config_path = os.path.join(ppo_pm.config_dir, 'config.yaml')
	logs_path = os.path.join(ppo_pm.logs_dir, 'train_log.csv')
	config = ppo_pm.config
	# pprint(config)
	# print('split'.center(60, '*'))

	random_seed = 0

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	env = gym.make(env_name, render_mode="rgb_array") #
	env.unwrapped.configure(config)
	obs, info = env.reset(seed=random_seed)
	np.random.seed(random_seed)
	random.seed(random_seed)
	torch.manual_seed(random_seed)
	# env.render()
	# state_dim = env.observation_space.shape[0]
	# action_dim = env.action_space.n
	if train:
		# env.config["high_speed_reward"] = 1.0
		if torch.cuda.is_available():
			print("Use GPU train")
		# env.render_mode = None
		if os.path.exists(model_path) and not retrain:
			raise ValueError("model path has existed, maybe override, please check the name!")
		pprint(env.unwrapped.config)
		T1 = time.perf_counter()
		state_shape = env.observation_space.shape
		if len(state_shape) == 2:
			state_dim = state_shape[0] * state_shape[1]
		else:
			raise ValueError
		action_dim = env.action_space.n
		agent = PPO(state_dim, action_dim, lr['actor_lr'], lr['critic_lr'],
					drl_param['lmbda'], drl_param['epochs'],
					drl_param['eps'], drl_param['gamma'], device)
		return_lists = ppo_train(env, agent, num_episodes, 
						   drl_type, model_path=model_path, retrain=retrain)
		T2 = time.perf_counter()
		print(f"trian time: {(T2-T1)/3600} hours")
		ppo_pm.save_config_file()
		if not retrain:
			ppo_pm.update_version_files()
		save_train_log(logs_path, **{"reward": return_lists[0], "episode_len": return_lists[1], "act_loss": return_lists[2], "crit_loss": return_lists[3]})
		episodes_list = list(range(len(return_lists[0])))
		mv_return = rl_utils.moving_average(return_lists[0], 9)
		plt.plot(episodes_list, mv_return, label='{}'.format(return_lists[1]))
		plt.legend()
		plt.xlabel('Episodes')
		plt.ylabel('Returns')
		plt.title('PPO on {}'.format(env_name))
		plt.show()
	else:
		actor_model_path = model_path.rsplit('.', 1)[0] + '_actor.pkl'
		critic_model_path = model_path.rsplit('.', 1)[0] + '_critic.pkl'
		print(model_path)
		frames = []
		env.config["duration"] = 180
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
		agent = PPO(state_dim, action_dim, lr['actor_lr'], lr['critic_lr'],
					drl_param['lmbda'], drl_param['epochs'],
					drl_param['eps'], drl_param['gamma'], device)
		print(f"load model: {model_path}")
		agent.actor.load_state_dict(torch.load(actor_model_path))
		agent.critic.load_state_dict(torch.load(critic_model_path))
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
			# print(action)
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
		gif_filename = os.path.join(ppo_pm.logs_dir, 'train_log.gif')
		display_frames_as_gif(frames[10:20], gif_filename)
		gif.save(frames, gif_filename, duration=10, unit='s', between='startend')
		T2 = time.perf_counter()
		print(f"{count}轮后停止, 总收益为{r_episode}")
		print(f"程序运行时间: {(T2-T1)} s")
	env.close()

