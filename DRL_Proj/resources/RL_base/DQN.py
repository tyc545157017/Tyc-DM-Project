import torch
import torch.nn as nn
import numpy as np
import collections
import matplotlib.pyplot as plt
import random
import gym
from tqdm import tqdm
import rl_utils

class ReplayBuffer:
	def __init__(self, capacity):
		self.buffer = collections.deque(maxlen=capacity)

	def add(self, state, action, reward, next_state, done):
		self.buffer.append((state, action, reward, next_state, done))

	def sample(self, batch_size):
		transitions = random.sample(self.buffer, batch_size)
		state, action, reward, next_state, done = zip(*transitions)
		return np.array(state), action, reward, np.array(next_state), done

	def size(self):
		return len(self.buffer)

class Qnet(nn.Module):
	def __init__(self, state_dim, hidden1_dim, action_dim):
		super(Qnet, self).__init__()
		self.fc1 = nn.Linear(state_dim, hidden1_dim)
		self.fc2 = nn.Linear(hidden1_dim, action_dim)

	def forward(self, x):
		x = nn.functional.relu(self.fc1(x))
		return self.fc2(x)

class VAnet(nn.Module):
	def __init__(self, state_dim, hidden1_dim, action_dim):
		super(VAnet, self).__init__()
		self.fc1 = nn.Linear(state_dim, hidden1_dim)
		self.fc_A = nn.Linear(hidden1_dim, action_dim)
		self.fc_V = nn.Linear(hidden1_dim, 1)

	def forward(self, x):
		x = nn.functional.relu(self.fc1(x))
		A = self.fc_A(x)
		V = self.fc_V(x)
		Q = A + V - A.mean(1).view(-1, 1)
		return Q

class DQN:
	def __init__(self, state_dim, hidden1_dim, action_dim,
				 learning_rate, gamma, epsilon, target_update, device, dqn_type='NormalDQN'):
		# self.state_dim = state_dim
		# self.hidden1_dim = hidden1_dim
		self.action_dim = action_dim
		if dqn_type == 'DuelingDQN':
			self.q_net = VAnet(state_dim, hidden1_dim, self.action_dim).to(device)
			self.target_q_net = VAnet(state_dim, hidden1_dim, self.action_dim).to(device)
		else:
			self.q_net = Qnet(state_dim, hidden1_dim, self.action_dim).to(device)
			self.target_q_net = Qnet(state_dim, hidden1_dim, self.action_dim).to(device)
		# self.learning_rate = learning_rate

		self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
		# self.criterion = nn.MSELoss()
		self.gamma = gamma
		self.epsilon = epsilon
		self.target_update = target_update
		self.device = device
		self.count = 0
		self.dqn_type = dqn_type

	def take_action(self, state):
		if np.random.random() < self.epsilon:
			action = np.random.randint(self.action_dim)
		else:
			state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
			# action = np.argmax(self.q_net(state).data.numpy())
			action = self.q_net(state).argmax().item()
		return action

	def best_action(self, state):
		state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
		action = self.target_q_net(state).argmax().item()
		return action

	def max_q_value(self, state):
		state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
		return self.q_net(state).max().item()

	def update(self, transitions_dict):
		states = torch.tensor(transitions_dict['states'], dtype=torch.float).to(self.device)
		actions = torch.tensor(transitions_dict['actions']).view(-1, 1).to(self.device)
		rewards = torch.tensor(transitions_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
		next_states = torch.tensor(transitions_dict['next_states'], dtype=torch.float).to(self.device)
		dones = torch.tensor(transitions_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

		q_values = self.q_net(states).gather(1, actions)
		if self.dqn_type == "DoubleDQN":
			next_actions = self.q_net(next_states).max(1)[1].view(-1, 1)
			max_next_q_values = self.target_q_net(next_states).gather(1, next_actions)
		else:
			max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
		q_target = rewards + self.gamma * max_next_q_values * (1 - dones)

		dqn_loss = torch.mean(nn.functional.mse_loss(q_values, q_target))
		self.optimizer.zero_grad()
		dqn_loss.backward()
		self.optimizer.step()

		if self.count % self.target_update == 0:
			self.target_q_net.load_state_dict(self.q_net.state_dict())

		self.count += 1


# class Double_DQN:
# 	def __init__(self, state_dim, hidden1_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
# 		self.action_dim = action_dim
# 		self.gamma = gamma
# 		self.epsilon = epsilon
# 		# self.learning_rate = learning_rate
# 		self.q_net = Qnet(state_dim, hidden1_dim, action_dim)
# 		self.target_q_net = Qnet(state_dim, hidden1_dim, action_dim)
# 		self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
# 		self.target_update = target_update
# 		self.device = device
# 		self.count = 0
#
# 	def take_action(self, state):
# 		if np.random.random() < self.epsilon:
# 			action = np.random.randint(self.action_dim)
# 		else:
# 			state = torch.tensor([state], dtype=torch.float).to(self.device)
# 			action = self.q_net(state).argmax().item()
# 		return action
#
# 	def best_actions(self, states):
# 		# state = torch.tensor([state], dtype=torch.float).to(self.device)
# 		actions = self.q_net(states).max(1)[1].view(-1, 1).to(self.device)
# 		return actions
#
# 	def best_action(self, state):
# 		state = torch.tensor([state], dtype=torch.float).to(self.device)
# 		action = self.target_q_net(state).argmax().item()
# 		return action
#
# 	def update(self, transitions_dict):
# 		states = torch.tensor(transitions_dict['states'], dtype=torch.float).to(self.device)
# 		actions = torch.tensor(transitions_dict['actions']).view(-1, 1).to(self.device)
# 		rewards = torch.tensor(transitions_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
# 		next_states = torch.tensor(transitions_dict['next_states'], dtype=torch.float).to(self.device)
# 		dones = torch.tensor(transitions_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
#
# 		q_values = self.q_net(states).gather(1, actions)
#
# 		maxq_actions = self.best_actions(next_states)
#
# 		max_next_q_values = self.target_q_net(next_states).gather(1, maxq_actions)
#
# 		target_q_values = rewards + self.gamma*max_next_q_values*(1-dones)
#
# 		dqn_loss = torch.mean(nn.functional.mse_loss(q_values, target_q_values))
#
# 		self.optimizer.zero_grad()
#
# 		dqn_loss.backward()
#
# 		self.optimizer.step()
#
# 		if self.count % self.target_update == 0:
# 			self.target_q_net.load_state_dict(self.q_net.state_dict())
#
# 		self.count += 1
def dis_to_con(discrete_aciton, env, action_dim):
	action_lowbound = env.action_space.low[0]
	action_upbound = env.action_space.high[0]
	return action_lowbound + (discrete_aciton / (action_dim - 1)) * (action_upbound - action_lowbound)

def dqn_train(env, replay_buffer, num_episodes,
			  hidden1_dim, lr, minimal_size, batch_size,
			  gamma, epsilon, target_update, device, dqn_type, env_name):
	state_dim = env.observation_space.shape[0]
	action_dim = 21
	print("\n开始{}训练".format(dqn_type))
	print("状态维度：{}, 动作维度：{}".format(state_dim, action_dim))
	agent = DQN(state_dim, hidden1_dim, action_dim, lr, gamma, epsilon, target_update, device, dqn_type)
	return_list = []
	max_q_value_list = []
	max_q_value = 0
	episode_len = []
	for i in range(10):
		with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
			for i_episode in range(int(num_episodes / 10)):
				episode_return = 0
				state, _ = env.reset()
				print(state)
				print(np.shape(state))
				done = False
				count = 0
				# print("i am here")
				while not done:
					# print(done)
					count += 1
					action = agent.take_action(state)
					# env.render()
					max_q_value = agent.max_q_value(state)*0.005 + max_q_value*0.995
					max_q_value_list.append(max_q_value)
					action_continuous = dis_to_con(action, env, action_dim)
					next_state, reward, _, done, _ = env.step([action_continuous])
					# print("next_state:{}, reward:{}".format(next_state, reward))
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
				episode_len.append(count)
				return_list.append(episode_return)
				if (i_episode + 1) % 10 == 0:
					pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
									  'return': '%.3f' % np.mean(return_list[-10:])})
				pbar.update(1)
	torch.save(agent.target_q_net.state_dict(), '{0}_model_for_{1}.pkl'.format(dqn_type, env_name))
	return (return_list, dqn_type, max_q_value_list)

if __name__ == "__main__":
	task = 'train'
	dqn_type = ["DQN", "DoubleDQN", "DuelingDQN"]#, "DuelingDQN"
	# dqn_type = ["DuelingDQN"]
	lr = 1e-2
	num_episodes = 200
	hidden1_dim = 128
	gamma = 0.98
	epsilon = 0.01
	target_update = 50
	buffer_size = 5000
	minimal_size = 1000
	batch_size = 64
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	env_name = 'Pendulum-v1'
	env = gym.make(env_name, render_mode="human")
	np.random.seed(0)
	random.seed(0)
	# env.seed(0)
	torch.manual_seed(0)
	# env = env.unwrapped()
	# env.render()
	replay_buffer = ReplayBuffer(buffer_size)
	# state_dim = env.observation_space.shape[0]
	# action_dim = env.action_space.n
	return_lists = []
	if task == 'train':
		for i in range(len(dqn_type)):
			return_lists.append(dqn_train(env, replay_buffer, num_episodes, hidden1_dim, lr, minimal_size,
										  batch_size, gamma, epsilon, target_update, device, dqn_type[i], env_name))
		for i in range(len(return_lists)):
			episodes_list = list(range(len(return_lists[i][0])))
			mv_return = rl_utils.moving_average(return_lists[i][0], 9)
			plt.plot(episodes_list, mv_return, label='{}'.format(return_lists[i][1]))
		plt.legend()
		plt.xlabel('Episodes')
		plt.ylabel('Returns')
		plt.title('DQN on {}'.format(env_name))
		plt.show()

		for i in range(len(return_lists)):
			frames_list = list(range(len(return_lists[i][2])))
			# mv_return = rl_utils.moving_average(return_lists[i][2], 9)
			plt.plot(frames_list, return_lists[i][2], label='{}'.format(return_lists[i][1]))
		plt.legend()
		plt.axhline(0, c='orange', ls='--')
		plt.axhline(10, c='red', ls='--')
		plt.xlabel('Frames')
		plt.ylabel('Q value')
		plt.title('DQN on {}'.format(env_name))
		plt.show()
	else:
		state_dim = env.observation_space.shape[0]
		action_dim = 11
		for i in range(len(dqn_type)):
			print("\n开始{}测试".format(dqn_type[i]))
			print("状态维度：{}, 动作维度：{}".format(state_dim, action_dim))
			agent = DQN(state_dim, hidden1_dim, action_dim, lr, gamma, epsilon, target_update, device, dqn_type[i])
			agent.target_q_net.load_state_dict(torch.load('{0}_model_for_{1}.pkl'.format(dqn_type[i], env_name)))
			state, _ = env.reset()
			done = False
			count = 0
			r_episode = 0
			while not done:
				count += 1
				action = agent.best_action(state)
				action_continuous = dis_to_con(action, env, action_dim)
				# env.render()
				s_, r, _, done, _ = env.step([action_continuous])
				r_episode += r
				state = s_
			print("%d轮后停止"%count)
			print("总收益为%d"%r_episode)
	# agent = Double_DQN(state_dim, hidden1_dim, action_dim, lr, gamma, epsilon, target_update, device)
	# return_list = []
	# if task == 'train':
	# 	for i in range(10):
	# 		with tqdm(total=int(num_episodes/10), desc='Iteration %d'%i) as pbar:
	# 			for i_episode in range(int(num_episodes/10)):
	# 				episode_return = 0
	# 				state = env.reset()
	# 				done = False
	# 				while not done:
	# 					action = agent.take_action(state)
	# 					env.render()
	# 					next_state, reward, done, _ = env.step(action)
	# 					replay_buffer.add(state, action, reward, next_state, done)
	# 					state = next_state
	# 					episode_return += reward
	# 					if replay_buffer.size() > minimal_size:
	# 						states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
	# 						transitions_dict={
	# 							"states":states,
	# 							"actions":actions,
	# 							"rewards":rewards,
	# 							"next_states":next_states,
	# 							"dones":dones
	# 						}
	# 						agent.update(transitions_dict)
	# 				return_list.append(episode_return)
	# 				if (i_episode + 1) % 10 == 0:
	# 					pbar.set_postfix({'episode':'%d'%(num_episodes/10*i + i_episode + 1),
	# 									  'return':'%.3f'%np.mean(return_list[-10:])})
	# 				pbar.update(1)
	#
	# 	torch.save(agent.target_q_net.state_dict(), 'double_dqn_model.pkl')
	# 	episodes_list = list(range(len(return_list)))
	# 	plt.plot(episodes_list, return_list)
	# 	plt.xlabel('Episodes')
	# 	plt.ylabel('Returns')
	# 	plt.title('DQN on {}'.format(env_name))
	# 	plt.show()
	#
	# 	mv_return = rl_utils.moving_average(return_list, 9)
	# 	plt.plot(episodes_list, mv_return)
	# 	plt.xlabel('Episodes')
	# 	plt.ylabel('Returns')
	# 	plt.title('DQN on {}'.format(env_name))
	# 	plt.show()
	# else:
	# 	agent.target_q_net.load_state_dict(torch.load('double_dqn_model.pkl'))
	# 	state = env.reset()
	# 	done = False
	# 	count = 0
	# 	r_episode = 0
	# 	for i in range(20):
	# 		print("第%d轮".center(50, '*')%i)
	# 		while not done:
	# 			count += 1
	# 			action = agent.best_action(state)
	# 			env.render()
	# 			s_, r, done, _ = env.step(action)
	# 			r_episode += r
	# 			state = s_
	# 		print("%d轮后停止"%count)
	# 		print("总收益为%d"%r_episode)

