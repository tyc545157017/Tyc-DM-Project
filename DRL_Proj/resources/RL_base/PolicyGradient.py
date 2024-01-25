import gym
import numpy as np
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
import rl_utils
import matplotlib.pyplot as plt

class PolicyNet(torch.nn.Module):
	def __init__(self, state_dim, hidden1_dim, action_dim):
		super(PolicyNet, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden1_dim)
		self.fc2 = torch.nn.Linear(hidden1_dim, action_dim)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		return F.softmax(self.fc2(x), dim=1)

class PolicyGradient:
	def __init__(self, state_dim, hidden1_dim, action_dim, learning_rate, gamma, device):
		# self.action_dim = action_dim
		self.policy_net = PolicyNet(state_dim, hidden1_dim, action_dim).to(device)
		self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
		self.gamma = gamma
		self.device = device

	def take_action(self, state):
		state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
		probs = self.policy_net(state)
		action_dist = torch.distributions.Categorical(probs)
		action = action_dist.sample()
		return action.item()

	def update(self, transition_dict):
		reward_list = transition_dict['rewards']
		state_list = transition_dict['states']
		action_list = transition_dict['actions']

		G = 0
		self.optimizer.zero_grad()
		for i in reversed(range(len(reward_list))):
			reward = reward_list[i]
			state = torch.tensor(np.array([state_list[i]]), dtype=torch.float).to(self.device)
			action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
			log_prob = torch.log(self.policy_net(state).gather(1, action))
			G = reward + self.gamma * G
			loss = -log_prob * G
			loss.backward()
		self.optimizer.step()

if __name__ == '__main__':
	learning_rate = 1e-3
	num_episodes = 1000
	hidden1_dim = 128
	gamma = 0.98
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	env_name = 'CartPole-v0'
	env = gym.make(env_name)
	env.seed(0)
	torch.manual_seed(0)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n
	agent = PolicyGradient(state_dim, hidden1_dim, action_dim, learning_rate, gamma, device)

	return_list = []
	for i in range(10):
		with tqdm(total=int(num_episodes/10), desc="Iteration:%d"%i) as pbar:
			for i_episode in range(int(num_episodes/10)):
				episode_return = 0
				transition_dict = {
					"states":[],
					"actions":[],
					"rewards":[],
					"next_states":[],
					"dones":[]
				}
				done = False
				state = env.reset()
				while not done:
					action = agent.take_action(state)
					next_state, reward, done, _ = env.step(action)
					transition_dict['states'].append(state)
					transition_dict['actions'].append(action)
					transition_dict['rewards'].append(reward)
					transition_dict['next_states'].append(next_state)
					transition_dict['dones'].append(done)
					state = next_state
					episode_return += reward
				return_list.append(episode_return)
				agent.update(transition_dict)
				if (i_episode + 1)%10 == 0:
					pbar.set_postfix({"episode":'%d'%(num_episodes / 10 * i + i_episode + 1),
									  "return":"%.3f"%np.mean(return_list[-10:])})
				pbar.update(1)

	episodes_list = list(range(len(return_list)))
	plt.plot(episodes_list, return_list)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('PolicyGradient on {}'.format(env_name))
	plt.show()

	mv_return = rl_utils.moving_average(return_list, 9)
	plt.plot(episodes_list, mv_return)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('PolicyGradient on {}'.format(env_name))
	plt.show()


