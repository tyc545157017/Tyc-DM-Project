import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils

class PolicyNet(torch.nn.Module):
	def __init__(self, state_dim, hidden_dim, action_dim):
		super(PolicyNet, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		return F.softmax(self.fc2(x), dim=1)

class ValueNet(torch.nn.Module):
	def __init__(self, state_dim, hidden_dim):
		super(ValueNet, self).__init__()
		self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
		self.fc2 = torch.nn.Linear(hidden_dim, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		return self.fc2(x)

class ActorCritic:
	def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
		# self.action_dim = action_dim
		self.actor_net = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)

		self.critic_net = ValueNet(state_dim, hidden_dim).to(device)
		self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_lr)

		self.gamma = gamma
		self.device = device

	def take_action(self, state):
		state = torch.tensor([state], dtype=torch.float).to(self.device)
		probs = self.actor_net(state)
		action_dist = torch.distributions.Categorical(probs)
		action = action_dist.sample()
		return action.item()

	def update(self, transition_dict):
		states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
		actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
		rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
		next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
		dones = torch.tensor(transition_dict['dones'] , dtype=torch.float).view(-1, 1).to(self.device)

		td_target = rewards + self.gamma * self.critic_net(next_states) * (1 - dones)
		td_delta = td_target - self.critic_net(states)
		log_probs = torch.log(self.actor_net(states).gather(1, actions))
		actor_loss = torch.mean(-log_probs * td_delta.detach())
		critic_loss = torch.mean(F.mse_loss(self.critic_net(states), td_target.detach()))
		self.actor_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		actor_loss.backward()
		critic_loss.backward()

		self.actor_optimizer.step()
		self.critic_optimizer.step()

if __name__ == '__main__':
	actor_lr = 1e-3
	critic_lr = 1e-2
	num_episodes = 1000
	hidden_dim = 128
	gamma = 0.98
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	env_name = 'CartPole-v0'
	env = gym.make(env_name)
	env.seed(0)
	torch.manual_seed(0)
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n
	agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)

	return_list	= rl_utils.train_on_policy_agent(env, agent, num_episodes)
	episodes_list = list(range(len(return_list)))
	plt.plot(episodes_list, return_list)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('Actor-Critic on {}'.format(env_name))
	plt.show()

	mv_return = rl_utils.moving_average(return_list, 9)
	plt.plot(episodes_list, mv_return)
	plt.xlabel('Episodes')
	plt.ylabel('Returns')
	plt.title('Actor-Critic on {}'.format(env_name))
	plt.show()