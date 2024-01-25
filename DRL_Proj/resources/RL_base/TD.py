import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(0)

class CliffWalkingGame:
	def __init__(self,  nrow=4, ncol=12):
		self.nrow = nrow
		self.ncol = ncol
		self.action = [[-1, 0], [1, 0], [0, -1], [0, 1]]
		self.x_pos = 0
		self.y_pos = self.nrow - 1

	def step(self, a):
		# if self.y_pos == self.nrow - 1 and self.x_pos > 0:
		# 	return (0, self.reset(), True)
		self.y_pos = min(self.nrow - 1, max(0, self.y_pos + self.action[a][0]))
		self.x_pos = min(self.ncol - 1, max(0, self.x_pos + self.action[a][1]))
		next_s = self.y_pos * self.ncol + self.x_pos
		r = -1
		done = False
		if self.y_pos == self.nrow - 1 and self.x_pos > 0:
			done = True
			if self.x_pos != self.ncol - 1:
				r = -100
		return (r, next_s, done)

	def reset(self):
		self.x_pos = 0
		self.y_pos = self.nrow - 1
		return self.y_pos * self.ncol + self.x_pos

class Sarsa:
	def __init__(self, env, epsilon, alpha, gamma, n_action = 4):
		self.env = env
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.n_action = n_action
		self.Qsa = np.zeros([self.env.nrow * self.env.ncol, n_action])

	def update(self, s0, a0, r, s1, a1):
		TD_error = r + self.gamma * self.Qsa[s1][a1] - self.Qsa[s0][a0]
		self.Qsa[s0][a0] += self.alpha * TD_error

	def take_action(self, state):
		if np.random.random() < self.epsilon:
			action = np.random.randint(self.n_action)
		else:
			action = np.argmax(self.Qsa[state])
		return action

	def best_action(self, state):
		maxq = np.max(self.Qsa[state])
		action = [0 for _ in range(self.n_action)]
		for i in range(self.n_action):
			if self.Qsa[state][i] == maxq:
				action[i] = 1
		return action

class nstep_Sarsa:
	def __init__(self, env, n_step, epsilon, alpha, gamma, n_action=4):
		self.env = env
		self.n_step = n_step
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.n_action = n_action
		self.Q_table = np.zeros([self.env.nrow*self.env.ncol, self.n_action])
		self.state_list = []
		self.action_list = []
		self.reward_list = []

	def take_action(self, state):
		if np.random.random() < self.epsilon:
			action = np.random.randint(self.n_action)
		else:
			action = np.argmax(self.Q_table[state])
		return action

	def best_action(self, state):
		maxq = np.max(self.Q_table[state])
		action = [0 for _ in range(self.n_action)]
		for i in range(self.n_action):
			if self.Q_table[state][i] == maxq:
				action[i] = 1
		return action

	def update(self, s0, a0, r, s1, r1, done):
		self.state_list.append(s0)
		self.action_list.append(a0)
		self.reward_list.append(r)
		if len(self.state_list) == self.n_step:
			G = self.Q_table[s1, r1]
			for i in reversed(range(self.n_step)):
				G = self.gamma*G + self.reward_list[i]
				if done and i > 0:
					s = self.state_list[i]
					a = self.action_list[i]
					self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
			s = self.state_list.pop(0)
			a = self.action_list.pop(0)
			self.reward_list.pop(0)
			self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
		if done:
			self.state_list = []
			self.action_list = []
			self.reward_list = []

class Qlearning:
	def __init__(self, env, epsilon, alpha, gamma, n_action=4):
		self.env = env
		self.epsilon = epsilon
		self.alpha = alpha
		self.gamma = gamma
		self.n_action = n_action
		self.Q_table = np.zeros([self.env.nrow*self.env.ncol, self.n_action])

	def take_action(self, state):
		if np.random.random() < self.epsilon:
			action = np.random.randint(self.n_action)
		else:
			action = np.argmax(self.Q_table[state])
		return action

	def best_action(self, state):
		action = [0 for _ in range(self.n_action)]
		Qmax = np.max(self.Q_table[state])
		for i in range(self.n_action):
			if self.Q_table[state, i] == Qmax:
				action[i] = 1
		return action

	def update(self, s0, a0, r, s1):
		# a1 = np.argmax(self.best_action(s1))
		TD_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
		self.Q_table[s0, a0] += self.alpha * TD_error


def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

if __name__ == "__main__":
	nrow = 4
	ncol = 12
	epsilon = 0.1
	alpha = 0.1
	gamma = 0.9
	n_step = 5
	env = CliffWalkingGame(nrow, ncol)
	# agent = nstep_Sarsa(env, n_step, epsilon, alpha, gamma)
	agent = Qlearning(env, epsilon, alpha, gamma)
	episodes_num = 500

	return_list = []
	for i in range(10):
		with tqdm(total=int(episodes_num / 10), desc='Iteration: %d'%i) as pbar:
			for i_episode in range(int(episodes_num/10)):
				episode_return = 0
				done = False
				now_s = env.reset()
				# now_a = agent.take_action(now_s)
				while not done:
					now_a = agent.take_action(now_s)
					reward, next_s, done = env.step(now_a)
					episode_return += reward
					# next_a = agent.take_action(next_s)
					# agent.update(now_s, now_a, reward, next_s, next_a, done)
					agent.update(now_s, now_a, reward, next_s)
					now_s = next_s
					# now_a = next_a
				return_list.append(episode_return)
				if (i_episode + 1)%10 ==0:
					pbar.set_postfix({'episode':"%d"%(episodes_num / 10 * i + i_episode + 1),
									  'return':'%.3f'%np.mean(return_list[-10:])})
				pbar.update(1)

	action_meaning = ['^', 'v', '<', '>']
	print('Sarsa算法最终收敛得到的策略为：')
	print_agent(agent, env, action_meaning, list(range(37, 47)), [47])

	episodes_list = list(range(len(return_list)))
	plt.plot(episodes_list, return_list)
	plt.xlabel("Episodes")
	plt.ylabel("Return")
	plt.title("Q learning on {}".format("CliffWalking"))
	plt.show()

