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
		# self.start_s = start_s

	def step(self, s, a):
		P_y = s // self.ncol
		P_x = s % self.ncol
		if P_y == self.nrow - 1 and P_x > 0:
			return (0, self.reset(), True)
		P_y_next = min(self.nrow - 1, max(0, P_y + self.action[a][0]))
		P_x_next = min(self.ncol - 1, max(0, P_x + self.action[a][1]))
		next_s = P_y_next * self.ncol + P_x_next
		r = -1
		done = False
		if P_y_next == self.nrow - 1 and P_x_next > 0:
			done = True
			if P_x_next != self.ncol - 1:
				r = -100
		return (r, next_s, done)

	def reset(self):
		self.x_pos = 0
		self.y_pos = self.nrow - 1
		return self.y_pos * self.ncol + self.x_pos

class Sarsa:
	def __init__(self, env, alpha, gamma, epsilon, epoch_num):
		self.env = env
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon
		self.epoch_num = epoch_num
		self.Qsa = [[0.0 for _ in range(len(self.env.action))] for _ in range(self.env.nrow * self.env.ncol)]
		# self.pi = [None for _ in range(self.env.nrow * self.env.ncol)]

	def TD_iteration(self):
		return_list = []
		for i in range(10):
			with tqdm(total=int(self.epoch_num/10), desc='Iteration %d'%i) as pbar:
				for i_episode in range(int(self.epoch_num/10)):
					episode_return = 0
					done = False
					s = self.set_state()
					a = self.get_pi(s)
					while not done:
						r, next_s, done = self.env.step(s, a)
						next_a = self.get_pi(next_s)
						episode_return += r
						self.Qsa[s][a] += self.alpha * (r + self.gamma * self.Qsa[next_s][next_a] - self.Qsa[s][a])
						s = next_s
						a = next_a
					return_list.append(episode_return)
					if (i_episode+1)%10 == 0:
						pbar.set_postfix({'episode':'%d'%(self.epoch_num / 10 * i + i_episode + 1),
										  'return':'%.3f'%np.mean(return_list[-10:])})
					pbar.update(1)
		episodes_list = list(range(len(return_list)))
		plt.plot(episodes_list, return_list)
		plt.xlabel('Episodes')
		plt.ylabel('Returns')
		plt.title('Sarsa on {}'.format('Cliff Walking'))
		plt.show()

	def set_state(self):
		return self.env.reset()

	def get_pi(self, s):
		if np.random.random() < self.epsilon:
			action = np.random.randint(len(self.env.action))
		else:
			action = self.Qsa[s].index(max(self.Qsa[s]))
		return action
		# maxq = max(self.Qsa[s])
		# for a in range(len(self.env.action)):
		# 	if self.Qsa[s][a] == maxq:
		# 		self.pi[s][a] = 1.0 * self.epsilon/len(self.env.action) + 1 - self.epsilon
		# 	else:
		# 		self.pi[s][a] = 1.0 * self.epsilon/len(self.env.action)






if __name__ == "__main__":
	nrow = 4
	ncol = 12
	env = CliffWalkingGame(nrow, ncol)
	epsilon = 0.1
	alpha = 0.1
	gamma = 0.9
	epoch_num = 500
	agent = Sarsa(env, alpha, gamma, epsilon, epoch_num)
	agent.TD_iteration()


	# x, y = [eval(p) for p in input("请输入初始状态的x，y坐标：").split()]
	# start_s = y * env.ncol + x
	# done = False
	# G = 0
	# cnt = 1
	# while not done:
	# 	print("第{}轮".format(cnt).center(20, "*"))
	# 	cnt += 1
	# 	a = eval(input("请输入动作(0：上，1：下，2：左，3：右)："))
	# 	r, next_s, done = env.step(start_s, a)
	# 	# print("next_s = ", next_s)
	# 	G += r
	# 	print("下一状态为({0}, {1})".format(next_s%env.ncol, next_s//env.ncol))
	# 	print("当前收益为：", r)
	# 	print("累计收益为：", G)
	# 	start_s = next_s
	# print("game over！")
