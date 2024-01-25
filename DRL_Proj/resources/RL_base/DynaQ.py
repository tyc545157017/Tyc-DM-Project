import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import time

from TD import *

np.random.seed(0)
random.seed(0)

# class Dyna_Q:
# 	def __init__(self, env, QL):
# 		self.env = env
# 		self.QL = QL
# 		self.Model = [[[] for _ in range(self.QL.n_action)] for _ in range(self.env.nrow * self.env.ncol)]






if __name__ == "__main__":
	nrow = 4
	ncol = 12
	epsilon = 0.1
	alpha = 0.1
	gamma = 0.9
	n_step = 5
	QL_number = 2
	env = CliffWalkingGame(nrow, ncol)
	# agent = nstep_Sarsa(env, n_step, epsilon, alpha, gamma)
	agent = Qlearning(env, epsilon, alpha, gamma)
	episodes_num = 300

	Model = {}
	return_list = []
	for i in range(10):
		with tqdm(total=int(episodes_num / 10), desc='Iteration: %d'%i) as pbar:
			for i_episode in range(int(episodes_num/10)):
				episode_return = 0
				done = False
				now_s = env.reset()
				while not done:
					now_a = agent.take_action(now_s)
					reward, next_s, done = env.step(now_a)
					agent.update(now_s, now_a, reward, next_s)
					Model[(now_s, now_a)] = (reward, next_s)
					episode_return += reward
					for _ in range(QL_number):
						(s, a), (r, s_) = random.choice(list(Model.items()))
						agent.update(s, a, r, s_)
					# agent.update(now_s, now_a, reward, next_s, next_a, done)
					now_s = next_s
				return_list.append(episode_return)
				if (i_episode + 1)%10 ==0:
					pbar.set_postfix({'episode':"%d"%(episodes_num / 10 * i + i_episode + 1),
									  'return':'%.3f'%np.mean(return_list[-10:])})
				pbar.update(1)

	action_meaning = ['^', 'v', '<', '>']
	print('Sarsa算法最终收敛得到的策略为：')
	print_agent(agent, env, action_meaning, list(range(37, 47)), [47])

	episodes_list = list(range(len(return_list)))
	plt.plot(episodes_list, return_list, label = 'Dyna Q')
	# plt.plot(range(100), range(100), label='test')
	plt.legend()
	plt.xlabel("Episodes")
	plt.ylabel("Return")
	plt.title("Dyna Q on {}".format("CliffWalking"))
	plt.show()

