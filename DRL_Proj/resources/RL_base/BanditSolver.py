import numpy as np
import matplotlib.pyplot as plt
#from KBandit import BernoulliBandit

class BernoulliBandit:
	def __init__(self, K):
		self.probs = np.random.uniform(size=K) #随机生成多臂老虎机获奖概率
		self.best_idx = np.argmax(self.probs) #获奖概率最大的臂
		self.best_prob =  self.probs[self.best_idx] #最大概率
		self.K = K

	def step(self, k):
		if np.random.rand() < self.probs[k]:
			return 1
		else:
			return 0

class Solver:
	def __init__(self, bandit):
		self.bandit = bandit	#多臂老虎机对象
		self.counts = np.zeros(self.bandit.K)	#每根拉杆被选择次数
		self.regret = 0.	#当前步的累计懊悔
		self.actions = []	#维护一个列表，记录每一步的动作
		self.regrets = []	#维护一个列表，记录每一步的累计懊悔

	def update_regret(self, k):
		#计算累计懊悔并保存，k为本次动作选择的拉杆编号
		self.regret += self.bandit.best_prob - self.bandit.probs[k]
		self.regrets.append(self.regret)

	def run_one_step(self):
		#返回当前动作选择哪一根拉杆，由每个具体的策略实现
		raise NotImplementedError

	def run(self, num_steps):
		for _ in range(num_steps):
			k = self.run_one_step()
			self.counts[k] += 1
			self.actions.append(k)
			self.update_regret(k)

class EpsilonGreedy(Solver):
	def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
		super(EpsilonGreedy, self).__init__(bandit)
		self.epsilon = epsilon
		self.estimates = np.array( [init_prob] * self.bandit.K)

	def run_one_step(self):
		if np.random.random() < self.epsilon:
			k = np.random.randint(0, self.bandit.K)
		else:
			k = np.argmax(self.estimates)
		r = self.bandit.step(k)
		self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
		return k

class DecayingEpsilonGreedy(Solver):
	def __init__(self, bandit, init_prob=1.0):
		super(DecayingEpsilonGreedy, self).__init__(bandit)
		self.estimates = np.array( [init_prob] * self.bandit.K)
		self.total_count = 0
	def run_one_step(self):
		self.total_count += 1
		if np.random.random() < 1. / self.total_count:
			k = np.random.randint(0, self.bandit.K)
		else:
			k = np.argmax(self.estimates)
		r = self.bandit.step(k)
		self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
		return k

class UCB(Solver):
	def __init__(self, bandit, coeff, init_prob=1.0):
		super(UCB, self).__init__(bandit)
		self.estimates = np.array([init_prob] * self.bandit.K)
		self.total_count = 0
		self.coeff = coeff

	def run_one_step(self):
		self.total_count += 1
		ucb = self.estimates + self.coeff * np.sqrt(np.log(self.total_count)/(2 * (self.counts + 1)))
		k = np.argmax(ucb)
		r = self.bandit.step(k)
		self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
		return k

class ThompsonSampling(Solver):
	def __init__(self, bandit):
		super(ThompsonSampling, self).__init__(bandit)
		self._a = np.ones(self.bandit.K)
		self._b = np.ones(self.bandit.K)

	def run_one_step(self):
		samples = np.random.beta(self._a, self._b)
		k = np.argmax(samples)
		r = self.bandit.step(k)

		self._a[k] += r
		self._b[k] += (1 - r)
		return k


def plot_result(solvers, solver_names):
	for idx, solver in enumerate(solvers):
		time_list = range(len(solver.regrets))
		plt.plot(time_list, solver.regrets, label=solver_names[idx])
	plt.xlabel('Time steps')
	plt.ylabel('Cumulative regrets')
	plt.title('%d-armed bandit'%solvers[0].bandit.K)
	plt.legend()
	plt.show()

if __name__ == '__main__':
	np.random.seed(1)
	K = 10
	bandit_10_arm = BernoulliBandit(K)
	print("随机生成一个%d臂伯努利老虎机"%K)
	print("获奖概率最大的拉杆为%d号，其获奖概率为%.4f"%
		  (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

######################多个epsilon对比##############################################
	# np.random.seed(1) #为什么要重新设置随机数种子？
	# epsilon = [1e-4, 0.01, 0.1, 0.25, 0.5]
	# epsilon_greedy_solver_lists = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilon]
	# epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilon]
	# for solver in epsilon_greedy_solver_lists:
	# 	solver.run(5000)
	# 	print("epsilon={0}时，累计懊悔为：{1}".format(solver.epsilon, solver.regret))
	# plot_result(epsilon_greedy_solver_lists, epsilon_greedy_solver_names)
##################################################################################

######################单个epsilon#################################################
	# np.random.seed(1)
	# epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
	# epsilon_greedy_solver.run(5000)
	# print('epsilon-贪婪算法的累计懊悔为：', epsilon_greedy_solver.regret)
	# print('前50次懊悔值：', epsilon_greedy_solver.regrets[:50])
	# plot_result([epsilon_greedy_solver], ["EpsilonGreedy"])
##################################################################################

########################随时间衰减的epsilon#########################################
	# np.random.seed(1)
	# epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
	# epsilon_greedy_solver.run(5000)
	# print('epsilon-贪婪算法的累计懊悔为：', epsilon_greedy_solver.regret)
	# print('前50次懊悔值：', epsilon_greedy_solver.regrets[:50])
	# plot_result([epsilon_greedy_solver], ["DecayingEpsilonGreedy"])
##################################################################################

######################## UCB上置信界算法 ###########################################
	# np.random.seed(1)
	# coeff = 1.0
	# epsilon_greedy_solver = UCB(bandit_10_arm, coeff)
	# epsilon_greedy_solver.run(5000)
	# print('epsilon-贪婪算法的累计懊悔为：', epsilon_greedy_solver.regret)
	# print('前50次懊悔值：', epsilon_greedy_solver.regrets[:50])
	# plot_result([epsilon_greedy_solver], ["UCB"])
##################################################################################

######################## 汤普森采样 Thompson Sampling ##############################
	np.random.seed(1)
	epsilon_greedy_solver = ThompsonSampling(bandit_10_arm)
	epsilon_greedy_solver.run(5000)
	print('epsilon-贪婪算法的累计懊悔为：', epsilon_greedy_solver.regret)
	# print('前50次懊悔值：', epsilon_greedy_solver.regrets[:50])
	plot_result([epsilon_greedy_solver], ["Thompson Sampling"])
##################################################################################