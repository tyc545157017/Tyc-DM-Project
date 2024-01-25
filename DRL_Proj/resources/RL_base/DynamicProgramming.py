import copy
import gym

class CliffWalkingEnv:
	def __init__(self, ncol=12, nrow=4):
		self.nrow = nrow
		self.ncol = ncol
		self.P = self.CreatP()

	def CreatP(self):
		P = [[[] for _ in range(4)] for _ in range(self.nrow * self.ncol)]
		#P[state][action] = [(p, next_state, reward, done)]
		change = [[0, -1], [0, 1], [-1, 0], [1, 0]] #up down left right
		for i in range(self.nrow):
			for j in range(self.ncol):
				for a in range(4):
					if i == self.nrow - 1 and j > 0:
						P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
						continue
					next_x = min(self.ncol - 1, max(0, j + change[a][0]))
					next_y = min(self.nrow - 1, max(0, i + change[a][1]))
					next_state = next_y * self.ncol + next_x
					r = -1
					done = False
					if next_y == self.nrow - 1 and next_x > 0:
						done = True
						if next_x != self.ncol - 1:
							r = -100
					P[i * self.ncol + j][a] = [(1, next_state, r, done)]
		return P

class PolicyIteration:
	def __init__(self, env, theta, gamma):
		self.env = env
		self.v = [0] * self.env.nrow * self.env.ncol
		self.pi = [[0.25, 0.25, 0.25, 0.25] for _ in range(self.env.nrow * self.env.ncol)]
		self.theta = theta
		self.gamma = gamma

	def policy_evaluation(self):
		cnt = 1
		while 1 :
			max_diff = 0
			new_v = [0] * self.env.nrow * self.env.ncol
			for s in range(self.env.nrow * self.env.ncol):
				qsa_list = []
				for a in range(4):
					qsa = 0
					for res in self.env.P[s][a]:
						p, next_s, r, done = res
						qsa += p * (r + self.gamma * self.v[next_s] * (1 - done))
					qsa_list.append(self.pi[s][a] * qsa)
				new_v[s] = sum(qsa_list)
				max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
			self.v = new_v
			if max_diff < self.theta:
				break
			cnt += 1
		print("策略评估经历了%d轮"%cnt)

	def policy_improvement(self):
		for s in range(self.env.nrow * self.env.ncol):
			qsa_list = []
			for a in range(4):
				qsa = 0
				for res in self.env.P[s][a]:
					p, next_s, r, done = res
					qsa += p * (r + self.gamma * self.v[next_s] * (1 - done))
				qsa_list.append(qsa)
			maxq = max(qsa_list)
			maxq_cnt = qsa_list.count(maxq)
			self.pi[s] = [1 / maxq_cnt if q == maxq else 0 for q in qsa_list]
		print("策略提升完成")
		return self.pi

	def policy_iteration(self):
		while 1:
			self.policy_evaluation()
			old_pi = copy.deepcopy(self.pi)
			new_pi = self.policy_improvement()
			if old_pi == new_pi:
				break


class ValueIteration:
	def __init__(self, env, theta, gamma):
		self.env = env
		self.v = [0] * env.nrow * env.ncol
		#self.pi = [[0.0 , 0.0, 0.0, 0.0] for _ in range(env.nrow * env.ncol)]
		self.pi = [None for _ in range(env.nrow * env.ncol)]
		self.theta = theta
		self.gamma = gamma

	def value_iteration(self):
		cnt = 1
		while 1:
			max_diff = 0
			v_new = [0] * self.env.nrow * self.env.ncol
			for s in range(self.env.nrow * self.env.ncol):
				qsa_list = []
				for a in range(4):
					qsa = 0
					for res in self.env.P[s][a]:
						p, next_s, r, done = res
						qsa += p * (r + self.gamma * self.v[next_s] * (1 - done))
					qsa_list.append(qsa)
				v_new[s] = max(qsa_list)
				max_diff = max(max_diff, abs(v_new[s] - self.v[s]))
			self.v = v_new
			if max_diff < theta:break
			cnt += 1
		print("状态价值评估花费%d轮"%cnt)
		self.get_policy()

	def get_policy(self):
		for s in range(self.env.nrow * self.env.ncol):
			qsa_list = []
			for a in range(4):
				qsa = 0
				for res in self.env.P[s][a]:
					p, next_s, r, done = res
					qsa += p * (r + self.gamma * self.v[next_s] * (1 - done))
				qsa_list.append(qsa)
			# a_idx = qsa_list.index(max(qsa_list))
			# self.pi[s][a_idx] = 1.0
			maxq = max(qsa_list)
			maxq_cnt = qsa_list.count(maxq)
			self.pi[s] = [1 / maxq_cnt if q == maxq else 0 for q in qsa_list]
		print("成功获取策略")





def print_agent(agent, action_meaning, disaster=[], end=[]):
	print("状态价值")
	for i in range(agent.env.nrow):
		for j in range(agent.env.ncol):
			print('%6.6s'%('%.4f'%agent.v[i*agent.env.ncol + j]), end=' ')
		print()

	print()

	print("最优策略")
	for i in range(agent.env.nrow):
		for j in range(agent.env.ncol):
			if i * agent.env.ncol + j in disaster:
				print("****", end=' ')
			elif i * agent.env.ncol + j in end:
				print("EEEE", end=' ')
			else:
				pi_str = ''
				for a in range(4):
					if agent.pi[i * agent.env.ncol + j][a] > 0:
						pi_str += action_meaning[a]
					else:
						pi_str += 'O'
				print(pi_str, end=' ')
		print()

if __name__ == "__main__":
	# env = CliffWalkingEnv()
	env = gym.make("FrozenLake-v1")
	env = env.unwrapped
	env.render()

	action_meaning = ['^', 'v', '<', '>']
	theta = 0.0001
	gamma = 0.9
	# disaster = [i for i in range(37, 47)]
	# end = [47]
	disaster = [11, 12, 5, 7]
	end = [15]
	# agent = PolicyIteration(env, theta, gamma)
	# agent.policy_iteration()
	agent = ValueIteration(env, theta, gamma)
	agent.value_iteration()
	print_agent(agent, action_meaning, disaster, end)


