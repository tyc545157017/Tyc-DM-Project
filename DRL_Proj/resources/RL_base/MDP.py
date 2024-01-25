##Markov Reward Process 马尔可夫奖励过程
import numpy as np
np.random.seed(0)
rewards = [-1, -2, -2, 10, 1, 0]

def compute_return(start_idx, chain, gamma):
	G = 0
	for i in reversed(range(start_idx, len(chain))):
		G = gamma*G + rewards[chain[i]-1]
	return G


def analytical_compute(P, rewards, gamma, states_num):
	rewards = np.array(rewards).reshape((-1, 1))
	value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
	return value


def join(str1, str2):
	return str1 + '-' + str2

def sample(MDP, Pi, max_timestep, number):
	S, A, P, R, gamma = MDP
	episodes = []
	for _ in range(number):
		episode = []
		s = S[np.random.randint(4)]
		timestep = 0
		while s != 's5' and timestep <= max_timestep:
			timestep += 1
			rand, temp = np.random.rand(), 0
			for a_opt in A:
				temp += Pi.get(join(s, a_opt), 0)
				if temp > rand:
					a = a_opt
					r = R.get(join(s, a), 0)
					break
			rand, temp = np.random.rand(), 0
			for s_next_opt in S:
				temp += P.get(join(join(s, a), s_next_opt), 0)
				if temp > rand:
					s_next = s_next_opt
					break
			episode.append((s, a, r, s_next))
			s = s_next
		episodes.append(episode)
	return episodes

def MC(episodes, V, N, gamma):
	for episode in episodes:
		G = 0
		for i in range(len(episode) - 1, -1, -1):
			(s, a, r, s_next) = episode[i]
			G = r + gamma * G
			N[s] = N[s] + 1
			V[s] = V[s] + (G - V[s]) / N[s]

def occupancy(episodes, s, a, max_timestep, gamma):
	rho = 0
	total_time = np.zeros(max_timestep)
	occur_time = np.zeros(max_timestep)
	for episode in episodes:
		for i in range(len(episode)):
			(s_opt, a_opt, r_opt, s_next_opt) = episode[i]
			total_time[i] += 1
			if s == s_opt and a == a_opt:
				occur_time[i] += 1

	for i in reversed(range(max_timestep)):
		if total_time[i]:
			rho += gamma**i * occur_time[i]/total_time[i]
	return (1 - gamma) * rho


# P = [
#     [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
#     [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
#     [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
#     [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
# ]
# P = np.array(P)
#
# rewards = [-1, -2, -2, 10, 1, 0]
# gamma = 0.5

# chain = [1, 2, 3, 6]
# start_idx = 0
# G = compute_return(start_idx, chain, gamma)
# value = analytical_compute(P, rewards, gamma, len(rewards))
# print("累计收益为:", G)
# print("解析计算价值：\n", value)

#Markov Decision Process 马尔可夫决策过程
S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
# 状态转移函数
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}
# 奖励函数
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}
gamma = 0.5  # 折扣因子
MDP = (S, A, P, R, gamma)

# 策略1,随机策略
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}
# 策略2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}


# gamma = 0.5
# # 转化后的MRP的状态转移矩阵
# P_from_mdp_to_mrp = [
#     [0.5, 0.5, 0.0, 0.0, 0.0],
#     [0.5, 0.0, 0.5, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.5, 0.5],
#     [0.0, 0.1, 0.2, 0.2, 0.5],
#     [0.0, 0.0, 0.0, 0.0, 1.0],
# ]
# P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
#
# R_from_mdp_to_mrp = [-0.5, -1.5, -1, 5.5, 0]
#
# V_states = analytical_compute(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, len(R_from_mdp_to_mrp))
# print("解析计算状态价值：\n", V_states)
max_timestep = 1000
sample_number = 1000
# episodes = sample(MDP, Pi_1, 20, 5)
# episodes = sample(MDP, Pi_1, 20, 1000)
# V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
# N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
# MC(episodes, V, N, gamma)
# print("蒙特卡洛采样计算的状态价值：\n", V)

episodes_1 = sample(MDP, Pi_1, max_timestep, sample_number)
episodes_2 = sample(MDP, Pi_2, max_timestep, sample_number)
rho_1 = occupancy(episodes_1, "s4", "概率前往", max_timestep, gamma)
rho_2 = occupancy(episodes_2, "s4", "概率前往", max_timestep, gamma)
print("策略一的（s4， 概率前往）占用度量为：", rho_1)
print("策略二的（s4， 概率前往）占用度量为：", rho_2)
# for i in range(5):
# 	print('第{}条采样序列:\n'.format(i+1), episodes[i])


