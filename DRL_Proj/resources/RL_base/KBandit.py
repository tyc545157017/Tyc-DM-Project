import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
	np.random.seed(1)
	K = 10
	bandit_10_arm = BernoulliBandit(K)
	print("随机生成一个%d臂伯努利老虎机"%K)
	print("获奖概率最大的拉杆为%d号，其获奖概率为%.4f"%
		  (bandit_10_arm.best_idx, bandit_10_arm.best_prob))