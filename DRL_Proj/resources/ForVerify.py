c = [[1, 2], [3, 4], [5, 6], [5, 7]]
# print(len(c))
import numpy as  np
import torch
import random
import matplotlib.pyplot as plt

# x = [i for i in range(20)]
# y = [1.*(i - 1)/(i + 1) for i in x]
# plt.plot(x, y)
# plt.show()

random.seed(1)
torch.manual_seed(0)
print(len(c))
c = np.array(c)
d = [1, 2, 3, 8, 5, 6, 7, 8]
# m1 = [2, 3]
# print(np.array([m1]))
# print(torch.tensor([m1]).shape)
# print(torch.tensor(np.array(m1)).shape)
d_zip = [(i, 2*i, 3*i - 1) for i in range(10)]
d_zip = torch.tensor(d_zip, dtype=torch.float)
s1 = torch.randn(2, 3)
print(s1)
print(torch.nn.functional.softmax(s1, dim=0))
# d_zip = torch.tensor(d_zip, dtype=torch.float)
a1 = np.zeros([2, 3])
a1[1, 2] = 6
a1[1][2] = 2
model = {}
model[(0, 1)] = (2, 3)
model[(0, 2)] = 1, 3
model[(2, 4)] = 3, 6
model[(0, 1)] = 3, 4
# print(torch.__version__)
# print(d_zip)
# print(d_zip.max(1)[1])
# print(d_zip.shape)
# transition = np.array(random.sample(d_zip, 4))
# print(np.array([transition]))
# print(np.array(transition).max(1)[0])
# a, b, c = zip(*transition)
# print(a, b, c)
# print(dict())

# e = d.pop(0)
# print(e)
# print(np.argmax(d))
#
# print(list(model.items()))

# print(eval(c)%3 - 1)

# import gym
# env = gym.make("FrozenLake-v1")  # 创建环境
# env = env.unwrapped  # 解封装才能访问状态转移矩阵P
# env.render()  # 环境渲染,通常是弹窗显示或打印出可视化的环境
#
# holes = set()
# ends = set()
# for s in env.P:
#     for a in env.P[s]:
#         for s_ in env.P[s][a]:
#             if s_[2] == 1.0:  # 获得奖励为1,代表是目标
#                 ends.add(s_[1])
#             if s_[3] == True:
#                 holes.add(s_[1])
# holes = holes - ends
# print("冰洞的索引:", holes)
# print("目标的索引:", ends)
#
# for a in env.P[14]:  # 查看目标左边一格的状态转移信息
#     print(env.P[14][a])