import copy
# import gym
import gymnasium as gym
from pprint import pprint
from gym import spaces
from gym import envs
import numpy as np
import tqdm
import torch
import time
import os
import sys
from datetime import datetime
import pandas as pd
from matplotlib import pyplot as plt
import rl_utils
import collections
import yaml
from MyParamManager import ParamManager
import pathlib

np.random.seed(0)
# np.set_printoptions(suppress=True)
# print(os.getcwd())
# home_dir = os.getcwd().rsplit('resources', 1)[0]
# print(home_dir)
# os.chdir(home_dir)
# print(os.getcwd())

version_config = {
	"tyc-highway-v0": {
		"PPO": 0,
		"SAC": 0,
		"DoubleDQN": 0,
		"DuelingDQN": 0
	},
	"highway-v0": {
		"CCC": 0,
		"SAC": 0,
		"DoubleDQN": 0,
		"DuelingDQN": 0
	}
}

lr = {
	'actor_lr': 5e-4,
	'critic_lr': 5e-3
}

drl_param = {
	'gamma': 0.9,
	'lmbda': 0.95,
	'epochs': 10,
	'eps': 0.2
}


class Rect(object):
    def __init__(self, area) -> None:
        self.__area = area
    
    @property
    def area(self):
        return self.__area

    @area.setter
    def area(self, value):
        self.__area = value
    

class Mytest(object):
    globalValue = 0
    def __init__(self, valuetest) -> None:
        self.valuetest = valuetest
    def functest(self):
        print("father")
class MytestSub(Mytest):
    globalValue = 2
    def subTest(self):
        print(self.valuetest)
    def changeValue(self, value):
        self.globalValue = value
    def functest(self):
        print("son")
        
def check_envs():
    print("Gym version: {}".format(gym.__version__))
    env_list = envs.registry.keys()
    env_ids = [env_item for env_item in env_list]
    print("There are {} envs in gym".format(len(env_ids)))
    print(env_ids)

def SpaceType():
    # Box类型, 创建连续空间, spaces.Box(low=, high=, shape=)
    action_space1 = spaces.Box(-10, 10, (2, 2))


    low = np.float32(np.zeros(3))
    high = np.float32(np.ones(3))
    action_space2 = spaces.Box(low=low, high=high)

    print(action_space2, action_space1)
    print(action_space1.sample())
    print(action_space2.sample())

    # Discrete类型，创建离散空间
    action_space3 = spaces.Discrete(5)
    print(action_space3.sample())

    # Dict类型，创建具有离散动作和连续动作的采用空间
    action_space = spaces.Dict({
        "action1": spaces.Discrete(10),
        "action2": spaces.Box(low=low, high=high)
    })
    print(action_space.sample())

    # MultiBinary, n-shape的binary space
    binary_space = spaces.MultiBinary(5) # ([3, 2])
    print(binary_space.sample())

    # MultiDiscrete, 一系列离散的action sapce
    multi_space = spaces.MultiDiscrete([3, 2]) # 两个discrete space
    print(multi_space.sample())

def runEnv():
    env_name = 'CartPole-v1'    # Pendulum-v1
    env = gym.make(env_name, render_mode='human')
    state = env.reset()
    print('initial state: {}'.format(state))
    step_num = 0
    while True:
        step_num += 1
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        print('state = {}; reward = {}'.format(state, reward))
        print('terminated = {}; truncated = {}'.format(terminated, truncated))
        if terminated:
            print('terminated done {}'.format(step_num))
            break
        if truncated:
            print('truncated done {}'.format(step_num))
            break
        # time.sleep(1)
    env.close()

def singalStar(common, *rest):
    print("Common args: ", common)
    print("Rest args: ", rest)
def doubleStar(common, **double):
    print("Common args: ", common)
    print("Rest args: ", double)

def torch_test():
    A = [[1, 2, 3, 4, 5]]
    V = [2]
    A = torch.tensor(A, dtype=torch.float)
    V = torch.tensor(V, dtype=torch.float)
    print(A)
    print(V)
    Q = A + V
    diff = A.mean(1)
    print(Q)
    print(diff)
    print(torch.mean(A, dim=-1))

def path_test():
    path = 'modle2\\sub'
    path = os.path.join(path, 'pathtest.txt')
    print(path)
    print(os.path.exists(path))
    if not os.path.exists(path):
        os.mkdir(path.rsplit('\\', 1)[0])
    with open(path, 'w') as file:
        file.write('123')

def save_train_log(log_name, **datas):
    try:
        if not os.path.exists(log_name):
            df = pd.DataFrame(columns=["reward", "episode_len"])
            df.to_csv(log_name, index=False)
        data_list = [datas["reward"], datas["episode_len"]]
        data_list = pd.DataFrame(data_list).T
        data_list.to_csv(log_name, mode='a', header=False, index=False)
    except Exception as e:
        print(type(e))
        raise KeyError("Can't find args")

def read_train_log(log_name):
    try:
        frame_data = pd.read_csv(log_name)
        train_data = np.array([frame_data["reward"], frame_data["episode_len"]])
        # train_data = np.array([frame_data.index[0], frame_data.index[1]])
        print(train_data.shape)
        return train_data
    except Exception as e:
        print(e)
        raise KeyError("Something error")

def plot_image():
    double_train_data = read_train_log(double_log_name)  
    double_avrg_reward = double_train_data[0] / double_train_data[1]
    # double_mv_return = rl_utils.moving_average(double_avrg_reward, 9)
    double_mv_return = rl_utils.moving_average( double_avrg_reward, 9)
    # double_mv_return = rl_utils.moving_average( double_mv_return, 9)
    # double_mv_return = rl_utils.moving_average( double_mv_return, 9)

    dueling_train_data = read_train_log(dueling_log_name)
    dueling_avrg_reward = dueling_train_data[0] / dueling_train_data[1]
    # dueling_mv_return = rl_utils.moving_average(dueling_avrg_reward, 9)
    dueling_mv_return = rl_utils.moving_average(dueling_avrg_reward, 9)
    # dueling_mv_return = rl_utils.moving_average(dueling_mv_return, 9)
    # dueling_mv_return = rl_utils.moving_average(dueling_mv_return, 9)

    print(double_train_data)
    episode_idx = [i for i in range(len(double_avrg_reward))]

    print(f"doube dqn :{double_train_data[1].sum()*0.21/3600} hours")
    print(f"PPO :{dueling_train_data[1].sum()*0.21/3600} hours")

    plt.plot(episode_idx,  double_mv_return, label='PPO')
    plt.plot(episode_idx, dueling_mv_return, label='SAC')
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Avrg_reward')
    plt.title('AC')
    plt.show()

def cache(func):
    cached_result = {}
    def wrapper(*args):
        if args in cached_result:
            print(f"Cache hit for {func.__name__}({args})")
            return cached_result[args]
        result = func(*args)
        cached_result[args] = result
        print(f"Cache miss for {func.__name__}({args}), result cached")
        return result
    return wrapper

@cache
def fibonacci(n):
    if n < 2:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

def fibonacci2(n):
    if n < 2:
        return n
    else:
        return fibonacci2(n - 1) + fibonacci2(n - 2)


def queueTest():
    d = collections.deque(maxlen=10)
    for i in range(30):
        # data = (i, i+1, i+3, i+7)
        d.append(i)
        time.sleep(0.5)
        print(d)
        # print(len(d))
        if d.count(3) >= 1:
            print('3 in d')

def readconfig(path):
    with open(path, 'r') as file:
        config  = yaml.load(file, Loader=yaml.FullLoader)
    pprint(config)


if __name__ == "__main__":
    torch.manual_seed(0)
    # check_envs()
    # SpaceType()
    # print(gym.__version__)
    # print(np.__version__)
    # print(tqdm.__version__)
    # runEnv()
    # doubleStar("hello", x="world", y=0)
    # doubleStar("hello", **{"x": "world", "y": 3})
    # c = np.random.randint(0, 10, (2, 3, 5))
    # a = [1, 2, 3]
    # print(a[-10:])
    print("next".center(60, '*'))
    # for i in range(2):
    #     print(c[i, ...].T)
    # torch_test()
    # dict1 = {'0':2, '1':3}
    # doubleStar("test", **dict1)
    # print(datetime.now())
    double_log_name = 'logs/PPO_model_for_tyc-highway-v0_v1_log.csv'
    dueling_log_name = 'logs/SAC_model_for_tyc-highway-v0_v3_log.csv'
    log_test_name = 'logs/DQN_train_log_v4_modify.csv'
    model_path = 'highway_dqn_model\DuelingDQN_model_for_highway-v0_v2.pkl'
    final_model_path = model_path.rsplit('.', 1)[0] + '_final.pkl'
    config_path = 'yaml_config/customized_config.yaml'
    print(final_model_path)
    # doubleStar("hello", x="1", y="2")
    # plot_image()
    # a = np.array(3)
    # print(int(a))
    # a = [2, 4]
    # print(a/2)
    # queueTest()
    # readconfig(config_path)
    """ pandas
    if os.path.exists(log_name):
        raise ValueError("Path exists! Maybe override!")
    data1 = [i for i in range(10)]
    data2 = [i+10 for i in range(10)]
    datas = {"reward": data1, "episode_len": data2}
    save_train_log(log_name, **{"reward": data1, "episode_len": data2})
    while input('please enter y continue:') is not 'y':
        print('nop')
    plot_image()
    df = pd.read_csv(dueling_log_name)
    pprint(df)
    df2 = df.drop(0)
    print(df.iloc[0])
    print(df['reward'])
    df.to_csv(log_test_name, index=False, encoding='utf-8')
    double_train_data = read_train_log(double_log_name) 
    dueling_log_name = read_train_log(dueling_log_name)
    print(dueling_log_name[1].sum()*0.3/3600)
    """

    """ @property
    rect1 = Rect(20)
    rect1.area = 30.0
    print(rect1.area)
    """

    """ Decorator test
    T1 = time.perf_counter()
    f1 = fibonacci(40)
    T2 = time.perf_counter()
    print(f"with decorator: {T2 - T1} s")

    T1 = time.perf_counter()
    f1 = fibonacci2(40)
    T2 = time.perf_counter()
    print(f"without decorator: {T2 - T1} s")
    """

    # test1 = Mytest(2)
    # test2 = MytestSub(3)
    # test2.functest()
    print(float('inf'))
    # data1 = torch.tensor([[0.0351, 0.8088, 0.0112, 0.1004, 0.0445]])
    # PM1 = ParamManager("tyc-highway-v0", "PPO", drl_param, lr, 3000)
    # PM1.save_config_file()
    '''
    for _ in range(20):
        print("split".center(60, '*'))
        data1_org = torch.randn(1, 5)
        print(data1_org)
        data1 = torch.softmax(data1_org, dim=1)
        print(data1)
        dist1 = torch.distributions.Categorical(data1)
        max_index = data1.argmax().item()
        sam = dist1.sample()
        if max_index == sam.item():
            print("true")
        else:
            print(f"exploring: max-{max_index} select-{sam.item()}")
    '''

    print(pd.__version__)