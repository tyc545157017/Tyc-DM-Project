import os
import yaml
from pprint import pprint

# lr = {
# 	'actor_lr': 5e-4,
# 	'critic_lr': 5e-3
# }
#
# drl_param = {
# 	'gamma': 0.9,
# 	'lmbda': 0.95,
# 	'epochs': 10,
# 	'eps': 0.2
# }

class ParamManager:
	def __init__(self, env_name, drl_type, drl_param, lr, train_episodes, train_version=-1, train=True):
		self.env_name = env_name
		self.drl_type = drl_type
		self.drl_param = drl_param
		self.lr = lr
		self.train_episodes = train_episodes
		self.train_version = train_version
		self.train = train
		self.version_path = os.path.join('paramfile', 'version.yaml')	#f'paramfile/version.yaml'
		self.version_config = {
			"tyc-highway-v0":{
				"PPO": 0,
				"SAC": 0,
				"DoubleDQN": 0,
				"DuelingDQN": 0
			},
			"highway-v0":{
				"PPO": 0,
				"SAC": 0,
				"DoubleDQN": 0,
				"DuelingDQN": 0
			}
		}
		self.change_work_dir()
		self.read_version_file(train_version)
		self._dirs = self.make_dirs()	#0: model, 1: config, 2:log
		self.config = dict()
		self.read_env_config()

	def make_dirs(self):
		over_dir = os.path.join('model', self.env_name, self.drl_type, f'v{self.train_version}')
		model_dir = os.path.join(over_dir, 'model')	#f'model/{self.env_name}/{self.drl_type}/v{self.train_version}/model'
		config_dir = os.path.join(over_dir, 'config')	#f'model/{self.env_name}/{self.drl_type}/v{self.train_version}/config'
		logs_dir = os.path.join(over_dir, 'logs')	#f'model/{self.env_name}/{self.drl_type}/v{self.train_version}/logs'
		_dirs = (model_dir, config_dir, logs_dir)
		if self.train:
			for _dir in _dirs:
				if not os.path.exists(_dir):
					os.makedirs(_dir)
		return _dirs

	def read_version_file(self, train_version):
		if not os.path.exists(self.version_path):
			try:
				os.mkdir(os.path.dirname(self.version_path))
				with open(self.version_path, 'w') as file:
					yaml.dump(self.version_config, file)
			except FileExistsError:
				print("floder existed")
		else:
			with open(self.version_path, 'r') as file:
				self.version_config = yaml.load(file, Loader=yaml.Loader)
			pprint(self.version_config)
			if self.train_version == -1:
				if self.train:
					self.train_version = self.version_config[self.env_name][self.drl_type]
				else:
					self.train_version = self.version_config[self.env_name][self.drl_type] - 1
					print(f"read last train model v{self.train_version} to test".center(100, '-'))
			# if self.train:
			# 	self.train_version = self.version_config[self.env_name][self.drl_type]
			# elif self.train_version == -1:
			# 	self.train_version = self.version_config[self.env_name][self.drl_type] - 1
			# 	print(f"read last train model v{self.train_version} to test".center(100, '-'))
			# else:
			# 	self.train_version = train_version
			# 	print(f"use customized version {self.train_version} to test".center(100, '-'))

	def update_version_files(self, **kwargs):
		if len(kwargs) != 0:
			for key, value in kwargs.items():
				self.version_config[key].update(value)
		self.version_config[self.env_name][self.drl_type] += 1
		pprint(self.version_config)
		with open(self.version_path, 'w') as file:
			yaml.dump(self.version_config, file)

	def read_env_config(self):
		fn = os.path.join('config', self.env_name + '.yaml')
		if not os.path.exists(fn):
			raise FileNotFoundError(f"Can not found {self.env_name} config file under '/config' floder, please setting")
		with open(fn, 'r') as f:
			self.config = yaml.load(f, Loader=yaml.Loader)

	def save_config_file(self):
		filename = os.path.join(self.config_dir, 'hyperparam.yaml')
		config_dict = dict()
		config_dict['learning_rate'] = self.lr
		config_dict['drl_param'] = self.drl_param
		config_dict['train_episodes'] = self.train_episodes
		config_dict['env_config'] = self.config
		with open(filename, 'w') as file:
			yaml.dump(config_dict, file)

	def change_work_dir(self):
		work_dir = __file__.rsplit('resources', 1)[0]
		os.chdir(work_dir)
		print(f'now work dir: {os.getcwd()}')

	@property
	def model_dir(self):
		return self._dirs[0]

	@property
	def config_dir(self):
		return self._dirs[1]

	@property
	def logs_dir(self):
		return self._dirs[2]




class PPOParam(ParamManager):
	def __init__(self, env_name,):
		super(PPOParam, self).__init__()