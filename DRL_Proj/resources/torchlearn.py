import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# model = torch.hub.load('pytorch/vision:v0.4.2', 'deeplabv3_resnet101', pretrained=True)
# print(model.eval())

class LinearRegressionModel(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(LinearRegressionModel, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim)

	def forward(self, x):
		out = self.linear(x)
		return out

input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 1000
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

x_train = [i for i in range(50)]
x_train = np.array(x_train, dtype=np.float32).reshape(-1, 1)
y_train = [2 * i + 3 for i in x_train]
y_train	= np.array(y_train, dtype=np.float32).reshape(-1, 1)

print(x_train.shape)
print(y_train.shape)

# model.load_state_dict(torch.load('model.pkl'))

for epoch in range(epochs):
	epoch += 1
	#数据预处理，转为tensor格式
	inputs = torch.from_numpy(x_train).to(device)
	labels = torch.from_numpy(y_train).to(device)

	#梯度清零，防止梯度累加
	optimizer.zero_grad()

	#前向传播，计算模型结果
	outputs = model(inputs)

	#计算损失
	loss = criterion(outputs, labels)

	#反向传播
	loss.backward()

	#更新权重参数
	optimizer.step()

	if epoch % 50 == 0:
		print('Epoch:{}, Loss:{}'.format(epoch, loss.item()))

torch.save(model.state_dict(), 'model.pkl')

# print(torch.from_numpy(x_train).requires_grad_())
predict = model(torch.from_numpy(x_train).requires_grad_())
# print(predict.data.numpy())
print(predict)
plt.plot(x_train, y_train, label="actual")
plt.plot(x_train, predict.data.numpy(), label="predict")
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.show()
#
# x = torch.empty(5, 3)
# x = torch.rand(5, 3)
# x = torch.tensor([5.5, 3])
# z = x.new_ones(5, 3, dtype=torch.double)
# y = torch.rand_like(z,  dtype=float)
#
# x = torch.randn(4, 4)
# y = x.view(16)
# z = x.view(8, -1)
#
# x = torch.ones(5)
# y = x.numpy()
#
# x = np.ones(5)
# y = torch.from_numpy(x)
#
# x = torch.randn(3, 4, requires_grad=True)
# x = torch.randn(3, 4)
# x.requires_grad = True
# b = torch.randn(3, 4, requires_grad=True)
# t = x + b
# print(t.requires_grad)
# y = t.sum()
# y.backward()
# print(b.grad)
# print(x)
# print(y.requires_grad)
# print(z)
# print(x[:,-1])


# print(y+z)
# print(torch.add(y, z))