import torch
from torch.autograd import Variable
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer, GradientDescentOptimizer

import matplotlib.pyplot as plt


"""
產生數據，並繪圖
"""
def datas_create(samples, center=[0.0, 0.0], radius=np.sqrt(2 / np.pi)): # 產生數據
    Xvals, yvals = [], []

    for i in range(samples):
        x = 2 * (np.random.rand(2)) - 1
        y = 0
        if np.linalg.norm(x - center) < radius:
            y = 1
        Xvals.append(x)
        yvals.append(y)
    return np.array(Xvals), np.array(yvals)

def plot_data(x, y, fig=None, ax=None):
    if fig == None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    reds = y == 0
    blues = y == 1
    ax.scatter(x[reds, 0], x[reds, 1], c="red", s=20, edgecolor="k")
    ax.scatter(x[blues, 0], x[blues, 1], c="blue", s=20, edgecolor="k")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")

Xdata, ydata = datas_create(500)
# fig, ax = plt.subplots(1, 1, figsize=(4, 4))
# plot_data(Xdata, ydata, fig=fig, ax=ax)
# plt.show()  # 畫出原始數據圖



"""
"""

dev = qml.device('default.qubit', wires=1)

# def statepreparation(a):

# 	for i in range(len(a)):
# 		qml.RX(np.pi * a[i], wires=i)
# 		qml.RY(np.pi * a[i], wires=i)

# def layer(W):
# 	qml.CNOT(wires=[0, 1])

# 	qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
# 	qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)


@qml.qnode(dev, interface='torch')
def circuit(weights, x=None, y=None):
	# statepreparation(angles)
	# print(weights)
	# print("==================")
	# print(x)
	# print('==================')
	# print(y)
	for W in weights:
		# qml.Rot(*x, wires=0)
		# qml.RX(x[0], wires=0)
		# qml.RY(x[1], wires=0)
		# qml.RX(x[2], wires=0)
		# qml.Rot(x[0], x[1] ,x[2], wires=0)
		# qml.Rot(W[0], W[1], W[2], wires=0)
		qml.Rot(*W, wires=0)
		qml.Rot(*x, wires=0)
		qml.Rot(*W, wires=0)
		qml.Rot(*x, wires=0)
		qml.Rot(*W, wires=0)
		qml.Rot(*x, wires=0)
		qml.Rot(*W, wires=0)
		if y == 0:
				y = np.array([[1, 0],[0, 0]])
		else:
				y = np.array([[0, 0],[0, 1]])
		# qml.RY(W[0, 1], wires=0)
		# print(y)

	return qml.expval(qml.Hermitian(y, wires=[0]))
	# return qml.expval(qml.PauliZ(0))

# def variational_classifier(Q_circuit, Q_bias , angles=None):

# 	output = circuit(Q_circuit, angles=angles) + Q_bias

# 	return output

# def square_loss(labels, predictions):
# 	loss = 0
# 	for l, p in zip(labels, predictions):
# 		# print(l)
# 		# print("=====================")
# 		# print(l - p[0] < 1e-5)
# 		# if (l - p[0]) **2 < 1e-6:
# 			loss = loss + (l - p[0]) ** 2
# 			# loss = loss + 1
# 	loss = loss / len(labels)
# 	print(loss)

# 	# print(torch.autograd.Variable(torch.from_numpy(np.array([loss]))))
# 	# return torch.autograd.Variable(torch.from_numpy(np.array([loss])), requires_grad=True)
# 	return loss

# def accuracy(labels, predictions):
# 	loss = 0
# 	for (l, p) in zip(labels, predictions):
# 		if (l - p[0]) **2 < 1e-6:
# 			loss = loss + 1
# 	loss = loss / len(labels)

# 	return loss

def cost(weights, x, y):
  loss = 0.0
  for i in range(len(x)):
    f = circuit(weights, x=x[i], y=y[i])
    # print(f)
    # print("----")
    loss = loss + (1 - f) ** 2
  # predictions = [variational_classifier(Q_circuit = Q_circuit, Q_bias = Q_bias, angles=item) for item in features]
  # print(accuracy(labels, predictions))
  # return square_loss(labels, predictions)
  return loss / len(x)

def test(weights, x, y):
  fidelity_values = []
  predicted = []
  y = [0 ,1]
  for i in range(len(x)):
    fidel_function = lambda y: circuit(weights, x=x[i], y=y)
    fidelities = [fidel_function(dm).item() for dm in y]
    # fidelities = [circuit(weights, x[i], label).item() for label in y]
    # print(fidelities)
    # print("++++++++++++++++++++++++++++++++")
    best_fidel = np.argmax(fidelities)
    predicted.append(best_fidel)
    fidelity_values.append(fidelities)

  # print(predicted)
  # print(fidelity_values)
  # print(predicted)
  return np.array(predicted), np.array(fidelity_values)
  # return predicted, fidelity_values

def accuracy_score(y_true, y_pred):
  score = y_true == y_pred
  return score.sum() / len(y_true)

def iterate_minbatches(inputs, targets, batch_size):
  for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
    idxs = slice(start_idx, start_idx + batch_size)
    yield inputs[idxs], targets[idxs]

def closure():
  opt.zero_grad()
  loss = cost(weights = Q_circuit[0], x = Xbatch, y = ybatch)
  loss.backward()
  # print(loss)
  # print("======")
  return loss

X = Xdata
Y = ydata
X = np.hstack((X, np.zeros((X.shape[0], 1))))
# Y = Y * 2 - np.ones(len(Y))

X_sample = torch.from_numpy(X)
y_sample = torch.from_numpy(Y)

dtype = torch.DoubleTensor

num_qubits = 1 #  wires
num_layers = 1 # layer

init_circuit = Variable(torch.tensor(0.1 * np.random.randn(num_layers, num_qubits, 3), device='cpu').type(dtype), requires_grad=True) # random circuit
init_bias = Variable(torch.tensor([0.0, 0.0, 0.0, 0.0], device='cpu').type(dtype), requires_grad=True) # random bias

Q_circuit = init_circuit
Q_bias = init_bias
batch_size = 150

# target_Q_circuit = Q_circuit.clone().detach()
# target_Q_bias = Q_bias.clone().detach()

opt = torch.optim.Adam([Q_circuit, Q_bias], lr = 0.6)

for i in range(20):
  for Xbatch, ybatch in iterate_minbatches(X_sample, y_sample, batch_size=batch_size):
    print(opt.step(closure))
  # batch_index = np.random.randint(0, len(X), (batch_size,))
  # for x in batch_index:
  #   X_train = X_sample[batch_index]
  #   Y_train = y_sample[batch_index]
    # p1n, p2n = opt.param_groups[0]["params"]
    # Q_circuit = p1n
    # opt.step(closure)
    # print(opt.step(closure))
  
  # opt.step(closure)


  # costn = cost(init_circuit, init_bias, X, Y)
  # print(opt.step(closure))
  predicted_train, fidel_train = test(Q_circuit, X_sample, y_sample)
  # print(len(Y[batch_index]))
  accuracy_train = accuracy_score(predicted_train, ydata)
  print(accuracy_train)
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
plot_data(Xdata, predicted_train, fig=fig, ax=ax)
plt.show()  # 畫出原始數據圖