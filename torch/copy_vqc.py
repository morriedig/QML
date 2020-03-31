import torch
from torch.autograd import Variable
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer, GradientDescentOptimizer


dev = qml.device('default.qubit', wires=4)
# dev = qml.device('qiskit.basicaer', wires=4)
def statepreparation(a):

	"""Quantum circuit to encode a the input vector into variational params

	Args:
		a: feature vector of rad and rad_square => np.array([rad_X_0, rad_X_1, rad_square_X_0, rad_square_X_1])
	"""
	
	# Rot to computational basis encoding
	# a = [a_0, a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8]

	for ind in range(len(a)):
		qml.RX(np.pi * a[ind], wires=ind)
		qml.RZ(np.pi * a[ind], wires=ind)

def layer(W):
	""" Single layer of the variational classifier.

	Args:
		W (array[float]): 2-d array of variables for one layer
	"""

	qml.CNOT(wires=[0, 1])
	qml.CNOT(wires=[1, 2])
	qml.CNOT(wires=[2, 3])


	qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
	qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
	qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
	qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)


@qml.qnode(dev, interface='torch')
def circuit(weights, angles=None):
	"""The circuit of the variational classifier."""
	# Can consider different expectation value
	# PauliX , PauliY , PauliZ , Identity  

	statepreparation(angles)
	
	for W in weights:
		layer(W)

	# return [qml.expval(qml.PauliZ(ind)) for ind in range(4)]
	return qml.expval(qml.PauliZ(1))

def variational_classifier(var_Q_circuit, var_Q_bias , angles=None):
	"""The variational classifier."""

	weights = var_Q_circuit

	raw_output = circuit(weights, angles=angles) + var_Q_bias

	return raw_output

def square_loss(labels, predictions):
	loss = 0
	for l, p in zip(labels, predictions):
		loss = loss + (l - p) ** 2
	loss = loss / len(labels)
	# return loss[1]
	return loss[1]

def cost(var_Q_circuit, var_Q_bias, features, labels):
	predictions = [variational_classifier(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, angles=item) for item in features]

	return square_loss(labels, predictions)

def closure():
  opt.zero_grad()
  loss = cost(var_Q_circuit = var_Q_circuit, var_Q_bias = var_Q_bias, features = batch_sampled, labels = Q_target)
  loss.backward()
  return loss

data = np.loadtxt("parity.txt")
X = data[:, :-1]
Y = data[:, -1]
Y = Y * 2 - np.ones(len(Y))

batch_sampled = torch.from_numpy(X)
Q_target = Y = torch.from_numpy(Y)

dtype = torch.DoubleTensor

num_qubits = 4 # 4 wires
num_layers = 2 # 兩層

var_init_circuit = Variable(torch.tensor(0.01 * np.random.randn(num_layers, num_qubits, 3), device='cpu').type(dtype), requires_grad=True)
var_init_bias = Variable(torch.tensor([0.0, 0.0, 0.0, 0.0], device='cpu').type(dtype), requires_grad=True)

var_Q_circuit = var_init_circuit
var_Q_bias = var_init_bias

var_target_Q_circuit = var_Q_circuit.clone().detach()
var_target_Q_bias = var_Q_bias.clone().detach()

opt = torch.optim.RMSprop([var_Q_circuit, var_Q_bias], lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
# opt = AdamOptimizer(0.01)

for i in range(50):
  opt.step(closure)
  p1n, p2n = opt.param_groups[0]["params"]
  costn = cost(var_init_circuit, var_init_bias, X, Y)
  print(costn.item())