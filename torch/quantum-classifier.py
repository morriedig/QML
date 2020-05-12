import torch
from torch.autograd import Variable
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer, AdamOptimizer, GradientDescentOptimizer


"""
opt didn't work
"""
dev = qml.device("default.qubit", wires=4)

def layer(W):
  qml.Rot(W[0, 0], W[0, 1], W[0, 2], wires=0)
  qml.Rot(W[1, 0], W[1, 1], W[1, 2], wires=1)
  qml.Rot(W[2, 0], W[2, 1], W[2, 2], wires=2)
  qml.Rot(W[3, 0], W[3, 1], W[3, 2], wires=3)

  qml.CNOT(wires=[0, 1])
  qml.CNOT(wires=[1, 2])
  qml.CNOT(wires=[2, 3])
  qml.CNOT(wires=[3, 0])

def statepreparation(x):
    # qml.BasisState(x, wires=[0, 1, 2, 3])
    for i in range(len(x)):
      qml.RY(np.pi * x[i]/2, wires=i)
      qml.RX(np.pi * x[i]/2, wires=i)
      qml.RZ(np.pi * x[i]/2, wires=i)
      # print(x[i])

@qml.qnode(dev, interface='torch')

def circuit(weights, x=None):
    statepreparation(x)
    for W in weights:
      layer(W)
    return qml.expval(qml.PauliZ(0))

def variational_classifier(var, bias, x=None):
  # weights = var[0]
  # bias = var[1]
  output = circuit(var, x=x) + bias
  # print(circuit(var, x=x))
  # print("output")
  # print(output)
  # print("output")
  return output
  # return circuit(var, x=x)


def square_loss(labels, predictions):
  loss = 0
  for l, p in zip(labels, predictions):
    loss = loss + (l - p) ** 2
  # print(loss)
  loss = loss / len(labels)

  return loss

def accuracy(labels, predictions):
  loss = 0
  for l, p in zip(labels, predictions):
    if abs(l-p) < 1e-5:
      loss = loss + 1
  loss = loss / len(labels)

  return loss

def cost(Q_circuit, Q_bias, X, Y):
  predictions = [variational_classifier(Q_circuit, Q_bias, x=x).item() for x in X]
  return square_loss(Y, predictions)

def closure():
  opt.zero_grad()
  # loss = cost(Q_circuit = Q_circuit, Q_bias = Q_bias, features = X, labels = Y)
  loss = cost(Q_circuit=Q_circuit, Q_bias=Q_bias, X=X_batch, Y=Y_batch)
  loss.backward()
  print(loss)
  print("++++++")
  return loss

data = np.loadtxt("parity.txt")

X_sample = data[:, :-1]
Y_sample = data[:, -1]
Y_sample = Y_sample * 2 - np.ones(len(Y_sample))

X_sample = torch.from_numpy(X_sample)
Y_sample = torch.from_numpy(Y_sample)
X_sample.requires_grad = True
Y_sample.requires_grad = True

# for i in range(5):
#   print("X = {}, Y = {: d}".format(X[i], int(Y[i])))
# print("...")

num_qubits = 4
num_layers = 2

dtype = torch.DoubleTensor
init_circuit = Variable(torch.tensor(0.1 * np.random.randn(num_layers, num_qubits, 3), device='cpu').type(dtype), requires_grad=True) # random circuit
init_bias = Variable(torch.tensor(0.0, device='cpu').type(dtype), requires_grad=True) # random bias

Q_circuit = init_circuit
Q_bias = init_bias

# opt = torch.optim.Adam([Q_circuit, Q_bias], lr = 0.2)
# opt = torch.optim.SGD([Q_circuit, Q_bias], lr = 1e-2)

batch_size = 5
opt = torch.optim.RMSprop([Q_circuit, Q_bias], lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

for it in range(50):
  batch_index = np.random.randint(0, len(X_sample), (batch_size,))
  X_batch = X_sample[batch_index]
  Y_batch = Y_sample[batch_index]

  var = opt.step(closure)
  # Compute accuracy

  predictions = [np.sign(variational_classifier(Q_circuit, Q_bias, x=x).item()) for x in X_sample]
  print("===================")
  print(Q_circuit)
  print(Q_bias)
  print("===================")
  # predictions = [variational_classifier(Q_circuit, x=x).item() for x in X]
  acc = accuracy(Y_sample, predictions)

  # print("Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(it + 1, cost(Q_circuit, Q_bias, X_sample, Y_sample), acc))