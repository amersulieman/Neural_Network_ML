import math
import numpy as np
import os.path as path
import random as rd

# config values
L = [2, 2, 2, 2]  # layers
alpha = 0.2
target_mse = 0.2
max_epoch = 1000
min_error = math.inf
min_error_epoch = -1
epoch = 0  # one epoch is one forward and backward sweep
mse = math.inf
error = []
epo = []

# load data
x_file = path.abspath("./X.txt")
y_file = path.abspath("./Y.txt")
X = np.loadtxt(x_file)
Y = np.loadtxt(y_file)
n_samples, n_features = np.shape(X)
n_target_output, n_of_classes = np.shape(X)
Y = Y.T

# build matrices for B and T and rest
def build_starting_betas():
    layers = len(L) - 1
    dataSetBetas = []
    for layer in range(layers):
        betaSet = [[rd.uniform(-0.7, 0.7), rd.uniform(-0.7, 0.7)] for x in range(3)]
        dataSetBetas.append(betaSet)
    return np.array(dataSetBetas)


def build_initial_Zs(input_data):
    layers = len(L)
    dataSet_Zs = []
    # for all layers except last one
    for layer in range(layers - 1):
        dataSet_Zs.append([np.ones((3, 1))])
    # last layer does not have bias so its vector is one smaller
    dataSet_Zs.append(np.ones((2, 1)))
    # First layer has input data
    dataSet_Zs[0] = np.concatenate((input_data, np.ones((n_samples, 1))), 1).T
    return np.array(dataSet_Zs)


def build_initial_term_errors():
    layers = len(L)
    dataSet_term_erros = []
    for layer in range(layers):
        dataSet_term_erros.append(np.zeros((2, 1)))
    return np.array(dataSet_term_erros)


def forward_propagate(B, Z):
    layers = len(L)
    dataSet_Ts = []
    # first layer which is input
    dataSet_Ts.append(np.ones((2, 1)))

    for i in range(layers - 1):
        T_value = B[i].T @ Z[i]
        dataSet_Ts.append(T_value)
        if (i + 1) < len(L):
            xx = np.concatenate((1 / (1 + np.exp(dataSet_Ts[i + 1] * -1))), 1)
            Z[i + 1] = np.concatenate(
                (1 / (1 + np.exp(dataSet_Ts[i + 1] * -1))), np.ones(n_samples, 1)
            )
        else:
            Z[i + 1] = 1 / (1 + math.exp(-dataSet_Ts[i + 1]))
    return np.array(dataSet_Ts)


Betas = build_starting_betas()
print(Betas.shape)
print()
print(Betas[0].T.shape)
Z = build_initial_Zs(X)
print(Z[0].shape)
delta = build_initial_term_errors()
forward_propagate(Betas, Z)
