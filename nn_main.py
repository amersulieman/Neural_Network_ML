import math
import numpy as np
import os.path as path
import random as rd

# config values
L = [2, 2, 2, 2]  # layers
alpha = 0.2
target_mse = 0.0001
max_epoch = 2000
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
n_target_output, n_of_classes = np.shape(Y)
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
    dataSet_term_erros.append(np.zeros((2, 46)))
    for layer in range(layers - 1):
        dataSet_term_erros.append(np.zeros((2, 46)))
    return np.array(dataSet_term_erros)


def activation_function(T):
    sig_of_T = 1 / (1 + np.exp(-T))
    return sig_of_T


def compute_CSqErr(real_output, predicted_output):
    CSqErr = 0
    squared_err = (real_output - predicted_output) ** 2
    # sum by column to get one row for both errors summed per prediction
    sum_of_columns = squared_err.sum(axis=0)
    CSqErr += sum_of_columns.sum()
    return CSqErr


def forward_propagate(B, Z):
    layers = len(L)
    dataSet_Ts = []
    # first layer which is input
    dataSet_Ts.append(np.ones((2, 46)))

    for i in range(layers - 1):
        # matrix multiply
        T_value = B[i].T @ Z[i]
        # add T to the Ts set
        dataSet_Ts.append(T_value)
        layer_bias = np.ones((n_samples, 1)).T
        if (i + 1) < layers - 1:
            f_sig = activation_function(dataSet_Ts[i + 1])
            Z[i + 1] = np.concatenate((f_sig, layer_bias))
        else:
            Z[i + 1] = activation_function(dataSet_Ts[i + 1])
    return np.array(dataSet_Ts)


def compute_delta_term_for_nodes(deltas):
    layers = len(L)
    """
        Compute error delta term for all nodes except input
        1) compute error for output nodes
        2) compute error for hidden layers
        Note: Delta for out put node using sigmoid function:
                Delta = (Z(output) - Y) times Z(output) times(1-Z(output))

    """
    # Compute delta for output node
    deltas[-1] = (Z[-1] - Y) * Z[-1] * (1 - Z[-1])

    # compute delta error term for rest of layers except input
    # subtract layers by 2 to exclude last layer
    for layer in range(layers - 2, 0, -1):
        # Z at that layer without its bias
        z_without_bias = Z[layer][:-1]
        weight = z_without_bias * (1 - z_without_bias)
        D = deltas[layer + 1].T
        for m in range(n_samples):
            # error summation for layer delta(L)
            summation = (D[m, :] * betas[layer][:-1]).sum(axis=1)
            deltas[layer][:, m] = weight[:, m] * summation


def update_weights(betas):
    layers = len(L)
    for i in range(layers - 1):
        weight = Z[i][:-1]
        v1 = np.zeros((L[i], L[i + 1]))
        v2 = np.zeros((1, L[i + 1]))
        D = deltas[i + 1].T
        for m in range(n_samples):
            v1 += weight[:, m] @ D[m, :]
            v2 += D[m, :]

        # UPDATE BETAS WITHOUT BIAS
        betas[i][:-1] -= (alpha / n_samples) * v1
        # UPDATE BIAS
        betas[i][-1, :] -= ((alpha / n_samples) * v2)[0]


betas = build_starting_betas()
Z = build_initial_Zs(X)
deltas = build_initial_term_errors()
while (mse > target_mse) and (epoch < max_epoch):
    print("mse =", mse)
    print("epoch = ", epoch)
    # This updates Z values and returns T values
    T_values = forward_propagate(betas, Z)
    CSqErr = compute_CSqErr(Y, Z[-1])
    # normalize err
    CSqErr /= L[-1]
    compute_delta_term_for_nodes(deltas)
    update_weights(betas)
    CSqErr /= n_samples
    mse = CSqErr
    epoch += 1
    error.append(mse)
    epo.append(epoch)

    if mse < min_error:
        min_error = mse
        min_error_epoch = epoch

print(min_error)
print(min_error_epoch)
