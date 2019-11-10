import math
import numpy as np
import os.path as path
import random as rd

# config values
L = [2, 2, 2, 2]  # layers
alpha = 0.2
target_mse = 0.0001
max_epoch = 10000
min_error = math.inf
min_error_epoch = -1
epoch = 0  # one epoch is one forward and backward sweep
mse = math.inf
error = []
epo = []
xxx = 0
# load data
x_file = path.abspath("./X.txt")
y_file = path.abspath("./Y.txt")
X = np.loadtxt(x_file)
Y = np.loadtxt(y_file)
n_samples, n_features = X.shape
n_target_output, n_of_classes = Y.shape
Y = Y.T

# build matrices for B and T and rest
def build_starting_betas():
    layers = len(L) - 1
    dataSetBetas = np.empty((layers,), dtype=object)
    for layer in range(layers):
        dataSetBetas[layer] = np.random.uniform(low=-0.7, high=0.8, size=(3, 2))
    return dataSetBetas


def build_initial_Zs(input_data):
    layers = len(L)
    dataSet_Zs = np.empty((4,), dtype=object)
    for i in range(layers - 1):
        dataSet_Zs[i] = np.zeros((L[i] + 1, 1))
    dataSet_Zs[-1] = np.zeros((L[-1], 1))
    # First layer has input data
    dataSet_Zs[0] = np.concatenate((input_data, np.ones((n_samples, 1))), axis=1).T
    return dataSet_Zs


def build_initial_term_errors():
    layers = len(L)
    dataSet_term_erros = np.empty((4,), dtype=object)
    for layer in range(layers):
        dataSet_term_erros[layer] = np.zeros((2, 46))
    return dataSet_term_erros


def compute_CSqErr(real_output, predicted_output):
    CSqErr = 0
    squared_err = (real_output - predicted_output) ** 2
    # sum by column to get one row for both errors summed per prediction
    sum_of_columns = squared_err.sum(axis=0)
    CSqErr += sum_of_columns.sum()
    return CSqErr


def forward_propagate(B, Z):
    layers = len(L)
    dataSet_Ts = np.empty((4,), dtype=object)
    # first layer which is input
    dataSet_Ts[0] = np.ones((2, 1))
    for i in range(layers - 1):
        dataSet_Ts[i + 1] = B[i].T @ Z[i]
        layer_bias = np.ones((n_samples, 1)).T
        if (i + 1) < layers - 1:
            f_sig = 1 / (1 + np.exp(-dataSet_Ts[i + 1]))
            Z[i + 1] = np.concatenate((f_sig, layer_bias))
        else:
            Z[i + 1] = 1 / (1 + np.exp(-dataSet_Ts[i + 1]))
    return dataSet_Ts


def compute_delta_term_for_nodes(deltas):
    """
        Compute error delta term for all nodes except input
        1) compute error for output nodes
        2) compute error for hidden layers
        Note: Delta for out put node using sigmoid function:
                Delta = (Z(output) - Y) times Z(output) times(1-Z(output))

    """
    # Compute delta for output node
    deltas[-1] = (Z[-1] - Y) * Z[-1] * (1 - Z[-1])
    layers = len(L)
    # compute delta error term for rest of layers except input
    # subtract layers by 2 to exclude last layer
    for layer in range(layers - 2, 0, -1):
        # Z at that layer without its bias
        z_without_bias = Z[layer][
            :-1,
        ]
        weight = z_without_bias * (1 - z_without_bias)
        D = deltas[layer + 1].T
        for m in range(n_samples):
            # error summation for layer delta(L)
            summation = (D[m,] * betas[layer][:-1,]).sum(axis=1)
            deltas[layer][:, m] = weight[:, m] * summation


def update_weights(betas):
    layers = len(L)
    for i in range(layers - 1):
        weight = Z[i][
            :-1,
        ]
        v1 = np.zeros((L[i], L[i + 1]))
        v2 = np.zeros((1, L[i + 1]))
        D = deltas[i + 1].T
        for m in range(n_samples):
            v1 = v1 + (weight[:, m] @ D[m,])
            v2 = (
                v2 + D[m,]
            )
        # UPDATE BETAS WITHOUT BIAS
        betas[i][:-1,] = betas[i][:-1,] - (alpha / n_samples) * v1
        # UPDATE BIAS
        betas[i][-1,] = betas[i][-1,] - (alpha / n_samples) * v2


betas = build_starting_betas()
# print("betas", betas, betas.shape)
Z = build_initial_Zs(X)
# print("Z", Z, Z.shape)
deltas = build_initial_term_errors()
# print("d", deltas, deltas.shape)
while (mse > target_mse) and (epoch < max_epoch):
    print("mse =", mse)
    print("epoch = ", epoch)
    # This updates Z values and returns T values
    T_values = forward_propagate(betas, Z)
    CSqErr = compute_CSqErr(Y, Z[-1])
    # normalize err
    CSqErr = CSqErr / 2
    compute_delta_term_for_nodes(deltas)
    update_weights(betas)
    CSqErr = CSqErr / 46
    mse = CSqErr
    epoch += 1
    error.append(mse)
    epo.append(epoch)

    if mse < min_error:
        min_error = mse
        min_error_epoch = epoch
    else:
        print("WARNING--------")
print(min_error)
print(min_error_epoch)
