import math
import numpy as np
import os.path as path
import random as rd

# build matrices for B and T and rest
def build_starting_betas():
    """
        Generate random betas within range [-0.7 0.7]
    """
    dataSetBetas = np.empty((layers_length - 1,), dtype=object)
    # first beta is for input beta so it is different size than rest
    dataSetBetas[0] = np.random.uniform(
        low=-0.7, high=0.8, size=(input_features + 1, nodes)
    )
    for layer in range(1, layers_length - 1):
        dataSetBetas[layer] = np.random.uniform(
            low=-0.7, high=0.8, size=(nodes + 1, nodes)
        )
    dataSetBetas[-1] = np.random.uniform(low=-0.7, high=0.8, size=(nodes + 1, outputs))
    return dataSetBetas


def build_initial_Zs(input_data):
    """
        Allocate space for Z, which is restult of sig function of T
    """
    dataSet_Zs = np.empty((layers_length,), dtype=object)
    for i in range(layers_length - 1):
        dataSet_Zs[i] = np.zeros((L[i] + 1, Nx))
    # output node Zs
    dataSet_Zs[-1] = np.zeros((outputs, Nx))
    # First layer has input data, so First Z is the inputs with bias
    dataSet_Zs[0] = np.concatenate((input_data, np.ones((Nx, 1))), axis=1).T
    return dataSet_Zs


def build_initial_deltas():
    """
        Allocate space for delta error, for backward propogation
    """
    initial_deltas = np.empty((layers_length,), dtype=object)
    for layer in range(layers_length):
        initial_deltas[layer] = np.zeros((L[layer], Nx))
    return initial_deltas


def compute_CSqErr(real_output, predicted_output):
    """
        Here we compute error squared 
    """
    CSqErr = 0
    squared_err = (real_output - predicted_output) ** 2
    # sum by column to get one row for both errors summed per prediction
    sum_of_columns = squared_err.sum(axis=0)
    CSqErr += sum_of_columns.sum()
    return CSqErr


def forward_propagate(B, Z):
    """
        Forward propagate the network so I can count error
    """
    Ts = np.empty((layers_length,), dtype=object)
    # first layer which is input, so it does not matter for T
    Ts[0] = np.ones((input_features, 1))
    for i in range(layers_length - 1):
        # The next T is equal to previous Beta times previous Z
        Ts[i + 1] = B[i].T @ Z[i]
        # layer bias to add per layer except output
        layer_bias = np.ones((Nx, 1)).T
        # Checks as long as we have not reached output layers
        if (i + 1) < layers_length - 1:
            # activation function of T
            T_sig = 1 / (1 + np.exp(-Ts[i + 1]))
            # add bias to result of NEXT Z
            Z[i + 1] = np.concatenate((T_sig, layer_bias))
        else:
            # Output nodes don't get bias added to them
            Z[i + 1] = 1 / (1 + np.exp(-Ts[i + 1]))


def compute_delta_error(deltas):
    """
        Compute error delta term for all nodes except input
        1) compute error for output nodes
        2) compute error for hidden layers
        Note: Delta for out put node using sigmoid function:
                Delta = (Z(output) - Y) times Z(output) times(1-Z(output))

    """
    # Compute delta for output node
    deltas[-1] = (Z[-1] - Y) * Z[-1] * (1 - Z[-1])
    # Compute from second to last layer till first layer delta errors
    for layer in range(layers_length - 2, 0, -1):
        # Z at that layer without its bias
        z_without_bias = Z[layer][:-1]
        weight = z_without_bias * (1 - z_without_bias)
        D = deltas[layer + 1].T
        for m in range(Nx):
            # error summation for layer delta(L)
            summation = (D[m,] * betas[layer][:-1]).sum(axis=1)
            deltas[layer][:, m] = weight[:, m] * summation


def update_weights(betas):
    """
        Update weights/betas after backward propagation done
    """
    for i in range(layers_length - 1):
        weight = Z[i][:-1]
        v1 = np.zeros((L[i], L[i + 1]))
        v2 = np.zeros((1, L[i + 1]))
        D = deltas[i + 1].T
        for m in range(Nx):
            temp = (D[m, :]).reshape(L[i + 1], 1)
            v1 = v1 + (weight[:, m] * temp).T
            v2 = v2 + D[m]
        # UPDATE BETAS WITHOUT BIAS
        betas[i][:-1,] = betas[i][:-1,] - (alpha / Nx) * v1
        # UPDATE BIAS
        betas[i][-1,] = betas[i][-1,] - (alpha / Nx) * v2


# config values
nodes = 20
L = [2, nodes, nodes, nodes, 10]  # layers
alpha = 0.2
target_mse = 0.0001
max_epoch = 200
min_error = math.inf
min_error_epoch = -1
epoch = 0  # one epoch is one forward and backward sweep
mse = math.inf
error = []
epo = []

# load data
x_file = path.abspath("./X.txt")
y_file = path.abspath("./Y.txt")
Y = np.loadtxt(y_file)
_, outputs = Y.shape
X = np.loadtxt(x_file)
Nx, input_features = X.shape
layers_length = len(L)
Y = Y.T

betas = build_starting_betas()
Z = build_initial_Zs(X)
deltas = build_initial_deltas()
while (mse > target_mse) and (epoch < max_epoch):
    print("mse =", mse)
    print("epoch = ", epoch)
    # This updates Z values and returns T values
    forward_propagate(betas, Z)
    CSqErr = compute_CSqErr(Y, Z[-1])
    # normalize err
    CSqErr = CSqErr / L[-1]
    # backward propagate
    compute_delta_error(deltas)
    # after knowing better beta values update original ones
    update_weights(betas)
    # divide error by number of samples
    CSqErr = CSqErr / Nx
    mse = CSqErr
    epoch += 1
    error.append(mse)
    epo.append(epoch)
    if mse < min_error:
        min_error = mse
        min_error_epoch = epoch

print(min_error)
print(min_error_epoch)
