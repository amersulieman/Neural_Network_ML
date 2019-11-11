import math
import numpy as np
import os.path as path
import random as rd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


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


def forward_propagate(B, Z, Nx):
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


def build_test_Zs(outputs_test):
    """
        Allocate space for test Z, which is restult of sig function of T
    """
    dataSet_Zs = np.empty((layers_length,), dtype=object)
    for i in range(layers_length):
        dataSet_Zs[i] = np.zeros((L[i] + 1, 1))
    # output node Zs
    dataSet_Zs[-1] = np.zeros((outputs_test, 1))
    return dataSet_Zs


def check_correct_prediction(data, output_sample, typeTest):
    # data is the last z, in training i will have to loop for each one
    maximum_probability = np.amax(data)
    # this gives me where index of hotshot is not zero, which means my number
    number = np.nonzero(output_sample)[0][0]
    if typeTest == "training":
        number_probability = data[number]
    else:
        number_probability = data[number][0]
    if maximum_probability == number_probability:
        return True


def test_data(z_test, test_data, test_Nx, test_min_error):
    test_correct_guesses = 0
    test_error = []
    for j in range(test_Nx):
        z_test[0] = np.row_stack((test_data[j, :].reshape(2, 1), [1]))
        forward_propagate(betas, z_test, 1)
        output_sample = (y_test[:, j]).reshape(L[-1], 1)
        is_correct = check_correct_prediction(
            z_test[-1], output_sample, typeTest="tests"
        )
        if is_correct:
            test_correct_guesses += 1
        test_CSqErr = compute_CSqErr(output_sample, z_test[-1])
        # normalize err
        test_CSqErr = test_CSqErr / L[-1]
        test_CSqErr = test_CSqErr / test_Nx
        test_error.append(test_CSqErr)
    test_mse = sum(test_error)
    test_msess.append(test_mse)
    if test_mse < test_min_error:
        test_min_error = test_mse
    return test_min_error, test_correct_guesses


layers_to_do = [1, 3, 5, 7]
num_nodes = [20, 30, 50, 70]
figures_counter = 0
max_drawing = 200
best_mses_per_layers = []
for layer_run in range(len(layers_to_do)):
    # config values
    nodes = num_nodes[layer_run]
    L = [2]
    for x in range(layers_to_do[layer_run]):
        L.append(nodes)
    L.append(10)
    alpha = 0.2
    target_mse = 0.0001
    max_epoch = 2000
    min_error = math.inf
    min_error_epoch = -1
    epoch = 0  # one epoch is one forward and backward sweep
    mse = math.inf
    error = []
    epo = []

    # load training data
    x_file = path.abspath("./X.txt")
    y_file = path.abspath("./Y.txt")
    Y = np.loadtxt(y_file)
    _, outputs = Y.shape
    X = np.loadtxt(x_file)
    Nx, input_features = X.shape
    layers_length = len(L)
    Y = Y.T
    training_accuracy = []
    #################################
    """
        Testing data
    """
    # LOAD TESTING DATA
    x_test = path.abspath("./X_test.txt")
    x_test = np.loadtxt(x_test)
    test_Nx, test_input_features = x_test.shape
    y_test = path.abspath("./Y_test.txt")
    y_test = np.loadtxt(y_test)
    _, test_outputs = y_test.shape
    y_test = y_test.T
    z_test = build_test_Zs(test_outputs)
    test_min_error = math.inf
    test_mse = math.inf
    test_msess = []
    tests_accuracy = []

    #################################

    betas = build_starting_betas()
    Z = build_initial_Zs(X)
    deltas = build_initial_deltas()
    while (mse > target_mse) and (epoch < max_epoch):
        total_test_correct_guesses = 0
        total_training_correct_guesses = 0
        print("mse =", mse)
        print("epoch = ", epoch)
        # This updates Z values and returns T values
        forward_propagate(betas, Z, Nx)
        for train_sample in range(Nx):
            output_sample = (Y[:, train_sample]).reshape(L[-1], 1)
            training_sample = Z[-1][:, train_sample]
            is_correct = check_correct_prediction(
                training_sample, output_sample, typeTest="training"
            )
            if is_correct:
                total_training_correct_guesses += 1
        train_acc = total_training_correct_guesses / Nx
        training_accuracy.append(train_acc)
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

        test_min_error, test_correct_guesses = test_data(
            z_test, x_test, test_Nx, test_min_error
        )
        total_test_correct_guesses += test_correct_guesses
        acc = total_test_correct_guesses / test_Nx
        tests_accuracy.append(acc)
        print("Train accuracy", train_acc)
        print("tests_accuracy", acc)
    best_mses_per_layers.append([min(error), min(test_msess)])
    print(test_msess)
    ax1 = plt.figure(figures_counter).gca()
    figures_counter += 1
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    epo = epo[:max_drawing]
    error = error[:max_drawing]
    plt.plot(epo, error)
    test_msess = test_msess[:max_drawing]
    plt.plot(epo, test_msess)
    plt.title("Training/Test MSE layers {}".format(layers_to_do[layer_run]))
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.legend(["y = Train MSE", "y = Test MSE"], loc="upper right")
    plt.savefig("Training_Test_MSE_layers{}.png".format(layers_to_do[layer_run]))

    ax2 = plt.figure(figures_counter).gca()
    figures_counter += 1
    training_accuracy = training_accuracy[:max_drawing]
    plt.plot(epo, training_accuracy)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    tests_accuracy = tests_accuracy[:max_drawing]
    plt.plot(epo, tests_accuracy)
    plt.title("Training/Test Accuracy layers{}".format(layers_to_do[layer_run]))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["y = Train Accuracy", "y = Test Accuracy"], loc="upper left")
    plt.savefig("Training_Test_Accuracy_layers{}.png".format(layers_to_do[layer_run]))

# mse_rows = []
# for i in range(max_epoch):
#     mse_rows.append([error[i], test_msess[i]])

fig = plt.figure(figures_counter)
figures_counter += 1
# ax = fig.add_subplot(111)
col_labels = ["Layers", "Training Mse", "Testing Mse"]

table_vals = [
    [layers_to_do[i], best_mses_per_layers[i][0], best_mses_per_layers[i][1]]
    for i in range(len(layers_to_do))
]

# Draw table
the_table = plt.table(
    cellText=table_vals, colWidths=[0.3] * 3, colLabels=col_labels, loc="center"
)
the_table.auto_set_font_size(False)
the_table.set_fontsize(24)
the_table.scale(4, 4)

# Removing ticks and spines enables you to get the figure only with table
plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.tick_params(axis="y", which="both", right=False, left=False, labelleft=False)
for pos in ["right", "top", "bottom", "left"]:
    plt.gca().spines[pos].set_visible(False)
plt.savefig(
    "matplotlib_table_layers{}.png".format(layers_to_do[layer_run]),
    bbox_inches="tight",
    pad_inches=0.05,
)
