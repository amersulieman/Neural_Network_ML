"""
    File takes Training Input and convert it to proper files to train with
"""

import numpy as np


def find_diminsion(sample, diminsion):
    """
        Function looks for width or heigh depend on what is called wiyj
    """
    found_dim_start = False
    found_dim_end = False
    backward_counter = -1
    # there are 16 columns/rows search from beginning and end till find first occurrance
    for side in range(16):
        if found_dim_start and found_dim_end:
            break
        if not found_dim_start:
            if diminsion == "width":
                dim_location = sample[:, side]
            else:
                dim_location = sample[side]
            # if set has more than one, then there was other number than -1
            if len(set(dim_location)) > 1:
                found_dim_start = True
                # width/column begin at that side
                dim_start = side
        if not found_dim_end:
            if diminsion == "width":
                dim_location = sample[:, backward_counter]
            else:
                dim_location = sample[backward_counter]
            # if set has more than one, then there was other number than -1
            if len(set(dim_location)) > 1:
                found_dim_end = True
                # 16 is because counter is negative to start from end of list
                dim_end = 16 + backward_counter
            else:
                # increment backward counter for next loop
                backward_counter -= 1
    diminsion_needed = dim_end - dim_start
    return diminsion_needed, dim_start, dim_end


def find_shadded_spots(sample, height, width_start, width_end):
    """
        NOTE:
        1) After finding width and height the location
            I pick 3 spots from where sample is drawn 
            and check how many shaded pixels it has
        2) The 3 spots needed: hight/2, height/4, height/6
    """
    spots = [height // 2, height // 4, height // 6]
    shaded = 0
    for location in spots:
        spot_row = sample[location][width_start : width_end + 1]
        white_spacesss = (spot_row == -1.0).sum()
        # IF out of 16 5 are white, then rest are shaded
        shaded += len(spot_row) - white_spacesss
    return shaded


def convert_hotshot(output_data):
    # list to hold all hotshot data
    hotshot_array = []
    for number in output_data:
        # create array of 10 zeros initially per sample
        hotshot_initial = np.zeros((1, 10))
        # sample at the location of the number output gets 1 and rest is zero
        hotshot_initial[0][int(number)] = 1
        hotshot_array.append(hotshot_initial)
    return np.array(hotshot_array)


def save_to_file(data, fmt, output_file):
    # convert array to matrix
    matrix = np.matrix(data)
    with open(output_file, "wb") as f:
        for line in matrix:
            np.savetxt(f, line, fmt=fmt)


def prep_data(fileName):
    raw_data = open(fileName, "rt")
    data = np.loadtxt(raw_data)
    # Get output column from train data
    y_output = data[:, 0]
    # delete the output column and keep features
    features_256 = np.delete(data, 0, axis=1)
    n_samples, _ = features_256.shape
    mapped_features = []
    # feature mapping
    for i in range(n_samples):
        # change row sample to columns and rows
        sample = features_256[i].reshape((16, 16))
        # get white spaces count for that sample
        num_white_space = (sample == -1.0).sum()
        width, width_start, width_end = find_diminsion(sample, diminsion="width")
        height, height_start, height_end = find_diminsion(sample, diminsion="height")
        feature1_ratio = width / height
        feature2_shaded_at_locations = find_shadded_spots(
            sample, height, width_start, width_end
        )
        mapped_features.append([feature1_ratio, feature2_shaded_at_locations])
    return y_output, mapped_features


# prep training data
print("Training data prep...")
training_output_data, training_features = prep_data("./train.txt")
print("Test data prep...")
testing_output_data, testing_features = prep_data("./test.txt")
print("Convert training output to hotshot...")
training_hotshot_output = convert_hotshot(training_output_data)
print("Convert testing output to hotshot...")
testing_hotshot_output = convert_hotshot(testing_output_data)


# save features of testing and training to files
print("Saving training features..")
training_features = np.array(training_features)
save_to_file(training_features, fmt="%.2f", output_file="X.txt")
print("Saving testing features..")
testing_features = np.array(testing_features)
save_to_file(testing_features, fmt="%.2f", output_file="X_test.txt")
# Save output hots of training and testing to files
print("Saving hotshot training output..")
save_to_file(training_hotshot_output, fmt="%d", output_file="Y.txt")
print("Saving hotshot testing output..")
save_to_file(testing_hotshot_output, fmt="%d", output_file="Y_test.txt")
