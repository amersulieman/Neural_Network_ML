import numpy as np

fileName = "./train.txt"
raw_data = open(fileName, "rt")
data = np.loadtxt(raw_data)
y_output = data[:, 0]
features_256 = np.delete(data, 0, axis=1)
six = features_256[0].reshape((16, 16))

# get white spaces count
num_white_space = (six == -1.0).sum()

# width
found_widthA = False
found_widthB = False
backward_counter = -1
for column in range(16):
    if found_widthB and found_widthA:
        break
    if not found_widthA:
        six_column = six[:, column]
        if len(set(six_column)) > 1:
            found_widthA = True
            widthA = column
    if not found_widthB:
        six_columnB = six[:, backward_counter]
        if len(set(six_columnB)) > 1:
            found_widthB = True
            widthB = 16 + backward_counter
        else:
            backward_counter -= 1
width = widthB - widthA

# height
found_heightA = False
found_heightB = False
backward_counter = -1
for row in range(16):
    if found_heightB and found_heightA:
        break
    if not found_heightA:
        if len(set(six[row])) > 1:
            found_heightA = True
            heightA = row
    if not found_heightB:
        if len(set(six[backward_counter])) > 1:
            found_heightB = True
            heighB = 16 + backward_counter
        else:
            backward_counter -= 1
height = heighB - heightA

