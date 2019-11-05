import numpy as np

fileName = "./train.txt"
raw_data = open(fileName, "rt")
data = np.loadtxt(raw_data)
y_output = data[:, 0]
features_256 = np.delete(data, 0, axis=1)
mapped_features = []
np.savetxt("Y.txt", y_output)
for x in range(features_256.shape[0]):
    print(x)
    # change it to columns and rows
    number = features_256[x].reshape((16, 16))
    # get white spaces count
    num_white_space = (number == -1.0).sum()

    # width
    found_widthA = False
    found_widthB = False
    backward_counter = -1
    for column in range(16):
        if found_widthB and found_widthA:
            break
        if not found_widthA:
            num_column = number[:, column]
            if len(set(num_column)) > 1:
                found_widthA = True
                widthA = column
        if not found_widthB:
            num_columnB = number[:, backward_counter]
            if len(set(num_columnB)) > 1:
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
            if len(set(number[row])) > 1:
                found_heightA = True
                heightA = row
        if not found_heightB:
            if len(set(number[backward_counter])) > 1:
                found_heightB = True
                heighB = 16 + backward_counter
            else:
                backward_counter -= 1
    height = heighB - heightA

    # pick 3 spots from where number is drawn and check how many white spaces they have
    # 3 spots needed: hight/2, height/4, height/6
    spots = [height // 2, height // 4, height // 6]
    shaded = 0
    for spot in spots:
        spot_row = number[spot][widthA : widthB + 1]
        white_spacesss = (spot_row == -1.0).sum()
        # this tells me out of that width pixles, minus white spaces and rest is shaded
        shaded += len(spot_row) - white_spacesss
    ratio = width / height
    mapped_features.append([ratio, shaded])

print(mapped_features)

a = np.array(mapped_features)
mat = np.matrix(a)
with open("X.txt", "wb") as f:
    for line in mat:
        np.savetxt(f, line, fmt="%.2f")

