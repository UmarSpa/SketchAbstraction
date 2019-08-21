import numpy as np
import scipy.misc
from bresenham import bresenham

def get_bounds(data, factor=10):
    """Generate bounds of data."""
    min_x, max_x, min_y, max_y = 0, 0, 0, 0
    abs_x, abs_y = 0, 0
    for i in range(len(data)):
        x = float(data[i, 0]) / factor
        y = float(data[i, 1]) / factor
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)
    return (min_x, max_x, min_y, max_y)

def make_grid(s_list, start_loc_val = [], normalizing_factor_val = []):
    """Generate 2D grid with starting value and normalizing factor."""
    def get_start_and_end(x):
        x = np.array(x)
        x = x[:, 0:2]
        x_start, x_end = x[0], x.sum(axis=0)
        x = x.cumsum(axis=0)
        x_max, x_min = x.max(axis=0), x.min(axis=0)
        center_loc = (x_max+x_min)*0.5
        return x_start-center_loc, x_end
    if len(s_list[0][0])==0: return []
    x_pos, y_pos = 0.0, 0.0
    result = [[x_pos, y_pos, 1]]
    for sample in s_list:
        s = sample[0]
        if start_loc_val == []: start_loc, delta_pos = get_start_and_end(s)
        else: start_loc = start_loc_val
        s[0, 0],  s[0, 1] = start_loc[0], start_loc[1]
        if normalizing_factor_val == []:
            min_x, max_x, min_y, max_y = get_bounds(np.array(s), 1.0)
            normalizing_factor = max(abs(min_x), abs(max_x), abs(min_y), abs(max_y))
            normalizing_factor = 1 if normalizing_factor == 0 else normalizing_factor
        else:
            normalizing_factor = normalizing_factor_val
        s[:,0:2] = s[:,0:2] / float(normalizing_factor)
        result += s.tolist()
        result[-1][2] = 1
    return np.array(result), start_loc, normalizing_factor

def grid_raster(data, strokeVals=[], output_max_dim=256.0):
    """Generate rastered respresentation of the 2D grid."""
    if strokeVals == []: outputImg = np.zeros((int(output_max_dim), int(output_max_dim)), dtype=np.int) + 1
    else: outputImg = np.zeros((int(output_max_dim), int(output_max_dim)), dtype=np.float)
    if data == []: return outputImg, np.zeros((int(output_max_dim), int(output_max_dim)), dtype=np.float32)

    data[:, 0:2] = data[:, 0:2] * (output_max_dim / 2 - 33)
    dataProcessed = data.copy()
    dataProcessed[0, 0] = output_max_dim / 2
    dataProcessed[0, 1] = output_max_dim / 2
    dataProcessed = dataProcessed.cumsum(axis=0)
    dataProcessed[:, 2] = data[:, 2]
    initX, initY = 0, 0
    lift_pen = 1
    for i in range(len(dataProcessed)):
        x, y = float(dataProcessed[i, 0]), float(dataProcessed[i, 1])
        if (lift_pen == 0):
            cordList = list(bresenham(initX, initY, int(x), int(y)))
            for cord in cordList:
                if strokeVals == []:
                    outputImg[cord[1], cord[0]] = 0
        lift_pen = data[i, 2]
        initX, initY = int(x), int(y)
    return outputImg

def global_to_standard(sketchBucket, sketchBucketLen):
    """Convert data with initial global coordinates to the one without initial cooridnates."""
    if len(sketchBucket) == 0: return [], [], []
    sketchArray = np.zeros([1, 3])
    sketchBucketArr = np.array(sketchBucket)
    for strokeID in range(sketchBucketArr.shape[0]):
        if strokeID == 0:
            sketchArray = np.append(sketchArray, sketchBucketArr[strokeID, 0:sketchBucketLen[strokeID] + 1, :], axis=0)
            if sum(sketchBucketArr[strokeID, 0, 0:2]) != 0:
                sketchArray[1, 0:2] = sketchArray[1, 0:2] - sketchArray[2, 0:2]
            continue
        if sum(sketchBucketArr[strokeID - 1, 0, 0:2]) == 0:
            temp = sketchBucketArr[strokeID - 1, 0:sketchBucketLen[strokeID - 1] + 1, :].sum(axis=0)
        else:
            temp = sketchBucketArr[strokeID - 1, 0:sketchBucketLen[strokeID - 1] + 1, :].sum(axis=0) - sketchBucketArr[strokeID - 1, 1, :]
        temp1 = sketchBucketArr[strokeID, 0, :] - temp  # sum of global - end point of the previous stroke
        if sum(sketchBucketArr[strokeID, 0, 0:2]) == 0: temp1 = temp1 + sketchBucketArr[strokeID, 1, :]
        tempBucketArr = sketchBucketArr[strokeID].copy()
        tempBucketArr[1, 0:2] = temp1[0:2]
        sketchArray = np.append(sketchArray, tempBucketArr[1:sketchBucketLen[strokeID] + 1, :], axis=0)
    return sketchArray[1:]

def sketch_raster(sketchData, preprocessImg=False, start_loc_val = [], normalizing_factor_val = []):
    """Generate the rastered representation of the vectorized data."""
    stroke_grid, _ , _ = make_grid([[sketchData.copy(), [0, 0]]], start_loc_val = start_loc_val, normalizing_factor_val = normalizing_factor_val)
    outputImg = grid_raster(stroke_grid.copy())
    if preprocessImg: outputImg = preprocessing(outputImg)
    outputImg = np.expand_dims(outputImg, axis=0)
    outputImg = np.expand_dims(outputImg, axis=3)
    return outputImg

def preprocessing(outputImg):
    """Preprocessing of the rastered image."""
    outputImg = outputImg != 1
    outputImg = scipy.ndimage.binary_dilation(outputImg)
    outputImg = (outputImg == False) * 255.0
    outputImg = outputImg[15:15 + 225, 15:15 + 225].astype(np.float32) - 250.42
    return outputImg

def sketch_raster_info(sketchBucket, sketchBucketLens):
    """Generate the starting point and normalizing factor for the 2D grid.."""
    return sketch_raster_info_(global_to_standard(sketchBucket.copy(), sketchBucketLens.copy()))

def sketch_raster_info_(sketchArray):
    _, startPos, normalizingFac = make_grid([[sketchArray.copy(), [0, 0]]])
    return startPos, normalizingFac

