import pandas as pd
import numpy as np
from sklearn import preprocessing

# # Read dataset
# df = pd.read_excel("data.xlsx")
#
# # Target class
# tar = "win"


# 입력받은 data에 맞춰서 dataframe 수정
def set_dataframe(dataframe, data_list, target):
    tmp = data_list.copy()
    tmp.insert(0, target)

    if len(data_list) < 2:
        return print("Error: data_list should has more than one values!")

    try:
        new_df = dataframe[tmp]
    except:
        print("Error: data_list has unknown attributes!")

    return new_df


# Attribute list of dataframe (Except target attribute)
def attr_list(dataframe, target):
    tmp = np.array(dataframe.drop(columns=[target])._get_axis(1))
    attr = []
    for i in tmp:
        attr.append(i)

    return attr


# Attribute list
attr = attr_list(df, tar)


# Calculate distance function(Use the Euclidean distance)
# Return distance list
def distance(dataframe, attributes):
    # variable list
    v = []

    for k in attributes :
        v.append(np.array(dataframe[k]))

    # Distance list
    d = []

    for i in range(len(v[0]) - 1):
        result = 0

        for j in range(0, len(attributes)):
            result += (v[j][-1] - v[j][i]) ** 2
        d.append(np.sqrt(result))

    return d


# Normalize function
def normalize(data, attributes):
    # Normalize each attribute values
    for i in attributes:
        tmp = data[i]
        tmp = preprocessing.scale(tmp)
        data[i] = tmp

    return data


# kNN-Algorithm function
def kNN(data, attributes, target, hyper_k):
    target_list = np.array(data[tar].drop_duplicates())

    # Normalize all values
    data = normalize(data, attributes)

    # Calculate & Merge distance attribute
    dist_series = pd.Series(distance(data, attributes))
    data["DISTANCE"] = dist_series

    # Sorting for Computing K-nearest Data
    sorted_data = data.sort_values(by="DISTANCE", ascending=True).head(k)

    # Target Class 0 and 1
    class_0 = 0
    class_1 = 0

    # Check K-nearest data either class0 or class1
    for i in range(0, hyper_k):
        if sorted_data.iat[i, 0] == target_list[0]:
            class_0 += 1
        else:
            class_1 += 1

    # If class0 is more than(or same) class1, new data is class0
    if class_0 >= class_1:
        data.iat[-1, 0] = target_list[0]
    # If class1 is more than class0, new data is class1
    else:
        data.iat[-1, 0] = target_list[1]

    return data

k = 5

##################  Test  #########################
# Merge test data into rear of DataFrame
# Target class value is 'NaN'
# df.loc[len(df)] = (np.NaN, 10, 2, 3, 4, 2)
#
# # Execute kNN-Algorithm
# df = kNN(df, attr, tar, k)
#
# # Print result DataFrame
# print(df)
# print("\nTest data size:", df.iat[-1, 2])
