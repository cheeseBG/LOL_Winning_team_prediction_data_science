import pandas as pd
import numpy as np
from sklearn import preprocessing


def predict(team_df,s_df, target):
    attr = attr_list(team_df, target)
    if attr.__len__() > 1:
        tmp_df = reset_dataframe(s_df, attr, target)
        tmp_df.loc[tmp_df.__len__()] = np.array(team_df.iloc[0, 0:])
        tmp_df = kNN(tmp_df, target, 3)
        print("\n****** Predict blue team ******")
        print(tmp_df.iloc[-1, 0:1])

# One-hot encoding( Categorical data -> numerical data)
def one_hot_encoding(dataframe):
    dataframe.replace(True, "1", inplace=True)
    dataframe.replace(False, "0", inplace=True)

    return dataframe

# Reset dataframe with input events
def reset_dataframe(dataframe, data_list, target):
    tmp = data_list.copy()
    tmp.insert(0, target)

    if len(data_list) < 2:
        return print("Error: data_list should has more than one values!")

    new_df = dataframe[tmp]

    return new_df


# Attribute list of dataframe (Except target attribute)
def attr_list(dataframe, target):
    tmp = np.array(dataframe.drop(columns=[target])._get_axis(1))
    attr = []
    for i in tmp:
        attr.append(i)

    return attr


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
def kNN(data, target, hyper_k):

    # Create attributes list
    attributes = attr_list(data, target)

    target_list = np.array(data[target].drop_duplicates())

    # Normalize all values
    data = normalize(data, attributes)

    # Calculate & Merge distance attribute
    dist_series = pd.Series(distance(data, attributes))
    data["DISTANCE"] = dist_series

    # Sorting for Computing K-nearest Data
    sorted_data = data.sort_values(by="DISTANCE", ascending=True).head(hyper_k)

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

    return data.drop(columns="DISTANCE")









