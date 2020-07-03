"""
Title : Data Science Clustering
Author : Junho Kim
"""

import pandas as pd
import glob
import os
from sklearn.cluster import KMeans

lose = pd.read_csv("match_loser_data_version1.csv")
win = pd.read_csv("match_winner_data_version1.csv")

print('match_loser_data : \n', lose)
print('match_win_data : \n', win)

# Path to read the entire file
input_file = r'/Users/kimjunho/PycharmProjects/Clustering'
# Path to save the combined file
output_file = r'/Users/kimjunho/PycharmProjects/Clustering/combined_data.csv'

# Collect files starting with match_*_data_version1.csv as a glove function
allFile_list = glob.glob(os.path.join(input_file, 'match_*_data_version1.csv'))
# Full output of read file path and filename
print(allFile_list)

allData = []  # Create an empty list to save the contents of the csv file you read
for file in allFile_list:
    df = pd.read_csv(file)  # for syntax reads the csv files
    allData.append(df)  # Add what you read to the empty list

dataCombine = pd.concat(allData, axis=0, ignore_index=True)  # Merge the contents of the list using the concat function
dataCombine.to_csv(output_file, index=False)  # to_csv function. (Set to False to remove the inductance)

df = pd.read_csv('combined_data.csv')

print('\nEntire match_data : \n', df, '\n')
all_column = ['firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald',
              'towerKills', 'inhibitorKills', 'baronKills', 'dragonKills', 'vilemawKills', 'riftHeraldKills']
x_column = ['towerKills', 'inhibitorKills', 'baronKills', 'dragonKills', 'vilemawKills', 'riftHeraldKills']
y_column = 'win'

data = df[all_column].astype('int')

# All x_column data
x_data = data.iloc[:, :]

# print(x_data)
firstBlood = input("firstBlood(True:1, False:0) : ")
firstTower = input("firstTower(True:1, False:0) : ")
firstInhibitor = input("firstInhibitor(True:1, False:0) : ")
firstBaron = input("firstBaron(True:1, False:0) : ")
firstDragon = input("firstDragon(True:1, False:0) : ")
firstRiftHerald = input("firstHerald(True:1, False:0) : ")
towerKills = input("towerKills : ")
inhibitorKills = input("inhibitorKills : ")
baronKills = input("baronKills : ")
dragonKills = input("dragonKills : ")
vilemawKills = input("vilemawKills : ")
riftHeraldKills = input("riftHeraldKills : ")

x_data.index = x_data.index + 1
x_data.loc[0] = [firstBlood, firstTower, firstInhibitor, firstBaron, firstDragon, firstRiftHerald,
                 towerKills, inhibitorKills, baronKills, dragonKills, vilemawKills, riftHeraldKills]
x_data = x_data.sort_index()

x = x_data.iloc[:, :].values

kmeans2 = KMeans(n_clusters=2)
y_kmeans2 = kmeans2.fit_predict(x)

x_data['clustering'] = y_kmeans2
x_data.to_csv('cluster_result.csv', index=False)

if y_kmeans2[1] == 0:
    Lose = 0
else:
    Lose = 1

if y_kmeans2[0] == Lose:
    print('\nPredicted Result : Lose')
else:
    print('\nPredicted Result : Win')








