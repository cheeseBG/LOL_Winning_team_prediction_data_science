"""
Title : Data Science Ensemble
Author : Junho Kim
"""

import pandas as pd
import numpy as np
import glob
import os
from sklearn.cluster import KMeans

lose = pd.read_csv("match_loser_data_version1.csv")
win = pd.read_csv("match_winner_data_version1.csv")

print('match_loser_data : \n', lose)
print('match_win_data : \n', win)

# Path to read the entire file
input_file = r'/Users/kimjunho/PycharmProjects/Ensemble'
# Path to save the combined file
output_file = r'/Users/kimjunho/PycharmProjects/Ensemble/combined_data.csv'

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

data = pd.read_csv('combined_data.csv')
print('\nEntire match_data : \n', data, '\n')

data_team = data.dropna(axis=0)
data_team2 = data_team[list(data_team.columns)[2:]]

data_team2 = data_team2[
    ['firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald',
     'towerKills', 'inhibitorKills', 'baronKills', 'dragonKills', 'vilemawKills', 'riftHeraldKills']].astype('int')
dict_winner2 = {'Win': 0, 'Fail': 1}
data_team['win'].map(dict_winner2).tolist()

X = data_team2
y = np.array(data_team['win'].map(dict_winner2).tolist())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=123456)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

print('The number of data : ', (len(data)))
print('The number of training : ', (len(y_train)), '(', round(len(y_train)*100/len(data), 2), '%)')
print('The number of testing : ', (len(y_test)), '(', round(len(y_test)*100/len(data), 2), '%)')

# Only testing data accuracy using K-Means clustering
kmeans = KMeans(n_clusters=2)
y_kmeans = kmeans.fit_predict(X_test)
kmeans_accuracy = accuracy_score(y_test, y_kmeans)

print('\n<K-means Clustering>')
# 목표, 예측 매트릭스
print('Confusion matrix : \n', confusion_matrix(y_test, y_kmeans))
print(f'The number of testing error : ', (y_test != y_kmeans).sum())
print(f'K-means clustering accuracy of testing data : {kmeans_accuracy:.3}')
# precision = 정밀도, recall = 재현율, fl-score = 정밀도, 민감도 조화평균
print('\n', classification_report(y_test, y_kmeans))

# Test data를 입력해 target data를 예측 (매번 달라짐)
predicted = classifier.predict(X_test)
accuracy = accuracy_score(y_test, predicted)

print('\n<Ensemble Learning>')
print('Confusion matrix : \n', confusion_matrix(y_test, predicted))
print(f'The number of testing error : ', (y_test != predicted).sum())
print(f'Mean accuracy score: {accuracy:.3}')
print('\n', classification_report(y_test, predicted))
