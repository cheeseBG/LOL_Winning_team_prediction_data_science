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

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Tree의 개수 Random Forest 분류 모듈 생성
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
# rfc.fin()에 훈련 데이터를 입력해 Random Forest 모듈을 학습
rf.fit(X_train, y_train)  # 소환사 코드를 float로 바꿔줘야함

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
predicted = rf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)

print('\n<Ensemble learning>')
# oob_score(out of bag score)로써 예측이 얼마나 정확한가에 대한 추정치
print('\n1. Random Forest Classifier')
print('Confusion matrix : \n', confusion_matrix(y_test, predicted))
print(f'The number of testing error : ', (y_test != predicted).sum())
print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')
print(f'Mean accuracy score: {accuracy:.3}')
print('\n', classification_report(y_test, predicted))

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

clf_gbc = GradientBoostingClassifier()
clf_gbc.fit(X_train, y_train)
y_pred = clf_gbc.predict(X_test)

print('\n2. Gradient Boosting Classifier')
print('Confusion matrix : \n', confusion_matrix(y_test, y_pred))
print(f'The number of testing error : ', (y_test != y_pred).sum())
print(f'Test accuracy : {(accuracy_score(y_test, y_pred)):.3}')
print('\n', classification_report(y_test, y_pred))

# firstBlood = input("firstBlood(True:1, False:0) : ")
# firstTower = input("firstTower(True:1, False:0) : ")
# firstInhibitor = input("firstInhibitor(True:1, False:0) : ")
# firstBaron = input("firstBaron(True:1, False:0) : ")
# firstDragon = input("firstDragon(True:1, False:0) : ")
# firstRiftHerald = input("firstHerald(True:1, False:0) : ")
# towerKills = input("towerKills : ")
# inhibitorKills = input("inhibitorKills : ")
# baronKills = input("baronKills : ")
# dragonKills = input("dragonKills : ")
# vilemawKills = input("vilemawKills : ")
# riftHeraldKills = input("riftHeraldKills : ")
#
# # Predicting input values ​​using k-means clustering
# X2 = X.index + 1
#
# X2 = X2.loc[:, :]
# X2.loc[0] = [firstBlood, firstTower, firstInhibitor, firstBaron, firstDragon, firstRiftHerald,
#                  towerKills, inhibitorKills, baronKills, dragonKills, vilemawKills, riftHeraldKills]
# X2 = X2.sort_index()
#
# kmeans2 = KMeans(n_clusters=2)
# y_kmeans2 = kmeans.fit_predict(X2)
# kmeans2_accuracy = accuracy_score(X2, y_kmeans2)
#
# if y_kmeans2[1] == 0:
#     Lose = 0
# else:
#     Lose = 1
#
# print('\n<Prediction using k-means clustering>')
#
# if y_kmeans2[0] == Lose:
#     print('Predicted Result : Lose')
#     print(f'Accuracy : {kmeans2_accuracy:.3}')
# else:
#     print('Predicted Result : Win')
#     print(f'Accuracy : {kmeans2_accuracy:.3}')
#
# # Predicting input values ​​using random forest
#
# X2_test = X_test.index + 1
# y2_test = y_test.index + 1
# X2_test = X2_test.sort_index()
#
# predicted2 = rf.predict(X2_test)
# y2_test[0] = predicted2[0]
# accuracy2 = accuracy_score(y2_test, predicted2)
#
# print('\n<Prediction using random forest>')
# print('len : ', len(y2_test))
# if predicted[0] == 0:
#     print('Predicted Result : Win')
#     print(f'Accuracy : {accuracy2:.3}')
# else:
#     print('Predicted Result : Lose')
#     print(f'Accuracy : {accuracy2:.3}')

