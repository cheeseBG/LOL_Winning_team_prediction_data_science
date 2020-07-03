import pandas as pd
import warnings
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from kNNAlgorithm.kNN_algorithm import one_hot_encoding, predict, kNN, reset_dataframe, attr_list


import glob
import os
from sklearn.cluster import KMeans


warnings.filterwarnings(action='ignore')

lose = pd.read_csv("match_loser_data_version1.csv")
win = pd.read_csv("match_winner_data_version1.csv")

print('match_loser_data : \n', lose)
print('match_win_data : \n', win)

allData = [lose, win]  # Create an empty list to save the contents of the csv file you read

df = pd.concat(allData, axis=0, ignore_index=True)  # Merge the contents of the list using the concat function


# Warning off
pd.set_option('mode.chained_assignment', None)


# Remove outlier function
def low_outlier(df, col):
    mean = df["Pick rate"].mean()
    std = df["Pick rate"].std()
    for i in df.index:
        val = df.at[i, col]
        if val < mean - 3 * std:
            df.at[i, col] = np.nan
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df


def value_checker(df, champ, tier):
    df_tier = df[df['Rank'] == tier]

    for i in df_tier.index:
        if champ == df.at[i, 'Name']:
            return 2
    for i in df.index:
        if champ == df.at[i, 'Name']:
            return 1
    return 0


winning = pd.read_csv('WinningRate_dirty.csv', encoding='utf-8')

# Set Target class
tar = "win"

# Remove exclude features
del df["Unnamed: 0"]
del df["teamId"]
del df["gameId"]
del df["bans"]
del df["dominionVictoryScore"]

# Change categrical data -> numerical data
df = one_hot_encoding(df)


menu = True
sel_menu = int()

menu_list = ["1", "2", "3", "4", "5", "6"]

while menu is True:
    print("\n")
    print("#" * 10 + " DataScience Term Project " + "#" * 10)
    print("\n# Menu List")
    print("# 1: Predicting LoL match result in live time")
    print("# 2: Display K-fold, Confusion matrix results (kNN algorithm) ")
    print("# 3: Display Clustering result with user input data")
    print("# 4: Display Clustering and Ensemble accuracy of test data")
    print("# 5: Data explore")
    print("# 6: Exit")

    sel_menu = input("\n# Select Menu: ")

    # If user enter value which is not in menu list,
    # send error message.
    if menu_list.count(sel_menu) == 0:
        print("Error: Enter  a number in menu list")
    # ########## Menu 1 ##########
    elif sel_menu == "1":
        print(winning)

        # Create Red & Blue team dataframe
        red_df = pd.DataFrame([np.NaN], columns=["win"])
        blue_df = pd.DataFrame([np.NaN], columns=["win"])

        sample_df1 = df.loc[0:1000]
        sample_df2 = df.loc[111300:112350]
        frames = [sample_df1, sample_df2]

        sample_df = pd.concat(frames)
        sample_df.reset_index(inplace=True)
        sample_df.drop(columns='index', inplace=True)

        lowcheck = low_outlier(winning, 'Pick rate')

        print(lowcheck)

        blueChamps = []
        redChamps = []

        evt1 = np.array([0, 0])
        evt2 = np.array([0, 0])
        evt3 = np.array([0, 0])
        evt4 = np.array([0, 0])
        evt5 = np.array([0, 0])
        evt6 = np.array([0, 0])
        evt7 = np.array([0, 0])
        evt8 = np.array([0, 0])
        evt9 = np.array([0, 0])
        evt10 = np.array([0, 0])
        evt11 = np.array([0, 0])
        evt12 = np.array([0, 0])

        checker = 1
        while checker == 1:

            if blueChamps == [] or redChamps == []:
                print("Blue Team Champions: ")
                while True:
                    champ_temp = input("Top: ")
                    tier_temp = input("Top's Tier: ")
                    valueCheck = value_checker(lowcheck, champ_temp, tier_temp)
                    if valueCheck == 2:
                        blueChamps.append(champ_temp)
                        break
                    elif valueCheck == 1:
                        print("No Data in your Tier. Sorry")
                    else:
                        print("Invalid Champion name.")
                while True:
                    champ_temp = input("Jungle: ")
                    tier_temp = input("Jungle's Tier: ")
                    valueCheck = value_checker(lowcheck, champ_temp, tier_temp)
                    if valueCheck == 2:
                        blueChamps.append(champ_temp)
                        break
                    elif valueCheck == 1:
                        print("No Data in your Tier. Sorry")
                    else:
                        print("Invalid Champion name.")
                while True:
                    champ_temp = input("Mid: ")
                    tier_temp = input("Mid's Tier: ")
                    valueCheck = value_checker(lowcheck, champ_temp, tier_temp)
                    if valueCheck == 2:
                        blueChamps.append(champ_temp)
                        break
                    elif valueCheck == 1:
                        print("No Data in your Tier. Sorry")
                    else:
                        print("Invalid Champion name.")
                while True:
                    champ_temp = input("Bottom: ")
                    tier_temp = input("Bottom's Tier: ")
                    valueCheck = value_checker(lowcheck, champ_temp, tier_temp)
                    if valueCheck == 2:
                        blueChamps.append(champ_temp)
                        break
                    elif valueCheck == 1:
                        print("No Data in your Tier. Sorry")
                    else:
                        print("Invalid Champion name.")
                while True:
                    champ_temp = input("Supporter: ")
                    tier_temp = input("Supporter's Tier: ")
                    valueCheck = value_checker(lowcheck, champ_temp, tier_temp)
                    if valueCheck == 2:
                        blueChamps.append(champ_temp)
                        break
                    elif valueCheck == 1:
                        print("No Data in your Tier. Sorry")
                    else:
                        print("Invalid Champion name.")

                print("Red Team Champions: ")
                while True:
                    champ_temp = input("Top: ")
                    tier_temp = input("Top's Tier: ")
                    valueCheck = value_checker(lowcheck, champ_temp, tier_temp)
                    if valueCheck == 2:
                        redChamps.append(champ_temp)
                        break
                    elif valueCheck == 1:
                        print("No Data in your Tier. Sorry")
                    else:
                        print("Invalid Champion name.")
                while True:
                    champ_temp = input("Jungle: ")
                    tier_temp = input("Jungle's Tier: ")
                    valueCheck = value_checker(lowcheck, champ_temp, tier_temp)
                    if valueCheck == 2:
                        redChamps.append(champ_temp)
                        break
                    elif valueCheck == 1:
                        print("No Data in your Tier. Sorry")
                    else:
                        print("Invalid Champion name.")
                while True:
                    champ_temp = input("Mid: ")
                    tier_temp = input("Mid's Tier: ")
                    valueCheck = value_checker(lowcheck, champ_temp, tier_temp)
                    if valueCheck == 2:
                        redChamps.append(champ_temp)
                        break
                    elif valueCheck == 1:
                        print("No Data in your Tier. Sorry")
                    else:
                        print("Invalid Champion name.")
                while True:
                    champ_temp = input("Bottom: ")
                    tier_temp = input("Bottom's Tier: ")
                    valueCheck = value_checker(lowcheck, champ_temp, tier_temp)
                    if valueCheck == 2:
                        redChamps.append(champ_temp)
                        break
                    elif valueCheck == 1:
                        print("No Data in your Tier. Sorry")
                    else:
                        print("Invalid Champion name.")
                while True:
                    champ_temp = input("Supporter: ")
                    tier_temp = input("Supporter's Tier: ")
                    valueCheck = value_checker(lowcheck, champ_temp, tier_temp)
                    if valueCheck == 2:
                        redChamps.append(champ_temp)
                        break
                    elif valueCheck == 1:
                        print("No Data in your Tier. Sorry")
                    else:
                        print("Invalid Champion name.")

            if blueChamps != [] and redChamps != []:
                print("Blue Team: ", blueChamps)
                print("Red Team; ", redChamps)

                while True:
                    print("\n\nEvent Selector")
                    print(
                        "1: FirstBlood / 2: FirstTower     / 3: FirstInhibitor / 4: FirstBaron   / 5: FirstDragon   / 6: FirstRiftHerald")
                    print(
                        "7: TowerKills / 8: InhibitorKills / 9: BaronKills     / 10: DragonKills / 11: Vilemawkills / 12: riftHeraldKills  ////  0: Exit")

                    i = input()
                    if i == '1':
                        if evt1[0] == 0 and evt1[1] == 0:
                            while True:
                                teamcheck = input("Which Team? (blue / red): ")
                                if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                    if teamcheck.lower() == 'blue':
                                        evt1[0] = 1

                                        # Predict
                                        blue_df["firstBlood"] = evt1[0]
                                        predict(blue_df, sample_df, tar)

                                    else:
                                        evt1[1] = 1

                                        # Predict
                                        red_df["firstBlood"] = evt1[1]
                                        predict(red_df, sample_df,tar)
                                    ##event 1 activate
                                    break;
                                else:
                                    print("Check the team")
                        else:
                            print("FirstBlood must use one time only.")
                    elif i == '2':
                        if evt2[0] == 0 and evt2[1] == 0:
                            while True:
                                teamcheck = input("Which Team? (blue / red): ")
                                if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                    if teamcheck.lower() == 'blue':
                                        evt2[0] = 1

                                        # Predict
                                        blue_df["firstTower"] = evt2[0]
                                        predict(blue_df,sample_df, tar)
                                    else:
                                        evt2[1] = 1

                                        # Predict
                                        red_df["firstTower"] = evt2[1]
                                        predict(red_df,sample_df, tar)
                                    ##event 2 activate
                                    break;
                                else:
                                    print("Check the team")
                        else:
                            print("FirstBlood must use one time only.")
                    elif i == '3':
                        if evt3[0] == 0 and evt3[1] == 0:
                            while True:
                                teamcheck = input("Which Team? (blue / red): ")
                                if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                    if teamcheck.lower() == 'blue':
                                        evt3[0] = 1

                                        # Predict
                                        blue_df["firstInhibitor"] = evt3[0]
                                        predict(blue_df,sample_df, tar)
                                    else:
                                        evt3[1] = 1

                                        # Predict
                                        red_df["firstInhibitor"] = evt3[1]
                                        predict(red_df,sample_df, tar)
                                    ##event 3 activate
                                    break;
                                else:
                                    print("Check the team")
                        else:
                            print("FirstBlood must use one time only.")
                    elif i == '4':
                        if evt4[0] == 0 and evt4[1] == 0:
                            while True:
                                teamcheck = input("Which Team? (blue / red): ")
                                if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                    if teamcheck.lower() == 'blue':
                                        evt4[0] = 1

                                        # Predict
                                        blue_df["firstBaron"] = evt4[0]
                                        predict(blue_df,sample_df, tar)
                                    else:
                                        evt4[1] = 1

                                        # Predict
                                        red_df["firstBaron"] = evt4[1]
                                        predict(red_df,sample_df, tar)
                                    ##event 4 activate
                                    break;
                                else:
                                    print("Check the team")
                        else:
                            print("FirstBlood must use one time only.")
                    elif i == '5':
                        if evt5[0] == 0 and evt5[1] == 0:
                            while True:
                                teamcheck = input("Which Team? (blue / red): ")
                                if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                    if teamcheck.lower() == 'blue':
                                        evt5[0] = 1

                                        # Predict
                                        blue_df["firstDragon"] = evt5[0]
                                        predict(blue_df,sample_df, tar)
                                    else:
                                        evt5[1] = 1

                                        # Predict
                                        red_df["firstDragon"] = evt5[1]
                                        predict(red_df,sample_df, tar)
                                    ##event 5 activate
                                    break;
                                else:
                                    print("Check the team")
                        else:
                            print("FirstBlood must use one time only.")
                    elif i == '6':
                        if evt6[0] == 0 and evt6[1] == 0:
                            while True:
                                teamcheck = input("Which Team? (blue / red): ")
                                if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                    if teamcheck.lower() == 'blue':
                                        evt6[0] = 1

                                        # Predict
                                        blue_df["firstRiftHerald"] = evt6[0]
                                        predict(blue_df,sample_df, tar)
                                    else:
                                        evt6[1] = 1

                                        # Predict
                                        red_df["firstRiftHerald"] = evt6[1]
                                        predict(red_df,sample_df, tar)
                                    ##event 6 activate
                                    break;
                                else:
                                    print("Check the team")
                        else:
                            print("FirstBlood must use one time only.")
                    elif i == '7':
                        while True:
                            teamcheck = input("Which Team? (blue / red): ")
                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                    if teamcheck.lower() == 'blue':
                                        evt7[0] = evt7[0] + 1

                                        # Predict
                                        blue_df["towerKills"] = evt7[0]
                                        predict(blue_df,sample_df, tar)
                                    else:
                                        evt7[1] = evt7[1] + 1

                                        # Predict
                                        red_df["towerKills"] = evt7[1]
                                        predict(red_df,sample_df, tar)
                                    break
                            else:
                                print("Check the team")


                    elif i == '8':
                        while True:
                            teamcheck = input("Which Team? (blue / red): ")
                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                    if teamcheck.lower() == 'blue':
                                        evt8[0] = evt8[0] + 1

                                        # Predict
                                        blue_df["inhibitorKills"] = evt8[0]
                                        predict(blue_df,sample_df, tar)
                                    else:
                                        evt8[1] = evt8[1] + 1

                                        # Predict
                                        red_df["inhibitorKills"] = evt8[1]
                                        predict(red_df,sample_df, tar)
                                    break;
                            else:
                                print("Check the team")

                    elif i == '9':
                        while True:
                            teamcheck = input("Which Team? (blue / red): ")
                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                    if teamcheck.lower() == 'blue':
                                        evt9[0] = evt9[0] + 1

                                        # Predict
                                        blue_df["baronKills"] = evt9[0]
                                        predict(blue_df,sample_df, tar)
                                    else:
                                        evt9[1] = evt9[1] + 1

                                        # Predict
                                        red_df["baronKills"] = evt9[1]
                                        predict(red_df,sample_df, tar)
                                    break
                            else:
                                print("Check the team")

                    elif i == '10':
                        while True:
                            teamcheck = input("Which Team? (blue / red): ")
                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                if teamcheck.lower() == 'blue':
                                    evt10[0] = evt10[0] + 1

                                    # Predict
                                    blue_df["dragonKills"] = evt10[0]
                                    predict(blue_df,sample_df, tar)
                                else:
                                    evt10[1] = evt10[1] + 1

                                    # Predict
                                    red_df["dragonKills"] = evt10[1]
                                    predict(red_df,sample_df, tar)
                                break

                            else:
                                print("Check the team")

                    elif i == '11':
                        while True:
                            teamcheck = input("Which Team? (blue / red): ")
                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                    if teamcheck.lower() == 'blue':
                                        evt11[0] = evt11[0] + 1

                                        # Predict
                                        blue_df["vilemawKills"] = evt11[0]
                                        predict(blue_df,sample_df, tar)
                                    else:
                                        evt11[1] = evt11[1] + 1

                                        # Predict
                                        red_df["vilemawKills"] = evt11[1]
                                        predict(red_df,sample_df, tar)
                                    break
                            else:
                                print("Check the team")

                    elif i == '12':
                        while True:
                            teamcheck = input("Which Team? (blue / red): ")
                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                    if teamcheck.lower() == 'blue':
                                        evt12[0] = evt12[0] + 1

                                        # Predict
                                        blue_df["riftHeraldKills"] = evt12[0]
                                        predict(blue_df,sample_df, tar)
                                    else:
                                        evt12[1] = evt12[1] + 1

                                        # Predict
                                        red_df["riftHeraldKills"] = evt12[0]
                                        predict(red_df,sample_df, tar)
                                    break
                            else:
                                print("Check the team")

                    elif i == '0':
                        checker = 0
                        break
                    else:
                        print("Invalid Event Select")
    # ########## Menu 2 ##########
    elif sel_menu == "2":
        # Extract 1000 sample data
        sample_df1 = df.loc[550:600]
        sample_df2 = df.loc[111300:111350]
        frames = [sample_df1, sample_df2]

        sample_df = pd.concat(frames)
        sample_df.reset_index(inplace=True)
        sample_df.drop(columns='index', inplace=True)
        print(sample_df)

        #######  Test  #######
        # Merge test data into rear of DataFrame
        # Target class value is 'NaN'
        k = 3

        print("\n< Test data> ")
        print(sample_df.iloc[-1, :])

        print("\nHyper Parmeter K = " + str(k))

        # Execute kNN-Algorithm
        sample_df = kNN(sample_df, tar, k)

        # Print result DataFrame
        print("\n< Data frame >")
        print(sample_df)
        print("\n< Prediction result  >")
        print(sample_df.iloc[-1, :])

        # # KFold
        # Set KFold k = 5
        kf = KFold(n_splits=5, shuffle=True)
        kcnt = 1
        k_range = range(2, 10)

        for train, test in kf.split(sample_df):
            train_df = sample_df.iloc[train, :]
            test_df = sample_df.iloc[test, :]

            train_df.reset_index(inplace=True)
            test_df.reset_index(inplace=True)
            train_df.drop(columns='index', inplace=True)
            test_df.drop(columns='index', inplace=True)

            accuracy_list = []
            print("\nK-fold "+str(kcnt)+" training. Wait please.")
            for k in k_range:
                cnt = 0
                for j in range(0, len(train_df)):
                    tmp = train_df.copy()
                    tmp2 = train_df.copy()
                    tmp.iloc[j, 0] = np.NaN

                    # Target class value is 'NaN'
                    tmp2.loc[len(train_df)] = np.array(tmp.iloc[j, 0:])
                    tmp2 = kNN(tmp2, tar, k)

                    if tmp2.iloc[len(train_df), 0] == tmp2.iloc[j, 0]:
                        cnt += 1

                accuracy_list.append(cnt / len(train_df))
            max_k = accuracy_list.index(max(accuracy_list)) + 3

            print("Training: Done")

            print("\nK-fold " + str(kcnt) + " validation")
            # Test with max k
            cnt2 = 0
            for i in range(0, len(test_df)):
                tmp = test_df.copy()
                tmp2 = test_df.copy()
                tmp.iloc[i, 0] = np.NaN

                # Target class value is 'NaN'
                tmp2.loc[len(train_df)] = np.array(tmp.iloc[i, 0:])
                tmp2 = kNN(tmp2, tar, max_k)

                if tmp2.iloc[len(test_df), 0] == tmp2.iloc[i, 0]:
                    cnt2 += 1
            accuracy = (cnt2 / len(test_df)) * 100
            print("Validation: Done")

            print("\nKFold: " + str(kcnt))
            print("Best hyperparmeter k = " + str(max_k))
            print("Validation accuracy: " + str(accuracy) + " %")
            kcnt += 1

            # Print Confusion Matrix
            tar_list = []
            for j in range(0, len(sample_df)):
                tmp = sample_df.copy()
                tmp2 = sample_df.copy()
                tmp.iloc[j, 0] = np.NaN

                # Target class value is 'NaN'
                tmp2.loc[len(sample_df)] = np.array(tmp.iloc[j, 0:])
                tmp2 = kNN(tmp2, tar, 5)
                print("epoch: " + str(j))
                tar_list.append(tmp2.iloc[len(sample_df), 0])


            tmp_sample_df = sample_df.copy()
            tmp_sample_df["Prediction"] = tar_list

            print("\n< Confusion Matrix >")
            print(metrics.confusion_matrix(sample_df['win'], tar_list), "\n")

            # Print Classification report
            print(metrics.classification_report(sample_df['win'], tar_list))
    # ########## Menu 3 ##########
    elif sel_menu == "3":
        print('\nEntire match_data : \n', df, '\n')
        all_column = ['firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald',
                      'towerKills', 'inhibitorKills', 'baronKills', 'dragonKills', 'vilemawKills', 'riftHeraldKills']
        x_column = ['towerKills', 'inhibitorKills', 'baronKills', 'dragonKills', 'vilemawKills', 'riftHeraldKills']
        y_column = 'win'

        df = df[all_column].astype('int')

        # All x_column data
        x_data = df.iloc[:, :]

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
    # ########## Menu 4 ##########
    elif sel_menu == "4":
        df = pd.read_csv('combined_data.csv')
        print('\nEntire match_data : \n', df, '\n')

        data_team = df.dropna(axis=0)
        data_team2 = data_team[list(data_team.columns)[2:]]

        data_team2 = data_team2[
            ['firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald',
             'towerKills', 'inhibitorKills', 'baronKills', 'dragonKills', 'vilemawKills', 'riftHeraldKills']].astype(
            'int')
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

        print('The number of data : ', (len(df)))
        print('The number of training : ', (len(y_train)), '(', round(len(y_train) * 100 / len(df), 2), '%)')
        print('The number of testing : ', (len(y_test)), '(', round(len(y_test) * 100 / len(df), 2), '%)')

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

        print('\n<Ensemble Learning with KNN Classifier>')
        print('Confusion matrix : \n', confusion_matrix(y_test, predicted))
        print(f'The number of testing error : ', (y_test != predicted).sum())
        print(f'Mean accuracy score: {accuracy:.3}')
        print('\n', classification_report(y_test, predicted))

    # ########## Menu 5 ##########
    elif sel_menu == "5":
        import data_explore
    # ########## Exit ##########
    elif sel_menu == "6":
        menu = False
        print("\nExit the program. Thank you!")
