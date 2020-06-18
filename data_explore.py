"""
Title : Data science data explore, explore winning rate & winning or lose team's data of League of Legends
Date : 18 Jun 2020
Author : seokhyeonSong
"""
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore",UserWarning)

"""
input : winning rate dataframe
output : none
description : show histogram of winning rate of all lanes & all ranks
"""
def histWin(wl):
    x = wl['Winning Rate'] # data extraction of winning rate
    hist = plt.hist(x, bins=100, range=(x.min(), x.max())) # histogram that has 100 x-bars of extracted data
    plt.title('Winning rate / all position / all rank' , size=18, fontweight='bold')
    plt.xlabel('Winning rate')
    plt.ylabel('Count')
    plt.show()
    histrank("Bronze", wl)
    histrank("Silver", wl)
    histrank("Gold", wl)
    histrank("Platinum", wl)
    histrank("Diamond+", wl)

"""
input : winning rate dataframe
output : none
description : show histogram of pick rate of all lanes & all ranks
"""
def histPick(wl):
    x = wl['Pick rate'] # data extraction of pick rate
    hist = plt.hist(x, bins=100, range=(x.min(), x.max())) # histogram that has 100 x-bars of extracted data
    plt.title('Pick rate / all position / all rank', size=18, fontweight='bold')
    plt.xlabel('Pick rate')
    plt.ylabel('Count')
    plt.show()
    histrankpick("Bronze", wl)
    histrankpick("Silver", wl)
    histrankpick("Gold", wl)
    histrankpick("Platinum", wl)
    histrankpick("Diamond+", wl)

"""
input : rank information, winning rate dataframe
output : none
description : show histogram of winning rate of given rank & all lanes
"""
def histrank(rank, data):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title("histogram of "+rank+"'s winning rate", size=18, fontweight='bold')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False) # to set xlabel of big plots
    ax.set_xlabel("Winning rate")
    top = fig.add_subplot(4, 4, 5) # set each position to not to be overlapped
    jungle = fig.add_subplot(4, 4, 7)
    mid = fig.add_subplot(4, 4, 10)
    bottom = fig.add_subplot(4, 4, 12)
    support = fig.add_subplot(4, 4, 13)
    alllane = fig.add_subplot(4, 4, 15)
    top.set_title(rank + "" + " / Top", size=9) # set each title of subplots
    jungle.set_title(rank + "" + " / Jungle", size=9)
    mid.set_title(rank + "" + " / Mid", size=9)
    bottom.set_title(rank + "" + " / Bottom", size=9)
    support.set_title(rank + "" + " / Supporter", size=9)
    alllane.set_title(rank + "" + " / All lanes", size=9)
    x = data[data["Rank"]==rank] # extract of given rank's dataframe
    top = top.hist(x[data["Lane"]=="Top"]["Winning Rate"],bins=20, range=(x[data["Lane"]=="Top"]['Winning Rate'].min(),x[data["Lane"]=="Top"]["Winning Rate"].max())) #set histogram of given rank's winning rate
    jungle = jungle.hist(x[data["Lane"] == "Jungle"]["Winning Rate"], bins=20, range=(
    x[data["Lane"] == "Jungle"]['Winning Rate'].min(), x[data["Lane"] == "Jungle"]["Winning Rate"].max()))
    mid = mid.hist(x[data["Lane"] == "Mid"]["Winning Rate"], bins=20, range=(
        x[data["Lane"] == "Mid"]['Winning Rate'].min(), x[data["Lane"] == "Mid"]["Winning Rate"].max()))
    bottom = bottom.hist(x[data["Lane"] == "Bottom"]["Winning Rate"], bins=20, range=(
        x[data["Lane"] == "Bottom"]['Winning Rate'].min(), x[data["Lane"] == "Bottom"]["Winning Rate"].max()))
    support = support.hist(x[data["Lane"] == "Supporter"]["Winning Rate"], bins=20, range=(
        x[data["Lane"] == "Supporter"]['Winning Rate'].min(), x[data["Lane"] == "Supporter"]["Winning Rate"].max()))
    alllane = alllane.hist(x["Winning Rate"], bins=20, range=(
        x['Winning Rate'].min(), x["Winning Rate"].max()))
    plt.show()

"""
input : rank information, winning rate dataframe
output : none
description : show histogram of pick rate of given rank & all lanes
"""
def histrankpick(rank, data):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title("histogram of " + rank + "'s pick rate" , size=18, fontweight='bold')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False) # to set xlabel of big plots
    ax.set_xlabel("pick rate")
    top = fig.add_subplot(4, 4, 5)# set each position to not to be overlapped
    jungle = fig.add_subplot(4, 4, 7)
    mid = fig.add_subplot(4, 4, 10)
    bottom = fig.add_subplot(4, 4, 12)
    support = fig.add_subplot(4, 4, 13)
    alllane = fig.add_subplot(4, 4, 15)
    top.set_title(rank + "" + " / Top" , size=9) # set each title of subplots
    jungle.set_title(rank + "" + " / Jungle", size=9)
    mid.set_title(rank + "" + " / Mid", size=9)
    bottom.set_title(rank + "" + " / Bottom", size=9)
    support.set_title(rank + "" + " / Supporter", size=9)
    alllane.set_title(rank + "" + " / All lanes", size=9)
    x = data[data["Rank"]==rank]
    top = top.hist(x[data["Lane"]=="Top"]["Pick rate"],bins=20, range=(x[data["Lane"]=="Top"]['Pick rate'].min(),x[data["Lane"]=="Top"]["Pick rate"].max())) #set histogram of given rank's pick rates
    jungle = jungle.hist(x[data["Lane"] == "Jungle"]["Pick rate"], bins=20, range=(
    x[data["Lane"] == "Jungle"]['Pick rate'].min(), x[data["Lane"] == "Jungle"]["Pick rate"].max()))
    mid = mid.hist(x[data["Lane"] == "Mid"]["Pick rate"], bins=20, range=(
        x[data["Lane"] == "Mid"]['Pick rate'].min(), x[data["Lane"] == "Mid"]["Pick rate"].max()))
    bottom = bottom.hist(x[data["Lane"] == "Bottom"]["Pick rate"], bins=20, range=(
        x[data["Lane"] == "Bottom"]['Pick rate'].min(), x[data["Lane"] == "Bottom"]["Pick rate"].max()))
    support = support.hist(x[data["Lane"] == "Supporter"]["Pick rate"], bins=20, range=(
        x[data["Lane"] == "Supporter"]['Pick rate'].min(), x[data["Lane"] == "Supporter"]["Pick rate"].max()))
    alllane = alllane.hist(x["Pick rate"], bins=20, range=(
        x['Pick rate'].min(), x["Pick rate"].max()))
    plt.show()

"""
input : winning rate dataframe
output : none
description : shows basic data exploration of winning rate dataframe
"""
def WRbasic(wl):
    print("Winning_Rate.csv's counts : " + str(len(wl)))
    print("columns : ")
    print(wl.columns.tolist())
    print("There are "+str(wl.isna().sum().sum())+" dirty data")
    print("We will just drop na data not at algorithm file")

"""
input : winning rate dataframe
output : none
description : shows rank data exploration of winning rate dataframe
"""
def rankdata(wl):
    print("\nRank data : ")
    print(wl['Rank'].drop_duplicates().tolist())
    print("Bronze count : " + str(len(wl[wl['Rank'] == 'Bronze'])))
    print("Silver count : " + str(len(wl[wl['Rank'] == 'Silver'])))
    print("Gold count : " + str(len(wl[wl['Rank'] == 'Gold'])))
    print("Platinum count : " + str(len(wl[wl['Rank'] == 'Platinum'])))
    print("Diamond+ count : " + str(len(wl[wl['Rank'] == 'Diamond+'])))

"""
input : winning rate dataframe
output : none
description : shows lane data exploration of winning rate dataframe
"""
def lanedata(wl):
    print("\nLane data : ")
    print(wl['Lane'].drop_duplicates().tolist())
    print("Top count : " + str(len(wl[wl['Lane'] == 'Top'])))
    print("Jungle count : " + str(len(wl[wl['Lane'] == 'Jungle'])))
    print("Mid count : " + str(len(wl[wl['Lane'] == 'Mid'])))
    print("Bottom count : " + str(len(wl[wl['Lane'] == 'Bottom'])))
    print("Supporter count : " + str(len(wl[wl['Lane'] == 'Supporter'])))

"""
input : winning rate dataframe
output : none
description : shows name data exploration of winning rate dataframe
"""
def namedata(wl):
    print("\nName data : ")
    print(sorted(wl['Name'].drop_duplicates().tolist()))
    print("total count : " + str(len(wl['Name'].drop_duplicates().tolist())))

"""
input : winning rate dataframe
output : none
description : shows winning rate data exploration of winning rate dataframe, boxplot of winning rates & histogram of winning rates 
"""
def winningratedata(wl):
    print("\nWinning rate data :")
    fig2, ax2 = plt.subplots()
    ax2.boxplot([wl['Winning Rate']], sym="b*") # set boxplot of winning rate
    plt.title("Box plot of Winning rate", size=18, fontweight='bold')
    plt.xticks([1], ['Winning rate'])
    plt.show()
    histWin(wl)

"""
input : winning rate dataframe
output : none
description : shows pick rate data exploration of winning rate dataframe , boxplot of pick rates & histogram of pick rates
"""
def pickratedata(wl):
    print("\nPick rate data :")
    fig2, ax2 = plt.subplots()
    ax2.boxplot([wl['Pick rate']], sym="b*") # set boxplot of pick rate
    plt.title("Box plot of Pick rate", size=18, fontweight='bold')
    plt.xticks([1], ['Pick rate'])
    plt.show()
    histPick(wl)

"""
input : winning team, lose team, total dataframe
output : none
description : shows each team's teamID data exploration
"""
def teamIDdata(windata,losedata,sumdata):
    print("\nWinning team's teamId data :")
    print(sorted(windata['teamId'].drop_duplicates().tolist()))
    print("Lose team's teamId data :")
    print(sorted(losedata['teamId'].drop_duplicates().tolist()))
    print("Total team's teamId data :")
    print(sorted(sumdata['teamId'].drop_duplicates().tolist()))
    print("Winning team's teamID==100 counts : " + str(len(windata[windata['teamId'] == 100])))
    print("Winning team's teamID==200 counts : " + str(len(windata[windata['teamId'] == 200])))
    print("Lose team's teamID==100 counts : " + str(len(losedata[losedata['teamId'] == 100])))
    print("Lose team's teamID==200 counts : " + str(len(losedata[losedata['teamId'] == 200])))
    print("Total team's teamID==100 counts : " + str(len(sumdata[sumdata['teamId'] == 100])))
    print("Total team's teamID==200 counts : " + str(len(sumdata[sumdata['teamId'] == 200])))

"""
input : winning team, lose team, total dataframe
output : none
description : shows each team's firstBlood data exploration , confusion matrix of firstblood and game wins
"""
def FBData(windata,losedata,sumdata):
    print("\nWinning team's firstBlood data :")
    print(sorted(windata['firstBlood'].drop_duplicates().tolist()))
    print("Lose team's firstBlood data :")
    print(sorted(losedata['firstBlood'].drop_duplicates().tolist()))
    print("Total team's firstBlood data :")
    print(sorted(sumdata['firstBlood'].drop_duplicates().tolist()))
    print("Winning team's firstblood==true counts : " + str(len(windata[windata['firstBlood'] == True])))
    print("Lose team's firstblood==true counts : " + str(len(losedata[losedata['firstBlood'] == True])))
    print("Total team's firstblood==true counts : " + str(len(sumdata[sumdata['firstBlood'] == True])))
    print("Winning team's firstblood==false counts : " + str(len(windata[windata['firstBlood'] == False])))
    print("Lose team's firstblood==false counts : " + str(len(losedata[losedata['firstBlood'] == False])))
    print("Total team's firstblood==false counts : " + str(len(sumdata[sumdata['firstBlood'] == False])))
    print("Winning team's firstblood==True percentage : " +str(len(windata[windata['firstBlood']==True])/len(windata)*100))
    print("Lose team's firstblood==False percentage : " + str(len(losedata[losedata['firstBlood']==False])/len(losedata)*100))
    print("Total team's firstblood==True goes game win percentage " + str(len(sumdata[sumdata['firstBlood']==True][sumdata['win']==True])*100/len(sumdata[sumdata['firstBlood']==True])))
    print("Total team's firstblood==False goes game lose percentage " + str(
        len(sumdata[sumdata['firstBlood'] == False][sumdata['win'] == False])*100 / len(
            sumdata[sumdata['firstBlood'] == False])))
    print("Total team's firstblood and game win or lose confusion matrix")
    ax = plt.axes()
    ax.set_title("Confusion matrix of first blood", size=18, fontweight='bold')
    CM = pd.crosstab(sumdata['firstBlood'],sumdata['win'],rownames=['firstBlood'],colnames=['win']) # get confusion matrix of firstblood & game win
    sn.heatmap(CM,annot=True, fmt='g') #set fmt='g' to show all numbers instead of e
    plt.show()

"""
input : winning team, lose team, total dataframe
output : none
description : shows each team's firstTower destroy data exploration , confusion matrix of firstTower destroy and game wins
"""
def FTData(windata,losedata,sumdata):
    print("\nWinning team's firstTower data :")
    print(sorted(windata['firstTower'].drop_duplicates().tolist()))
    print("Lose team's firstTower data :")
    print(sorted(losedata['firstTower'].drop_duplicates().tolist()))
    print("Total team's firstTower data :")
    print(sorted(sumdata['firstTower'].drop_duplicates().tolist()))
    print("Winning team's firstTower==true counts : " + str(len(windata[windata['firstTower'] == True])))
    print("Lose team's firstTower==true counts : " + str(len(losedata[losedata['firstTower'] == True])))
    print("Total team's firstTower==true counts : " + str(len(sumdata[sumdata['firstTower'] == True])))
    print("Winning team's firstTower==false counts : " + str(len(windata[windata['firstTower'] == False])))
    print("Lose team's firstTower==false counts : " + str(len(losedata[losedata['firstTower'] == False])))
    print("Total team's firstTower==false counts : " + str(len(sumdata[sumdata['firstTower'] == False])))
    print("Winning team's firstTower==True percentage : " + str(
        len(windata[windata['firstTower'] == True])*100 / len(windata)))
    print("Lose team's firstTower==False percentage : " + str(
        len(losedata[losedata['firstTower'] == False])*100 / len(losedata)))
    print("Total team's firstTower==True goes game win percentage " + str(
        len(sumdata[sumdata['firstTower'] == True][sumdata['win'] == True])*100 / len(
            sumdata[sumdata['firstTower'] == True])))
    print("Total team's firstTower==False goes game lose percentage " + str(
        len(sumdata[sumdata['firstTower'] == False][sumdata['win'] == False])*100 / len(
            sumdata[sumdata['firstTower'] == False])))
    print("Total team's firstTower and game win or lose confusion matrix")
    ax = plt.axes()
    ax.set_title("Confusion matrix of first tower destroy", size=18, fontweight='bold')
    CM = pd.crosstab(sumdata['firstTower'], sumdata['win'], rownames=['firstTower'], colnames=['win']) # get confusion matrix of first tower destory & game win
    sn.heatmap(CM, annot=True, fmt='g')
    plt.show()

"""
input : winning team, lose team, total dataframe
output : none
description : shows each team's firstInhibitor destroy data exploration , confusion matrix of firstInhibitor destory and game wins
"""
def FIData(windata,losedata,sumdata):
    print("\nWinning team's firstInhibitor data :")
    print(sorted(windata['firstInhibitor'].drop_duplicates().tolist()))
    print("Lose team's firstInhibitor data :")
    print(sorted(losedata['firstInhibitor'].drop_duplicates().tolist()))
    print("Total team's firstInhibitor data :")
    print(sorted(sumdata['firstInhibitor'].drop_duplicates().tolist()))
    print("Winning team's firstInhibitor==true counts : " + str(len(windata[windata['firstInhibitor'] == True])))
    print("Lose team's firstInhibitor==true counts : " + str(len(losedata[losedata['firstInhibitor'] == True])))
    print("Total team's firstInhibitor==true counts : " + str(len(sumdata[sumdata['firstInhibitor'] == True])))
    print("Winning team's firstInhibitor==false counts : " + str(len(windata[windata['firstInhibitor'] == False])))
    print("Lose team's firstInhibitor==false counts : " + str(len(losedata[losedata['firstInhibitor'] == False])))
    print("Total team's firstInhibitor==false counts : " + str(len(sumdata[sumdata['firstInhibitor'] == False])))
    print("Winning team's firstInhibitor==True percentage : " + str(
        len(windata[windata['firstInhibitor'] == True])*100 / len(windata)))
    print("Lose team's firstInhibitor==False percentage : " + str(
        len(losedata[losedata['firstInhibitor'] == False])*100 / len(losedata)))
    print("Total team's firstInhibitor==True goes game win percentage " + str(
        len(sumdata[sumdata['firstInhibitor'] == True][sumdata['win'] == True])*100 / len(
            sumdata[sumdata['firstInhibitor'] == True])))
    print("Total team's firstInhibitor==False goes game lose percentage " + str(
        len(sumdata[sumdata['firstInhibitor'] == False][sumdata['win'] == False])*100 / len(
            sumdata[sumdata['firstInhibitor'] == False])))
    print("Total team's firstInhibitor and game win or lose confusion matrix")
    ax = plt.axes()
    ax.set_title("Confusion matrix of first inhibitor destroy", size=16, fontweight='bold')
    CM = pd.crosstab(sumdata['firstInhibitor'], sumdata['win'], rownames=['firstInhibitor'], colnames=['win']) # get confusion matrix of first inhibitor destroy & game win
    sn.heatmap(CM, annot=True, fmt='g')
    plt.show()

"""
input : winning team, lose team, total dataframe
output : none
description : shows each team's firstBaron kill data exploration , confusion matrix of firstBaron kill and game wins
"""
def FBrData(windata,losedata,sumdata):
    print("\nWinning team's firstBaron data :")
    print(sorted(windata['firstBaron'].drop_duplicates().tolist()))
    print("Lose team's firstBaron data :")
    print(sorted(losedata['firstBaron'].drop_duplicates().tolist()))
    print("Total team's firstBaron data :")
    print(sorted(sumdata['firstBaron'].drop_duplicates().tolist()))
    print("Winning team's firstBaron==true counts : " + str(len(windata[windata['firstBaron'] == True])))
    print("Lose team's firstBaron==true counts : " + str(len(losedata[losedata['firstBaron'] == True])))
    print("Total team's firstBaron==true counts : " + str(len(sumdata[sumdata['firstBaron'] == True])))
    print("Winning team's firstBaron==false counts : " + str(len(windata[windata['firstBaron'] == False])))
    print("Lose team's firstBaron==false counts : " + str(len(losedata[losedata['firstBaron'] == False])))
    print("Total team's firstBaron==false counts : " + str(len(sumdata[sumdata['firstBaron'] == False])))
    print("Winning team's firstBaron==True percentage : " + str(
        len(windata[windata['firstBaron'] == True])*100 / len(windata)))
    print("Lose team's firstBaron==False percentage : " + str(
        len(losedata[losedata['firstBaron'] == False])*100 / len(losedata)))
    print("Total team's firstBaron==True goes game win percentage " + str(
        len(sumdata[sumdata['firstBaron'] == True][sumdata['win'] == True])*100 / len(
            sumdata[sumdata['firstBaron'] == True])))
    print("Total team's firstBaron==False goes game lose percentage " + str(
        len(sumdata[sumdata['firstBaron'] == False][sumdata['win'] == False])*100 / len(
            sumdata[sumdata['firstBaron'] == False])))
    print("Total team's firstBaron and game win or lose confusion matrix")
    ax = plt.axes()
    ax.set_title("Confusion matrix of first baron kill", size=18, fontweight='bold')
    CM = pd.crosstab(sumdata['firstBaron'], sumdata['win'], rownames=['firstBaron'], colnames=['win']) # get confusion matrix of first baron kill & game win
    sn.heatmap(CM, annot=True, fmt='g')
    plt.show()

"""
input : winning team, lose team, total dataframe
output : none
description : shows each team's firstDragon kill data exploration , confusion matrix of firstDragon kill and game wins
"""
def FDData(windata,losedata,sumdata):
    print("\nWinning team's firstDragon data :")
    print(sorted(windata['firstDragon'].drop_duplicates().tolist()))
    print("Lose team's firstDragon data :")
    print(sorted(losedata['firstDragon'].drop_duplicates().tolist()))
    print("Total team's firstDragon data :")
    print(sorted(sumdata['firstDragon'].drop_duplicates().tolist()))
    print("Winning team's firstDragon==true counts : " + str(len(windata[windata['firstDragon'] == True])))
    print("Lose team's firstDragon==true counts : " + str(len(losedata[losedata['firstDragon'] == True])))
    print("Total team's firstDragon==true counts : " + str(len(sumdata[sumdata['firstDragon'] == True])))
    print("Winning team's firstDragon==false counts : " + str(len(windata[windata['firstDragon'] == False])))
    print("Lose team's firstDragon==false counts : " + str(len(losedata[losedata['firstDragon'] == False])))
    print("Total team's firstDragon==false counts : " + str(len(sumdata[sumdata['firstDragon'] == False])))
    print("Winning team's firstDragon==True percentage : " + str(
        len(windata[windata['firstDragon'] == True])*100 / len(windata)))
    print("Lose team's firstDragon==False percentage : " + str(
        len(losedata[losedata['firstDragon'] == False])*100 / len(losedata)))
    print("Total team's firstDragon==True goes game win percentage " + str(
        len(sumdata[sumdata['firstDragon'] == True][sumdata['win'] == True])*100 / len(
            sumdata[sumdata['firstDragon'] == True])))
    print("Total team's firstDragon==False goes game lose percentage " + str(
        len(sumdata[sumdata['firstDragon'] == False][sumdata['win'] == False])*100 / len(
            sumdata[sumdata['firstDragon'] == False])))
    print("Total team's firstDragon and game win or lose confusion matrix")
    ax = plt.axes()
    ax.set_title("Confusion matrix of first dragon kill", size=18, fontweight='bold')
    CM = pd.crosstab(sumdata['firstDragon'], sumdata['win'], rownames=['firstDragon'], colnames=['win']) # get confusion matrix of first dragon kill & game win
    sn.heatmap(CM, annot=True, fmt='g')
    plt.show()

"""
input : winning team, lose team, total dataframe
output : none
description : shows each team's firstRiftherald kill data exploration , confusion matrix of firstRiftherald kill and game wins
"""
def FRfData(windata,losedata,sumdata):
    print("\nWinning team's firstRiftHerald data :")
    print(sorted(windata['firstRiftHerald'].drop_duplicates().tolist()))
    print("Lose team's firstRiftHerald data :")
    print(sorted(losedata['firstRiftHerald'].drop_duplicates().tolist()))
    print("Total team's firstRiftHerald data :")
    print(sorted(sumdata['firstRiftHerald'].drop_duplicates().tolist()))
    print("Winning team's firstRiftHerald==true counts : " + str(len(windata[windata['firstRiftHerald'] == True])))
    print("Lose team's firstRiftHerald==true counts : " + str(len(losedata[losedata['firstRiftHerald'] == True])))
    print("Total team's firstRiftHerald==true counts : " + str(len(sumdata[sumdata['firstRiftHerald'] == True])))
    print("Winning team's firstRiftHerald==false counts : " + str(len(windata[windata['firstRiftHerald'] == False])))
    print("Lose team's firstRiftHerald==false counts : " + str(len(losedata[losedata['firstRiftHerald'] == False])))
    print("Total team's firstRiftHerald==false counts : " + str(len(sumdata[sumdata['firstRiftHerald'] == False])))
    print("Winning team's firstRiftHerald==True percentage : " + str(
        len(windata[windata['firstRiftHerald'] == True])*100 / len(windata)))
    print("Lose team's firstRiftHerald==False percentage : " + str(
        len(losedata[losedata['firstRiftHerald'] == False])*100 / len(losedata)))
    print("Total team's firstRiftHerald==True goes game win percentage " + str(
        len(sumdata[sumdata['firstRiftHerald'] == True][sumdata['win'] == True])*100 / len(
            sumdata[sumdata['firstRiftHerald'] == True])))
    print("Total team's firstRiftHerald==False goes game lose percentage " + str(
        len(sumdata[sumdata['firstRiftHerald'] == False][sumdata['win'] == False])*100 / len(
            sumdata[sumdata['firstRiftHerald'] == False])))
    print("Total team's firstRiftHerald and game win or lose confusion matrix")
    ax = plt.axes()
    ax.set_title("Confusion matrix of first riftherald kill", size=18, fontweight='bold')
    CM = pd.crosstab(sumdata['firstRiftHerald'], sumdata['win'], rownames=['firstRiftHerald'], colnames=['win']) # get confusion matrix of first riftherald kill & game win
    sn.heatmap(CM, annot=True, fmt='g')
    plt.show()

"""
input : winning team, lose team, total dataframe
output : none
description : shows each team's tower kill # data exploration , histogram & boxplot of tower kill #
"""
def TKData(windata,losedata,sumdata):
    print("\nWinning team's towerKills data : ")
    print(sorted(windata['towerKills'].drop_duplicates().tolist()))
    print("Lose team's towerKills data :")
    print(sorted(losedata['towerKills'].drop_duplicates().tolist()))
    print("Total team's towerKills data :")
    print(sorted(sorted(sumdata['towerKills'].drop_duplicates().tolist())))
    winTK = windata["towerKills"]
    loseTK = losedata["towerKills"]
    totalTK = sumdata["towerKills"]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False) # to set xlabel of big plots
    ax.set_xlabel("tower kills")
    ax.set_title("histogram of tower kills", size=18, fontweight='bold')
    win = fig.add_subplot(4, 2, 3)
    lose = fig.add_subplot(4, 2, 6)
    total = fig.add_subplot(4, 2, 7)
    win.set_title("Winning teams")
    lose.set_title("Lose teams")
    total.set_title("Total teams")
    win = win.hist(winTK, bins=max(windata['towerKills'].drop_duplicates().tolist()))
    lose = lose.hist(loseTK, bins=max(losedata['towerKills'].drop_duplicates().tolist()))
    total = total.hist(totalTK, bins=max(sumdata['towerKills'].drop_duplicates().tolist()))
    plt.show()
    fig2, ax2 =plt.subplots()
    ax2.boxplot([winTK,loseTK,totalTK], sym="b*") # get boxplots of winning team, lose team, total team's tower kill
    plt.title("Box plots of tower kills", size=18, fontweight='bold')
    plt.xticks([1,2,3],['Winning team', 'Lose team', 'total team'])
    plt.show()

"""
input : winning team, lose team, total dataframe
output : none
description : shows each team's inhibitor kill # data exploration , histogram & boxplot of inhibitor kill #
"""
def IKData(windata,losedata,sumdata):
    print("\nWinning team's inhibitorKills data : ")
    print(sorted(windata['inhibitorKills'].drop_duplicates().tolist()))
    print("Lose team's inhibitorKills data :")
    print(sorted(losedata['inhibitorKills'].drop_duplicates().tolist()))
    print("Total team's inhibitorKills data :")
    print(sorted(sumdata['inhibitorKills'].drop_duplicates().tolist()))
    winTK = windata["inhibitorKills"]
    loseTK = losedata["inhibitorKills"]
    totalTK = sumdata["inhibitorKills"]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("inhibitor kills")
    ax.set_title("histogram of inhibitor kills", size=16, fontweight='bold')
    win = fig.add_subplot(4, 2, 3)
    lose = fig.add_subplot(4, 2, 6)
    total = fig.add_subplot(4, 2, 7)
    win.set_title("Winning teams")
    lose.set_title("Lose teams")
    total.set_title("Total teams")
    win = win.hist(winTK, bins=max(windata['inhibitorKills'].drop_duplicates().tolist()))
    lose = lose.hist(loseTK, bins=max(losedata['inhibitorKills'].drop_duplicates().tolist()))
    total = total.hist(totalTK, bins=max(sumdata['inhibitorKills'].drop_duplicates().tolist()))
    plt.show()
    fig2, ax2 =plt.subplots()
    ax2.boxplot([winTK,loseTK,totalTK], sym="b*") # get boxplots of winning team, lose team, total team's inhibitor destroy
    plt.title("Box plots of inhibitor kills", size=18, fontweight='bold')
    plt.xticks([1,2,3],['Winning team', 'Lose team', 'total team'])
    plt.show()

"""
input : winning team, lose team, total dataframe
output : none
description : shows each team's baron kill # data exploration , histogram & boxplot of baron kill #
"""
def BKData(windata,losedata,sumdata):
    print("\nWinning team's baronKills data : ")
    print(sorted(windata['baronKills'].drop_duplicates().tolist()))
    print("Lose team's baronKills data :")
    print(sorted(losedata['baronKills'].drop_duplicates().tolist()))
    print("Total team's baronKills data :")
    print(sorted(sumdata['baronKills'].drop_duplicates().tolist()))
    winTK = windata["baronKills"]
    loseTK = losedata["baronKills"]
    totalTK = sumdata["baronKills"]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("baron kills")
    ax.set_title("histogram of baron kills", size=18, fontweight='bold')
    win = fig.add_subplot(4, 2, 3)
    lose = fig.add_subplot(4, 2, 6)
    total = fig.add_subplot(4, 2, 7)
    win.set_title("Winning teams")
    lose.set_title("Lose teams")
    total.set_title("Total teams")
    win = win.hist(winTK, bins=max(windata['baronKills'].drop_duplicates().tolist()))
    lose = lose.hist(loseTK, bins=max(losedata['baronKills'].drop_duplicates().tolist()))
    total = total.hist(totalTK, bins=max(sumdata['baronKills'].drop_duplicates().tolist()))
    plt.show()
    fig2, ax2 =plt.subplots()
    ax2.boxplot([winTK,loseTK,totalTK], sym="b*") # get boxplots of winning team, lose team, total team's baron kill
    plt.title("Box plots of baron kills", size=18, fontweight='bold')
    plt.xticks([1,2,3],['Winning team', 'Lose team', 'total team'])
    plt.show()

"""
input : winning team, lose team, total dataframe
output : none
description : shows each team's dragon kill # data exploration , histogram & boxplot of dragon kill #
"""
def DKData(windata,losedata,sumdata):
    print("\nWinning team's dragonKills data : ")
    print(sorted(windata['dragonKills'].drop_duplicates().tolist()))
    print("Lose team's dragonKills data :")
    print(sorted(losedata['dragonKills'].drop_duplicates().tolist()))
    print("Total team's dragonKills data :")
    print(sorted(sumdata['dragonKills'].drop_duplicates().tolist()))
    winTK = windata["dragonKills"]
    loseTK = losedata["dragonKills"]
    totalTK = sumdata["dragonKills"]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("dragon kills")
    ax.set_title("histogram of dragon kills", size=18, fontweight='bold')
    win = fig.add_subplot(4, 2, 3)
    lose = fig.add_subplot(4, 2, 6)
    total = fig.add_subplot(4, 2, 7)
    win.set_title("Winning teams")
    lose.set_title("Lose teams")
    total.set_title("Total teams")
    win = win.hist(winTK, bins=max(windata['dragonKills'].drop_duplicates().tolist()))
    lose = lose.hist(loseTK, bins=max(losedata['dragonKills'].drop_duplicates().tolist()))
    total = total.hist(totalTK, bins=max(sumdata['dragonKills'].drop_duplicates().tolist()))
    plt.show()
    fig2, ax2 =plt.subplots()
    ax2.boxplot([winTK,loseTK,totalTK], sym="b*")
    plt.title("Box plots of dragon kills", size=18, fontweight='bold') # get boxplots of winning team, lose team, total team's dragon kill
    plt.xticks([1,2,3],['Winning team', 'Lose team', 'total team'])
    plt.show()

"""
input : winning team, lose team, total dataframe
output : none
description : shows each team's riftheralds kill # data exploration , histogram & boxplot of riftheralds kill #
"""
def RKData(windata,losedata,sumdata):
    print("\nWinning team's riftHeraldKills data : ")
    print(sorted(windata['riftHeraldKills'].drop_duplicates().tolist()))
    print("Lose team's riftHeraldKills data :")
    print(sorted(losedata['riftHeraldKills'].drop_duplicates().tolist()))
    print("Total team's riftHeraldKills data :")
    print(sorted(sumdata['riftHeraldKills'].drop_duplicates().tolist()))
    winTK = windata["riftHeraldKills"]
    loseTK = losedata["riftHeraldKills"]
    totalTK = sumdata["riftHeraldKills"]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_xlabel("riftHerald kills")
    ax.set_title("histogram of riftHerald kills", size=18, fontweight='bold')
    win = fig.add_subplot(4, 2, 3)
    lose = fig.add_subplot(4, 2, 6)
    total = fig.add_subplot(4, 2, 7)
    win.set_title("Winning teams")
    lose.set_title("Lose teams")
    total.set_title("Total teams")
    win = win.hist(winTK, bins=max(windata['riftHeraldKills'].drop_duplicates().tolist()))
    lose = lose.hist(loseTK, bins=max(losedata['riftHeraldKills'].drop_duplicates().tolist()))
    total = total.hist(totalTK, bins=max(sumdata['riftHeraldKills'].drop_duplicates().tolist()))
    plt.show()
    fig2, ax2 =plt.subplots()
    ax2.boxplot([winTK,loseTK,totalTK], sym="b*") # get boxplots of winning team, lose team, total team's riftherald kill
    plt.title("Box plots of riftHerald kills", size=18, fontweight='bold')
    plt.xticks([1,2,3],['Winning team', 'Lose team', 'total team'])
    plt.show()

"""
input : winning rate dataframe
output : none
description : execute exploration of winning rate dataframe
"""
def WRD(wl):
    WRbasic(wl)
    wl = wl.dropna()
    rankdata(wl)
    lanedata(wl)
    namedata(wl)
    winningratedata(wl)
    pickratedata(wl)

"""
input : winning team, lose team, total dataframe
output : none
description : shows each team's basic data exploration
"""
def WLTbasic(windata,losedata):
    print("\nWinning team data count : " + str(len(windata)))
    print("lose team data count : " + str(len(losedata)))
    print("Winning team columns")
    print(windata.columns.tolist())
    print("Lose team columns")
    print(losedata.columns.tolist())
    print("We won't use 1. Unnamed column 2. vilemawKills, 3. domonionVictoryScore 4. bans 5. gameId")
    print("because they are garbage data")
    print("There are "+str(windata.isna().sum().sum())+" dirty data at winning team data")
    print("There are "+str(losedata.isna().sum().sum())+" dirty data at lose team data")
    print("At here we will just drop na data, not at algorithm files")

"""
input : winning team, lose team, total dataframe
output : none
description : execute exploration of winning, lose, total team's data
"""
def WLTD(windata,losedata):
    WLTbasic(windata,losedata)
    windata = windata.dropna()
    losedata = losedata.dropna()
    windata = windata.drop([windata.columns[0], "vilemawKills", "dominionVictoryScore", "bans", "gameId"], axis=1)
    losedata = losedata.drop([losedata.columns[0], "vilemawKills", "dominionVictoryScore", "bans", "gameId"], axis=1) # at WLTbasic() we show raw data but we will use dataframe that arranged out with our needs so arrange data
    windata['win'] = windata['win'].apply(lambda x: True) # to use confusion matrix we set win or lose values same as other confusion matrix's element
    losedata['win'] = losedata['win'].apply(lambda x: False)
    sumdata = pd.merge(windata, losedata, how='outer')
    FBData(windata,losedata,sumdata)
    FTData(windata,losedata,sumdata)
    FIData(windata,losedata,sumdata)
    FBrData(windata,losedata,sumdata)
    FDData(windata,losedata,sumdata)
    FRfData(windata,losedata,sumdata)
    TKData(windata,losedata,sumdata)
    IKData(windata,losedata,sumdata)
    BKData(windata,losedata,sumdata)
    DKData(windata,losedata,sumdata)
    RKData(windata,losedata,sumdata)

windata = pd.read_csv("match_winner_data_version1_dirty.csv")
losedata =pd.read_csv("match_loser_data_version1_dirty.csv")
wl = pd.read_csv("WinningRate_dirty.csv")
i = 1
while True :
    print("Data science data explore")
    print("1: Winning rate explore   2: Winning & lose team's data explore   3: All of them   0: quit ")
    i=input()
    if i=='1':
        WRD(wl)
    elif i=='2':
        WLTD(windata,losedata)
    elif i=='3':
        WRD(wl)
        WLTD(windata,losedata)
    elif i=='0':
        print("Exit data explore")
        break
    else :
        print("Wrong input")
