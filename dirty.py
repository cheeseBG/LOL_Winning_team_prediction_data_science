"""
Title : Data science data defiling for practice
Date : 18 Jun 2020
Author : seokhyeonSong
"""

import pandas as pd
import random
import os

wl = pd.read_csv("WinningRate.csv")
lose = pd.read_csv("match_loser_data_version1.csv")
win = pd.read_csv("match_winner_data_version1.csv")
for i in range(len(wl)):
    p = random.randint(1,100)
    if p<6: #for 5 in 100 case
        k = random.randint(0,len(wl.columns.tolist())-1) # select which column to be defiled
        wl.iloc[i,k] = None #make data dirty
for i in range(len(win)):
    p = random.randint(1,100)
    if p<6:
        k=random.randint(0,len(win.columns.tolist())-1)
        win.iloc[i,k] = None
for i in range(len(lose)):
    p = random.randint(1,100)
    if p<6:
        k=random.randint(0,len(lose.columns.tolist())-1)
        lose.iloc[i,k] = None
wl.to_csv(os.getcwd()+"\\WinningRate_dirty.csv",mode='w',index=False) #save it as dirty version
win.to_csv(os.getcwd()+"\\match_winner_data_version1_dirty.csv",mode='w',index=False)
lose.to_csv(os.getcwd()+"\\match_loser_data_version1_dirty.csv",mode='w',index=False)