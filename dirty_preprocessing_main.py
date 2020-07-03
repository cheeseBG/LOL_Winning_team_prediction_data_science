import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings(action='ignore')


def low_outlier (df, col):
          mean = df["Pick rate"].mean()
          std = df["Pick rate"].std()
          for i in df.index:
                    val = df.at[i, col]
                    if val < mean - 3 * std:
                              df.at[i,col] = np.nan
          df = df.dropna()
          df = df.reset_index(drop=True)
          return df

def value_checker (df, champ, tier):
          df_tier = df[df['Rank'] == tier]
                       
          for i in df_tier.index: 
                    if champ == df.at[i, 'Name']:
                              return 2
          for i in df.index:
                    if champ == df.at[i, 'Name']:
                              return 1
          return 0


winning = pd.read_csv('WinningRate_dirty.csv', encoding = 'utf-8')

print(winning)


lowcheck = low_outlier (winning, 'Pick rate')

print(lowcheck)

blueChamps = []
redChamps = []

evt1 = 0
evt2 = 0
evt3 = 0
evt4 = 0
evt5 = 0
evt6 = 0
evt7 = 0
evt8 = 0
evt9 = 0
evt10 = 0
evt11 = 0
evt12 = 0
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
                              print("Event Selector")
                              print("1: FirstBlood / 2: FirstTower     / 3: FirstInhibitor / 4: FirstBaron   / 5: FirstDragon   / 6: FirstRiftHerald")
                              print("7: TowerKills / 8: InhibitorKills / 9: BaronKills     / 10: DragonKills / 11: Vilemawkills / 12: riftHeraldKills  ////  0: Exit")

                              i=input()
                              if i == '1':
                                        if evt1 == 0:
                                                  evt1 = 1
                                                  while True:
                                                            teamcheck = input("Which Team? (blue / red): ")
                                                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                                                      ##event 1 activate
                                                                      break;
                                                            else:
                                                                      print("Check the team")
                              elif i == '2':
                                        if evt2 == 0:
                                                  evt2 = 1
                                                  while True:
                                                            teamcheck = input("Which Team? (blue / red): ")
                                                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                                                      ##event 2 activate
                                                                      break;
                                                            else:
                                                                      print("Check the team")
                              elif i == '3':
                                        if evt3 == 0:
                                                  evt3 = 1
                                                  while True:
                                                            teamcheck = input("Which Team? (blue / red): ")
                                                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                                                      ##event 3 activate
                                                                      break;
                                                            else:
                                                                      print("Check the team")
                              elif i == '4':
                                        if evt4 == 0:
                                                  evt4 = 1
                                                  while True:
                                                            teamcheck = input("Which Team? (blue / red): ")
                                                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                                                      ##event 4 activate
                                                                      break;
                                                            else:
                                                                      print("Check the team")
                              elif i == '5':
                                        if evt5 == 0:
                                                  evt5 = 1
                                                  while True:
                                                            teamcheck = input("Which Team? (blue / red): ")
                                                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                                                      ##event 5 activate
                                                                      break;
                                                            else:
                                                                      print("Check the team")
                              elif i == '6':
                                        if evt6 == 0:
                                                  evt6 = 1
                                                  while True:
                                                            teamcheck = input("Which Team? (blue / red): ")
                                                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                                                      ##event 6 activate
                                                                      break;
                                                            else:
                                                                      print("Check the team")
                              elif i == '7':
                                        if evt7 == 0:
                                                  evt7 = 1
                                                  while True:
                                                            teamcheck = input("Which Team? (blue / red): ")
                                                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                                                      break;
                                                            else:
                                                                      print("Check the team")
                                                  while True:
                                                            scorecheck = input("How many scores?: ")
                                                            if scorecheck.isdigit():
                                                                      ##event 7 activate
                                                                      break;
                                                            else:
                                                                      print("Write digit")

                              elif i == '8':
                                        if evt8 == 0:
                                                  evt8 = 1
                                                  while True:
                                                            teamcheck = input("Which Team? (blue / red): ")
                                                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                                                      break;
                                                            else:
                                                                      print("Check the team")
                                                  while True:
                                                            scorecheck = input("How many scores?: ")
                                                            if scorecheck.isdigit():
                                                                      ##event 8 activate
                                                                      break;
                                                            else:
                                                                      print("Write digit")
                              elif i == '9':
                                        if evt9 == 0:
                                                  evt9 = 1
                                                  while True:
                                                            teamcheck = input("Which Team? (blue / red): ")
                                                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                                                      break;
                                                            else:
                                                                      print("Check the team")
                                                  while True:
                                                            scorecheck = input("How many scores?: ")
                                                            if scorecheck.isdigit():
                                                                      ##event 9 activate
                                                                      break;
                                                            else:
                                                                      print("Write digit")
                              elif i == '10':
                                        if evt10 == 0:
                                                  evt10 = 1
                                                  while True:
                                                            teamcheck = input("Which Team? (blue / red): ")
                                                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                                                      break;
                                                            else:
                                                                      print("Check the team")
                                                  while True:
                                                            scorecheck = input("How many scores?: ")
                                                            if scorecheck.isdigit():
                                                                      ##event 10 activate
                                                                      break;
                                                            else:
                                                                      print("Write digit")
                              elif i == '11':
                                        if evt11 == 0:
                                                  evt11 = 1
                                                  while True:
                                                            teamcheck = input("Which Team? (blue / red): ")
                                                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                                                      break;
                                                            else:
                                                                      print("Check the team")
                                                  while True:
                                                            scorecheck = input("How many scores?: ")
                                                            if scorecheck.isdigit():
                                                                      ##event 11 activate
                                                                      break;
                                                            else:
                                                                      print("Write digit")
                              elif i == '12':
                                        if evt12 == 0:
                                                  evt12 = 1
                                                  while True:
                                                            teamcheck = input("Which Team? (blue / red): ")
                                                            if teamcheck.lower() == 'blue' or teamcheck.lower() == 'red':
                                                                      break;
                                                            else:
                                                                      print("Check the team")
                                                  while True:
                                                            scorecheck = input("How many scores?: ")
                                                            if scorecheck.isdigit():
                                                                      ##event 12 activate
                                                                      break;
                                                            else:
                                                                      print("Write digit")
                              elif i == '0':
                                        checker = 0
                                        break
                              else:
                                        print("Invalid Event Select")
                              
  
                                        

                                  
          
                    
