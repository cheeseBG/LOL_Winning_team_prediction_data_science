import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings(action='ignore')


def low_outlier (df, col):
          for i in df.index:
                    val = df.at[i, col]
                    if val < 0.5:
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
checker = 1
while checker == 1:

          if blueChamps == [] or redChamps == []:
                    print("Blue Team Champions: ")
                    while True:
                              champ_temp = input("Top: ")
                              tier_temp = input("Top's Tier: ")
                              valueCheck = value_checker(winning, champ_temp, tier_temp)
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
                              valueCheck = value_checker(winning, champ_temp, tier_temp)
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
                              valueCheck = value_checker(winning, champ_temp, tier_temp)
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
                              valueCheck = value_checker(winning, champ_temp, tier_temp)
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
                              valueCheck = value_checker(winning, champ_temp, tier_temp)
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
                              valueCheck = value_checker(winning, champ_temp, tier_temp)
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
                              valueCheck = value_checker(winning, champ_temp, tier_temp)
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
                              valueCheck = value_checker(winning, champ_temp, tier_temp)
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
                              valueCheck = value_checker(winning, champ_temp, tier_temp)
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
                              valueCheck = value_checker(winning, champ_temp, tier_temp)
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
                              print("1: Event 1 / 2: Event 2 / 3: Event 3 / 4: Event 4 / 5: Event5")
                              print("6: Event 6 / 7: Event 7 / 8: Event 8 / 9: Event 9 / 10: Event10  ////  0: Exit")

                              i=input()
                              if i == '1':
                                        evt1 = 1 #Add Event 1
                              elif i == '2':
                                        evt2 = 1 #Add Event 2
                              elif i == '3':
                                        evt3 = 1 #Add Event 3
                              elif i == '4':
                                        evt4 = 1 #Add Event 4
                              elif i == '5':
                                        evt5 = 1 #Add Event 5
                              elif i == '6':
                                        evt6 = 1 #Add Event 6
                              elif i == '7':
                                        evt7 = 1 #Add Event 7
                              elif i == '8':
                                        evt8 = 1 #Add Event 8
                              elif i == '9':
                                        evt9 = 1 #Add Event 9
                              elif i == '10':
                                        evt10 = 1 #Add Event 10
                              elif i == '0':
                                        checker = 0
                                        break
                              else:
                                        print("Invalid Event Select")
                              
  
                                        

                                  
          
                    
