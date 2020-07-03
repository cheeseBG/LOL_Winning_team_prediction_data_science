# LOL_Winning_team_prediction_data_science

This is repository for data science term project.


# Summary

There are 3 kinds of codes
1. data plotting
2. real-time prediction
3. model analysis

Data plotting code will show the analysis of our data<br>
Real-time prediction code will receive event from you, and will predict whether win or lose<br>
Model analysis will show the analysis of our used learning model - kNN, k-means<br>

# Data plotting

Data plotting will show how data is consisted<br>

    data_explore.py will work for data plotting

Data is from [here](https://lolalytics.com/) with 10.13 patch version, Korea server and [kaggle data](https://www.kaggle.com/gyejr95/league-of-legendslol-ranked-games-2020-ver1)<br>
First data is used for prediction result is same. To be referred.<br>
Second data is used for prediction.<br>
If you run data plotting code, you may see the various analysis of our data<br>

# Real-time prediction

    dirty_processing_main.py will work for real-time prediction and model analysis

If you run real-time prediction, you may get the prediction of your game.<br>
Each input should be done by hand with keyboard.<br>
Prediction is used with knn algorithm - which we implemented not module.<br>
If prediction result is same as win-win or lose-lose, then average winning rate of your team will decides the win<br>

# Model analysis

     dirty_processing_main.py will work for real-time prediction and model analysis

We analyzed our data with knn and k-means<br>
knn will show confusion matrix k-fold evaluation with hyper-parameter tuning<br>
k-means used ensemble learning to enhance algorithm and show confusion matrix of our data<br>
