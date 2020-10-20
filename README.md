# flipr-hackathon-7.0

## Which model have you used for Total IPL 2020 Runs prediction for each player? Explain your model.
Solution:
### Preprocessing the data.
1. For the highest scores having ‘*’ (Not out) we create an additional column taking only binary values (0 or 1) for the presence and absence of ‘*’.
2. The strings with ‘*’ are removed with the help of string manipulations on regular expressions (regex).
3. The ‘Avg’ column which contained ‘-’ for the players who never got out was replaced with their total runs.
4. We then calculate the significance of each feature on the ‘2019_Runs’ using ‘SelectKBest’ and calculating the chi-square to find the confidence Interval of each feature.
5. We drop the features that have a less score in chi-square and use only the significant features.
6. We use label encodings to encode the dataset.
7. Post preprocessing, a RandomForestRegressor was employed, the hyperparameters were n_trees = 100, max_features='auto'
8. Other Machine Learning algorithms such as AdaBoost, XGBoost,lgbm were conducted in order to achieve the highest score in the appropriate metrics used and as stated below.
9. After training the model we predict the 2019_Runs.
10.The metric employed was r2_score.

### Predicting 2020_Runs:
1. The same RandomForestRegressor model was trained using the features and 2019_Runs instead of the 2018_Runs.
2. This model then predicts 2020_Runs.
