# -*- coding: utf-8 -*-
 

#Importing the necessary libraries
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import SelectKBest,chi2
import pandas as pd
import matplotlib.pyplot as plt

#Preprocessing
def load_dataset(filename):
  dataset = pd.read_csv(filename)
   
   
  X = dataset.drop(columns=['2019_Runs','PLAYER'])#.drop(columns=['PLAYER','Mat','HS','2019_Runs','PLAYER','100','50'])
  y = dataset['2019_Runs']
  X = X.astype(str)
  return X,y

# feature selection
def select_features(X_train, y_train, X_test):
	fs = SelectKBest(score_func=chi2, k='all')
	fs.fit(X_train, y_train)
	X_train_fs = fs.transform(X_train)
	X_test_fs = fs.transform(X_test)
	return X_train_fs, X_test_fs, fs

#Specify the path 
path = r'C:\Users\adeep\Downloads\DATA.csv'
X,y = load_dataset(path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)


X_train_fs, X_test_fs, fs = select_features(X_train , y_train , X_test )
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
    
    
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
plt.show()

"""
Features are in the order 
0 : Mat
1 : Inns
2 : NO
3 : 2018_Runs
4 : HS   : Highest Score
5 : Avg  : Average
6 : BF   : Balls Faced
7 : SR   : Strike Rate 
8 : 100  : Number of 100s
9 : 50   : Number of 50s
10: 6s   : Number of 6s
11: 4s   : Number of 4s
12 : N_O : Not outs
"""