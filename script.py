from pybaseball import playerid_lookup
from pybaseball import statcast_batter
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

player=playerid_lookup('judge','aaron')
player_id=player.iloc[:,2].values[0]   #592450

stats = statcast_batter('2017-03-01', '2017-10-01', player_id)
#stats.to_pickle('aaron_judge_stats.pkl')

stats['type'] = stats['type'].map({'S': 1, 'B': 0})
stats = stats.dropna(subset=['plate_x', 'plate_z', 'type'])  # Drop rows with NaN values

plt.scatter(x=stats['plate_x'], y=stats['plate_z'], c=stats['type'], cmap=plt.cm.coolwarm, alpha=0.25)
plt.show()

training_set, validation_set = train_test_split(stats, random_state=1)

# Separate features and target variable
X_train = training_set[['plate_x', 'plate_z']]
y_train = training_set['type']
X_val = validation_set[['plate_x', 'plate_z']]
y_val = validation_set['type']

# Check for NaN values in the training and validation sets
print("NaN values in training set:", X_train.isnull().sum().sum())
print("NaN values in validation set:", X_val.isnull().sum().sum())

# Instantiate the classifier and fit the model
classifier = SVC(kernel='rbf', gamma=10, C=10)
classifier.fit(X_train, y_train)

# Score the model on the validation set
score = classifier.score(X_val, y_val)  #0.82
print("Validation Set Score:", score)