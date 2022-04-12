import pandas as pd
import numpy as np  
from datetime import datetime 
from functions.Team_Augury_load_transform_saved import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


X_train, y_train = load_and_preprocess()
print ('Testing load, length of X, y:', X_train.shape, y_train.shape)

#load pkl'd GBT clf
filename = "models/GradientBoostingClassifier_vanilla_model.sav" #note this is the 'Vanillia model', not the optimised tuned one
GB_loaded = pickle.load(open(filename, 'rb'))

#test load
result = GB_loaded.score(X_train, y_train)
print('Testing model load: Accuracy result for GBT:' , result)
print (len(GB_loaded.steps[1][1].feature_importances_)) #This shows the pickled model only has 35 features whereas it should have 776...
print (GB_loaded.steps)
print (GB_loaded.steps[1][1], type(GB_loaded.steps[1][1]))
print (len(GB_loaded.steps[1][1].feature_importances_), ': length of features importances') # how to access the model from the pipeline

# print (clf.steps[1][1].feature_importances_)
# print (clf.get_feature_names())

# Plot feature importances
figure(figsize=(10, 50), dpi=80)

def plot_feature_importance(model):
  n_features = X_train.shape[1]
  plt.barh(np.arange(n_features), model.feature_importances_, align='center')
  plt.yticks(np.arange(n_features), X_train.columns)
  plt.xlabel("Feature importance")
  plt.ylabel("Feature")
  plt.ylim(-1, n_features)
# plot_feature_importance(clf.steps[1][1])
