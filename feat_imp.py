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
print ('Type: ', type(X_train) )
#print (X_train.columns)

#load pkl'd GBT clf
filename = "models/GradientBoostingClassifier_vanilla_model.sav" #note this is the 'Vanillia model', not the optimised tuned one
GB_loaded = pickle.load(open(filename, 'rb'))

#test load
result = GB_loaded.score(X_train, y_train)
print('Testing model load: Accuracy result for GBT:' , result)

print (GB_loaded.steps)
print ('The second element of the first slice: ', type(GB_loaded.steps[0][1] )) #This index's the col transformer. However, AttributeError: Transformer numerical (type Pipeline) does not provide get_feature_names.
print (GB_loaded.steps[1][1], type(GB_loaded.steps[1][1]))
print (len(GB_loaded.steps[1][1].feature_importances_), ': length of features importances') # accesses the model from the pipeline

# print (clf.steps[1][1].feature_importances_)
# print (clf.get_feature_names())

## Explore Feature importances and Labels prior to plotting
fimp = GB_loaded.steps[1][1].feature_importances_ #note len of 805 due to transformation, notably the one hot encoding day/hours
clabels = X_train.columns #note len of 776 'untransformed'

#print (GB_loaded[:-1].get_feature_names_out())

print (GB_loaded['preprocessor'].transformers_[1][1]['onehot'].get_feature_names()) #get's names from transformed array of one hot feat names

print (type (GB_loaded['preprocessor'].transformers_[1][1])) #['onehot'].get_feature_names())
#print (GB_loaded['preprocessor'].transformers_[1][1].get_feature_names_out())  #AttributeError: 'Pipeline' object has no attribute 'get_feature_names_out'
print (type (GB_loaded['preprocessor'].transformers_[1][1]['onehot']))#.get_feature_names())

# for i in range(0,1): #will be upto 40 of these 
#   sl_start = 0
#   sl_stop = 20
#   print (f'slice to {sl_stop} of fimp: {fimp[sl_start:sl_stop]}')
#   print (f'slice to {sl_stop} of clabels: {clabels[sl_start:sl_stop]}')
#   sl_start += 20
#   sl_stop += 20






# Plot feature importances
figure(figsize=(10, 50), dpi=80)

def plot_feature_importance(model):
  n_features = X_train.shape[1] + 29 # Note X_train is transformed by the pipeline clf from 776 to 805
  plt.barh(np.arange(n_features), model.feature_importances_, align='center')
  plt.yticks(np.arange(n_features), X_train.columns)
  plt.xlabel("Feature importance")
  plt.ylabel("Feature")
  plt.ylim(-1, n_features)

# plot_feature_importance(GB_loaded.steps[1][1])
