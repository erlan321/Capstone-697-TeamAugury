import pandas as pd
import numpy as np  
from datetime import datetime 
from functions.Team_Augury_load_transform_saved import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import cla, figure


X_train, y_train = load_and_preprocess()
print ('Testing load, length of X, y:', X_train.shape, y_train.shape)
#print ('Type: ', type(X_train) )
#print (X_train.columns)

#load pkl'd GBT clf
filename = "models/GradientBoostingClassifier_vanilla_model.sav" #note this is the 'Vanillia model', not the optimised tuned one
GB_loaded = pickle.load(open(filename, 'rb'))

#test load
result = GB_loaded.score(X_train, y_train)
print('Testing model load: Accuracy result for GBT:' , result)
#print (GB_loaded['preprocessor'].transformers_[1][1]['onehot'].get_feature_names()) #get's names from transformed array of one hot feat names

## Explore Feature importances and Labels prior to plotting
fimp = GB_loaded.steps[1][1].feature_importances_ #note len of 805 due to transformation, notably the one hot encoding day/hours
clabels = list(X_train.columns) #note len of 776 'untransformed'
clabels = clabels + list(GB_loaded['preprocessor'].transformers_[1][1]['onehot'].get_feature_names() ) 
#print (len(clabels), 'is the length of the labels') #note this is two more than required as the original hours/days features still in here....
#remove last two days of week as we can see where these land and understand from Viz where they should be inserted...
clabels.remove('x1_5')
clabels.remove('x1_6')

combined_ns = list(zip(clabels,fimp))

#print (f'first 40 {combined_ns[:40]}')

#recombining features, how I think it works:
clabels_v2 = list(X_train.columns)
clabels_v2.remove('time_hour')
clabels_v2.remove('day_of_week')
#print (clabels_v2[387:390])

clabels_v2 = clabels_v2[0:3] + clabels_v2[387:390] + list(GB_loaded['preprocessor'].transformers_[1][1]['onehot'].get_feature_names()) + clabels_v2[3:387]+clabels_v2[390:]

new_combined_ns = list(zip(clabels_v2,fimp))
#print (f'first 40 {new_combined_ns[:40]}')
#print (new_combined_ns[418:421])


# Plot feature importances
figure(figsize=(10, 80), dpi=100)

def plot_feature_importance(model):
  n_features = X_train.shape[1] + 29 # Note X_train is transformed by the pipeline clf from 776 to 805
  plt.barh(np.arange(n_features), model.feature_importances_, align='center')
  plt.yticks(np.arange(n_features), clabels_v2) #new labels V2
  plt.xlabel("Feature importance")
  plt.ylabel("Feature")
  plt.ylim(-1, n_features)

plot_feature_importance(GB_loaded.steps[1][1])
plt.savefig("saved_work/feat_imp_v3.png")
