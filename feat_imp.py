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
import altair as alt
from altair_saver import save


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
# clabels = list(X_train.columns) #note len of 776 'untransformed'
# clabels = clabels + list(GB_loaded['preprocessor'].transformers_[1][1]['onehot'].get_feature_names() ) 
# #print (len(clabels), 'is the length of the labels') #note this is two more than required as the original hours/days features still in here....
# #remove last two days of week as we can see where these land and understand from Viz where they should be inserted...
# clabels.remove('x1_5')
# clabels.remove('x1_6')
# combined_ns = list(zip(clabels,fimp))
#print (f'first 40 {combined_ns[:40]}')

#recombining features to match labels to features:
clabels_v2 = list(X_train.columns)
clabels_v2.remove('time_hour')
clabels_v2.remove('day_of_week')

clabels_v2 = clabels_v2[0:3] + clabels_v2[387:390] + list(GB_loaded['preprocessor'].transformers_[1][1]['onehot'].get_feature_names()) + clabels_v2[3:387]+clabels_v2[390:]

#new_combined_ns = list(zip(clabels_v2,fimp))
#print (f'first 40 {new_combined_ns[:40]}')

# Plot feature importances 1 the long chart
figure(figsize=(10, 80), dpi=100)

def plot_feature_importance(model):
  n_features = X_train.shape[1] + 29 # Note X_train is transformed by the pipeline clf from 776 to 805
  plt.barh(np.arange(n_features), model.feature_importances_, align='center')
  plt.yticks(np.arange(n_features), clabels_v2) #new labels V2
  plt.xlabel("Feature importance")
  plt.ylabel("Feature")
  plt.ylim(-1, n_features)

# Unhash lines 62 and 63 to create the long chart of all features
# plot_feature_importance(GB_loaded.steps[1][1])
# plt.savefig("saved_work/feat_imp_v3.png")

# Plot feature importances 2 focussed charts

#step 1 aggreate the SBERT scores from the 805 features: 
# the first 3 are originals
# the next  3 are also originals
# the next 31 are: 0-23 hours of day 0-6 day of week
# Remainder are SBERT posts (384) and SBERT comments(384) 

first_six_scores = fimp[0:6]
hours_and_days = fimp[6:37]
sbert_post = fimp[37:421]
sbert_comment = fimp[421:806]

#test lengths
print ('Testing length of feature groups: should be 6,31,384,384.',len(first_six_scores), len(hours_and_days), len(sbert_post), len(sbert_comment))

sbert_post    = np.array([np.sum(sbert_post)])
sbert_comment = np.array([np.sum(sbert_comment)])

#step 2 create new labels and array of scores
scoring_array = np.concatenate(
  (first_six_scores,
  hours_and_days,
  sbert_post,
  sbert_comment),
  axis=None
)

short_label_list = clabels_v2[0:37] + ['SBERT_posts', 'SBERT_comments']
new_combined_ns = (zip(short_label_list,scoring_array))
df = pd.DataFrame(new_combined_ns, columns=['Feature','Importance'])
#print (df)

# Unhash lines 98-104 to create the final aggregated chart
# chart = alt.Chart(df).mark_bar().encode(
#   x = 'Feature:N',
#   y = 'Importance:Q'
# )

# save(chart, "saved_work/feat_imp_short.png")
