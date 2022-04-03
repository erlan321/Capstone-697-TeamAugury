

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from functions import Team_Augury_feature_functions

def generate_X_y(dataframe, target="popular_hr_3", post_basic=True, post_temporal=True, comment_basic=True, post_sent=True,
                 comment_sent=True, post_sBERT=True, comment_sBERT=True, increment=3):
                # targets: 'popular_hr_3', 'popular_hr_6','popular_hr_24', 'popular_max'
                # increments = 0, 3, 6
  from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
  from sklearn.decomposition import LatentDirichletAllocation
  import pandas as pd

  df = dataframe.copy()
  
  ### y - value decisions  ###
  y_column = [target]
    
  ### add the target column, or hot_proxy, for regressors
  
  ### set binary popular classification
  popular_max_threshold, popular_hr_24_threshold, popular_hr_6_threshold, popular_hr_3_threshold = 20, 5, 10, 10


  if target == "popular_max": 
    df['max_hotness_proxy'] = df.groupby(['post_id'])['upvotes_vs_hours'].transform( lambda x: np.max(x) ).fillna(value=0)
    df['popular_max'] = df.apply( lambda x: 1 if x['max_hotness_proxy']>=popular_max_threshold else 0, axis=1)
  elif target == "popular_hr_24":
    df['hot_proxy_hr_24'] = df.apply(lambda x: x['upvotes_vs_hours'] if x['hours_since_created']==24 else np.nan, axis=1)
    df['hot_proxy_hr_24'] = df.groupby(['post_id'])['hot_proxy_hr_24'].fillna( method='backfill').fillna(value=0) 
    df['popular_hr_24'] = df.apply( lambda x: 1 if x['hot_proxy_hr_24']>=popular_hr_24_threshold else 0, axis=1)
  elif target == "popular_hr_6":
    df['hot_proxy_hr_6'] = df.apply(lambda x: x['upvotes_vs_hours'] if x['hours_since_created'] in [6,12,18,24] else np.nan, axis=1)
    df['hot_proxy_hr_6'] = df.groupby(['post_id'])['hot_proxy_hr_6'].fillna( method='backfill').fillna(value=0) 
    df['hot_proxy_hr_6'] = df.apply(lambda x: np.nan if x['hours_since_created'] in [6,12,18,24] else x['hot_proxy_hr_6'], axis=1)
    df['hot_proxy_hr_6'] = df.groupby(['post_id'])['hot_proxy_hr_6'].fillna( method='backfill').fillna(value=0) 
    df['popular_hr_6'] = df.apply( lambda x: 1 if x['hot_proxy_hr_6']>=popular_hr_6_threshold else 0, axis=1)
  elif target == "popular_hr_3":
    df['hot_proxy_hr_3'] = df.apply(lambda x: x['upvotes_vs_hours'] if x['hours_since_created'] in [3,6,9,12,15,18,21,24] else np.nan, axis=1)
    df['hot_proxy_hr_3'] = df.groupby(['post_id'])['hot_proxy_hr_3'].fillna( method='backfill').fillna(value=0)
    df['hot_proxy_hr_3'] = df.apply(lambda x: np.nan if x['hours_since_created'] in [3,6,9,12,15,18,21,24] else x['hot_proxy_hr_3'], axis=1)
    df['hot_proxy_hr_3'] = df.groupby(['post_id'])['hot_proxy_hr_3'].fillna( method='backfill') .fillna(value=0) 
    df['popular_hr_3'] = df.apply( lambda x: 1 if x['hot_proxy_hr_3']>=popular_hr_3_threshold else 0, axis=1)

  else:
    print("target not in options, returning None")
    return None


  ### X - value decisions  ###

  ### column names broken into categories for ease of use
  X_col_post_basic = ['post_author_karma', 'number_comments_vs_hrs']
  X_col_post_temporal = ['time_hour', 'day_of_week']
  X_col_comment_basic = ['avg_comment_upvotes_vs_hrs', 'avg_comment_author_karma']
  X_col_post_sentiment = ['post_sentiment']
  X_col_comment_sentiment = ['avg_comment_sentiment']
  X_col_post_sbert = [f'post_sbert_{"{:03d}".format(i)}' for i in range(1, 385)]
  X_col_comment_sbert = [f'avg_comment_sbert_{"{:03d}".format(i)}' for i in range(1, 385)]

  ### columns that will need a standard scaler
  standardize_columns = []
  if post_basic == True: standardize_columns.extend(['number_comments_vs_hrs','post_author_karma'])
  if comment_basic == True: standardize_columns.extend(['avg_comment_upvotes_vs_hrs','avg_comment_author_karma'])

  ### concat the X_value feature columns we want from the above categories
  X_columns = []
  if post_basic==True: X_columns.extend(X_col_post_basic)
  if post_sent==True: X_columns.extend(X_col_post_sentiment)
  if post_temporal==True: X_columns.extend(X_col_post_temporal)
  if post_sBERT==True: X_columns.extend(X_col_post_sbert)
  if comment_basic==True: X_columns.extend(X_col_comment_basic)
  if comment_sent==True: X_columns.extend(X_col_comment_sentiment)
  if comment_sBERT==True: X_columns.extend(X_col_comment_sbert)

  ### apply increment
  ### Choose either hour 0 or hours in 3 hour or 6 hour increments
  if increment == 0: 
    df = df[df.hours_since_created==0]
  elif increment == 6:
    df = df[df.hours_since_created.isin([0,6,12,18])]
  elif increment == 3:
    df = df[df.hours_since_created.isin([0,3,6,9,12,15,18,21])]
  else:
    print("increment not in options, returning None")
    return None


  ### Create table

  df = df.copy()[X_columns + y_column]

  ### Apply one hot encoding prior to split (keeping column count equal)
  if post_temporal==True:
    df_dummies = pd.get_dummies(data=df, columns=X_col_post_temporal)
    added_cols = list(set(df_dummies.columns)-set(df.columns))
    X_columns = X_columns + added_cols
    for col in X_col_post_temporal:
      X_columns.remove(col)
    df = df_dummies


  ### train/test split
  X_train, X_test, y_train, y_test = train_test_split(df[X_columns], df[y_column], random_state = 42, test_size=0.2)

  ### apply standard scaling to select columns
  if standardize_columns == []:
    pass
  else:
    X_train, X_test = Team_Augury_feature_functions.standard_scale_column(X_train, X_test, standardize_columns)
  
  return X_train, X_test, y_train, y_test


def test_classifiers(X_train, X_test, y_train, y_test):

    #Pipeline approach taken from https://medium.com/vickdata/a-simple-guide-to-scikit-learn-pipelines-4ac0d974bdcf

  # Set classifiers
  classifiers = [
                DummyClassifier(strategy = 'constant', random_state=42, constant=0),
                LogisticRegression(random_state = 42),
                KNeighborsClassifier(3),
                SVC(kernel="rbf", C=0.025, probability=True),   #literature review will need to check against winners of iteration against SVC
                DecisionTreeClassifier(),
                RandomForestClassifier(),
                AdaBoostClassifier(),
                GradientBoostingClassifier()
                ]

  
  import warnings
  warnings.filterwarnings('ignore')  ### BE CAREFUL USING THIS :) Supressing the warning that y_values coming in as dataframe

  # Evaluate each classifier only on training data
  dic_test_results = {}
  for classifier in classifiers:
    clf = classifier
    clf.fit(X_train, y_train.values)
    y_pred = clf.predict(X_test)
    dic_test_results[clf.__class__.__name__] = round(accuracy_score(y_test, y_pred),3)
    dic_test_results[clf.__class__.__name__ + '_f1'] = round(f1_score(y_test, y_pred),3)
    dic_test_results[clf.__class__.__name__ + '_precision'] = round(precision_score(y_test, y_pred),3)
    dic_test_results[clf.__class__.__name__ + '_recall'] = round(recall_score(y_test, y_pred),3)
    dic_test_results[clf.__class__.__name__ + '_roc_auc'] = round(roc_auc_score(y_test, y_pred),3)

  return dic_test_results