import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
from functions import Team_Augury_feature_functions

def test_classifiers(dataframe, target="popular_hr_3", post_basic=True, post_temporal=True, comment_basic=True, post_sent=True,
                 comment_sent=True, post_sBERT=True, comment_sBERT=True, increment=0):
                # targets: 'popular_hr_3', 'popular_hr_6','popular_hr_24', 'popular_max'
                # increments = 0, 3, 6
  

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
  numeric_features = []
  if post_basic == True: numeric_features.extend(['number_comments_vs_hrs','post_author_karma'])
  if comment_basic == True: numeric_features.extend(['avg_comment_upvotes_vs_hrs','avg_comment_author_karma'])

  ### Apply one hot encoding prior to split (keeping column count equal)
  if post_temporal==True:
    ### columns that will need a onehotencode scaler
    categorical_features = X_col_post_temporal


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
  X = df[X_columns]
  y = df[y_column].values.ravel()
  
  # Set classifiers
  classifiers = [
                DummyClassifier(strategy = 'constant', random_state=42, constant=0),
                LogisticRegression(random_state = 42),
                KNeighborsClassifier(),
                SVC(kernel="rbf", C=0.025, probability=True),   #literature review will need to check against winners of iteration against SVC
                DecisionTreeClassifier(random_state=42),
                RandomForestClassifier(random_state=42),
                AdaBoostClassifier(random_state=42),
                GradientBoostingClassifier(random_state=42)
                ]

  
  import warnings
  warnings.filterwarnings('ignore')  ### BE CAREFUL USING THIS :) Supressing the warning that y_values coming in as dataframe

  numeric_transformer = Pipeline(
      steps=[("scaler", StandardScaler())]
  )

  categorical_transformer = Pipeline(steps=[
          ('onehot', OneHotEncoder(handle_unknown='ignore', categories='auto'))])

  if numeric_features != [] and categorical_features != []:
    preprocessor = ColumnTransformer(
            transformers=[
                ('numerical', numeric_transformer, numeric_features),
                ('categorical', categorical_transformer, categorical_features)])
  elif numeric_features != [] and categorical_features == []:
    preprocessor = Pipeline(steps=[
        ('numerical', numeric_transformer, numeric_features)])
  elif numeric_features == [] and categorical_features != []:
    preprocessor = Pipeline(steps=[
        ('categorical', categorical_transformer, categorical_features)])
  else:
    preprocessor = None

  # Setup kfold cross validation
  #cv = KFold(n_splits=5, shuffle=True, random_state=42)

  # Moved to StratifiedKFold due to imbalanced dataset https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  

  #Scoring metrics
  scoring = {'acc': 'accuracy', 'f1': 'f1'}
  
  # Evaluate each classifier only on training data
  dic_test_results = {}
  for classifier in classifiers:
    if preprocessor != None:
      clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    else:
      clf = Pipeline(steps=[('classifier', classifier)])
    
    # Perform cross validation
    scores = cross_validate(clf, X, y, scoring=scoring, return_train_score=True, cv=cv, n_jobs=-1)

    dic_test_results[classifier.__class__.__name__ + "_train_acc"] = round(np.mean(scores["train_acc"]),3)
    dic_test_results[classifier.__class__.__name__ + "_test_acc"] = round(np.mean(scores["test_acc"]),3)
    dic_test_results[classifier.__class__.__name__ + "_train_f1"] = round(np.mean(scores["train_f1"]),3)
    dic_test_results[classifier.__class__.__name__ + "_test_f1"] = round(np.mean(scores["test_f1"]),3)

  return dic_test_results