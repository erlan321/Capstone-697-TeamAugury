import pandas as pd
import numpy as np  
import psycopg2 
from functions import Team_Augury_SQL_func
from functions.Team_Augury_feature_functions import generate_features_csv
from sklearn.metrics import accuracy_score, f1_score
import pickle

choice = input("Load data (Y) or Run full featurization (N)?:").upper()

if choice == 'N':

  # connect to database
  print("Connecting to database and loading posts/comments...")
  conn = psycopg2.connect(
      host = 'capstone697.czsaza7am68b.us-east-1.rds.amazonaws.com' ,
      port = '5432',
      database = 'Capstone697', 
      user = 'read_o',  
      password = 'EAC697Reddit' , 
      connect_timeout=3)

  ### NEW SQL Variables
  sr_id = '(4, 31760, 31764, 31766)'   #[(4, '2qhhq', 'investing'), (16, 'mouw', 'science'), (130, '2qh1i', 'AskReddit'), (18162, '2qh3l', 'news'), (31760, '2qjuv', 'StockMarket'), (31764, '2qjfk', 'stocks'), (31766, '2th52', 'wallstreetbets')]
  lower_timestamp = '2022-04-04 09:00:00'
  upper_timestamp = '2022-04-18 11:00:00' 

  #get data
  post_data, comments_data = Team_Augury_SQL_func.sql_by_timestamp(conn,sr_id,lower_timestamp,upper_timestamp)
  print("Loaded {} posts".format(len(post_data.post_id.unique())))
  print("Loaded {} comments".format(len(comments_data.post_id.unique())))

  csv_folder = "saved_work/"

# Uncomment to generate features and save data to csv (~10 minutes)
  print("Featurizing...")
  feature_df = generate_features_csv(post_data,comments_data, csv_folder + "backup_features_data_final.csv")

else:
  print("Loading features...")
  feature_df = pd.read_csv("saved_work/backup_features_data_final.csv")

targets = ['popular_hr_3'] #, 'popular_hr_6','popular_hr_24'] #, 'popular_max'] --> Team decision based on EDA
increments = [0] #, 3, 6] --> team decision based on EDA


rnd_state = 42

df = feature_df.copy()  #so code can be reused more easily

# set target variable and calculate in df
y_column = ["popular_hr_3"]
popular_hr_3_threshold = 10

df['hot_proxy_hr_3'] = df.apply(lambda x: x['upvotes_vs_hours'] if x['hours_since_created'] in [3,6,9,12,15,18,21,24] else np.nan, axis=1)
df['hot_proxy_hr_3'] = df.groupby(['post_id'])['hot_proxy_hr_3'].fillna( method='backfill').fillna(value=0)
df['hot_proxy_hr_3'] = df.apply(lambda x: np.nan if x['hours_since_created'] in [3,6,9,12,15,18,21,24] else x['hot_proxy_hr_3'], axis=1)
df['hot_proxy_hr_3'] = df.groupby(['post_id'])['hot_proxy_hr_3'].fillna( method='backfill') .fillna(value=0) 
df['popular_hr_3'] = df.apply( lambda x: 1 if x['hot_proxy_hr_3']>=popular_hr_3_threshold else 0, axis=1)

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
numeric_features = ['number_comments_vs_hrs','post_author_karma', 'avg_comment_upvotes_vs_hrs','avg_comment_author_karma']

categorical_features = X_col_post_temporal

### concat the X_value feature columns we want from the above categories
X_columns = []
X_columns.extend(X_col_post_basic)
X_columns.extend(X_col_post_sentiment)
X_columns.extend(X_col_post_temporal)
X_columns.extend(X_col_post_sbert)
X_columns.extend(X_col_comment_basic)
X_columns.extend(X_col_comment_sentiment)
X_columns.extend(X_col_comment_sbert)

### increment selected by design = 0h
df = df[df.hours_since_created==0]

### Create table
df = df.copy()[X_columns + y_column]
X = df[X_columns]
y = df[y_column].values.ravel()

print("Loading models...")
#load pkl'd SVC clf
filename = "models/SVC_rbf_final_model.pkl"
SVC_loaded = pickle.load(open(filename, 'rb'))

filename = "models/SVC_rbf2_final_model.pkl"
SVC2_loaded = pickle.load(open(filename, 'rb'))

filename = "models/SVC_linear_final_model.pkl"
SVCL_loaded = pickle.load(open(filename, 'rb'))


#load pkl'd GBT clf
filename = "models/GradientBoostingClassifier_doublecheck_model.pkl"
GBT_loaded = pickle.load(open(filename, 'rb'))

#load pkl'd LR clf
filename = "models/LogisticRegression_final_baseline_model.pkl"
LR_loaded = pickle.load(open(filename, 'rb'))


print("Results...")
print("LR Baseline Accuracy: {}".format(accuracy_score(y, LR_loaded.predict(X))))
print("LR Baseline F1: {}".format(f1_score(y, LR_loaded.predict(X))))
print("...")
print("SVC RBF Selected Model Accuracy: {}".format(accuracy_score(y, SVC_loaded.predict(X))))
print("SVC RBF Selected Model F1: {}".format(f1_score(y, SVC_loaded.predict(X))))
print("...")
print("SVC RBF Alternate Model Accuracy: {}".format(accuracy_score(y, SVC2_loaded.predict(X))))
print("SVC RBF Alternate Model F1: {}".format(f1_score(y, SVC2_loaded.predict(X))))
print("...")
print("SVC Linear Alternate Model Accuracy: {}".format(accuracy_score(y, SVCL_loaded.predict(X))))
print("SVC Linear Alternate Model F1: {}".format(f1_score(y, SVCL_loaded.predict(X))))
print("...")
print("GBDT double check Accuracy: {}".format(accuracy_score(y, GBT_loaded.predict(X))))
print("GBDT double check F1: {}".format(f1_score(y, GBT_loaded.predict(X))))
