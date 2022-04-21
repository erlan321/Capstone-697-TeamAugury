import pandas as pd
import numpy as np  
import psycopg2 
from functions import Team_Augury_SQL_func
from functions.Team_Augury_feature_functions import generate_features_csv
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import pickle
from functions.Team_Augury_feature_functions import feature_to_x_y

rnd_state = 42

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
  lower_timestamp = '2022-03-01 14:30:00'
  upper_timestamp = '2022-04-18 10:00:00' 

  #get data
  post_data, comments_data = Team_Augury_SQL_func.sql_by_timestamp(conn,sr_id,lower_timestamp,upper_timestamp)
  print("Loaded {} posts".format(len(post_data.post_id.unique())))
  print("Loaded {} comments".format(len(comments_data.post_id.unique())))

  csv_folder = "saved_work/"

# Uncomment to generate features and save data to csv (~10 minutes)
  print("Featurizing...")
  feature_df = generate_features_csv(post_data,comments_data, csv_folder + "backup_features_data.csv")

else:
  print("Loading features...")
  feature_df = pd.read_csv("saved_work/backup_features_data.csv")

targets = ['popular_hr_3'] #, 'popular_hr_6','popular_hr_24'] #, 'popular_max'] --> Team decision based on EDA
increments = [0] #, 3, 6] --> team decision based on EDA

X, y = feature_to_x_y(feature_df)

numeric_features = ['number_comments_vs_hrs','post_author_karma', 'avg_comment_upvotes_vs_hrs','avg_comment_author_karma']
categorical_features = ['time_hour', 'day_of_week']

print("Training model...")
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', categories='auto'))])

preprocessor = ColumnTransformer(
            transformers=[
                ('numerical', numeric_transformer, numeric_features),
                ('categorical', categorical_transformer, categorical_features)], remainder="passthrough")


classifier = LogisticRegression(random_state=rnd_state, solver="lbfgs", C=1.0, penalty="l2", max_iter=100000, multi_class="ovr", n_jobs=-1, class_weight="balanced")

clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
clf.fit(X,y)
filename = "models/"+ classifier.__class__.__name__ + "_final_baseline_model.pkl"
pickle.dump(clf, open(filename, 'wb'))

print("Model saved to {}".format(filename))