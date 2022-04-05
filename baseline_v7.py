import pandas as pd
import numpy as np  
from datetime import datetime
import psycopg2 
from functions import Team_Augury_SQL_func
from functions.Team_Augury_Iterate import test_classifiers
from functions.Team_Augury_feature_functions import generate_features_csv
import itertools
from tqdm import tqdm

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
  lower_timestamp = '2022-03-01 14:30:00'  #batch 260 3/1/2022 14:37:31
  upper_timestamp = '2022-04-04 08:00:00'  #batch 900	2022-03-28 07:21:27

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

targets = ['popular_hr_3', 'popular_hr_6','popular_hr_24'] #, 'popular_max']
increments = [0, 3, 6]

# Use the below if you want to test out all potential features combinations
""" features = ["post_basic", "post_temporal", "comment_basic", "post_sent", "comment_sent", "post_sBERT", "comment_sBERT"]
parameters = []
for p in itertools.product([True,False],repeat=7):
    params = dict(zip(features,p))
    parameters.append(params)

for params in parameters:
  if all(value == False for value in params.values()):
    parameters.remove(params) """

# With this set of parameters, the processing time is ~5 minutes
parameters = [{'post_basic': True, 'post_temporal': True, 'comment_basic': True, 'post_sent': True, 'comment_sent': True, 'post_sBERT': True, 'comment_sBERT': True},
              {'post_basic': True, 'post_temporal': True, 'comment_basic': True, 'post_sent': True, 'comment_sent': False, 'post_sBERT': True, 'comment_sBERT': False},
              {'post_basic': True, 'post_temporal': True, 'comment_basic': True, 'post_sent': True, 'comment_sent': True, 'post_sBERT': False, 'comment_sBERT': False},
              {'post_basic': True, 'post_temporal': True, 'comment_basic': True, 'post_sent': False, 'comment_sent': False, 'post_sBERT': False, 'comment_sBERT': False}]


results = []
for target in targets:
  print("target: {}".format(target))
  for increment in increments:
    print("increment: {}".format(increment))
    for params in tqdm(parameters):
      """ filter = lsr[(lsr['target'] == target) & (lsr['increment'] == increment) & (lsr['post_basic'] == params["post_basic"]) & (lsr['post_temporal'] == params["post_temporal"])
                    & (lsr['comment_basic'] == params["comment_basic"]) & (lsr['post_sent'] == params["post_sent"]) & (lsr['comment_sent'] == params["comment_sent"])
                    & (lsr['post_sBERT'] == params["post_sBERT"]) & (lsr['comment_sBERT'] == params["comment_sBERT"])]
   """
      #if filter.shape[0] == 0:
      classifier_tests = test_classifiers(feature_df, target=target, increment=increment, post_basic=params["post_basic"], post_temporal=params["post_temporal"], comment_basic=params["comment_basic"],
                  post_sent=params["post_sent"], comment_sent=params["comment_sent"], post_sBERT=params["post_sBERT"], comment_sBERT=params["comment_sBERT"])
      dic_results = {"target":target, "increment":increment}
      dic_results.update(params)
      dic_results.update(classifier_tests)
      results.append(dic_results)
    
      temp_df = pd.DataFrame.from_records(results)
      temp_df.to_csv("saved_work/temp.csv")

results_df = pd.DataFrame.from_records(results)
results_df.to_csv("saved_work/results.csv")
