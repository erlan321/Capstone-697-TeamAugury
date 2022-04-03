import pandas as pd
import numpy as np  
from datetime import datetime
import psycopg2 
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer ###Vader Sentiment

from sentence_transformers import SentenceTransformer  ### SBERT

# from detoxify import Detoxify  ### detoxify
from profanity_filter import ProfanityFilter

from functions import Team_Augury_SQL_func
from functions import Team_Augury_feature_functions
from functions.Team_Augury_Iterate import generate_X_y
from functions.Team_Augury_Iterate import test_classifiers
import itertools
from tqdm import tqdm

# connect to database
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
upper_timestamp = '2022-03-28 08:00:00'  #batch 900	2022-03-28 07:21:27

#get data
post_data, comments_data = Team_Augury_SQL_func.sql_by_timestamp(conn,sr_id,lower_timestamp,upper_timestamp)
print(post_data.batch_id.min(),post_data.batch_id.max())
print(post_data.sr.unique())
print(len(post_data.post_id.unique()))



##### MOVE TO A FUNCTION #####
# ### Featurization comes from stored functions imported from .py files above.  Comment out any part we don't want.


# ### Start by cleaning profanity from posts and comments
# post_data = Team_Augury_feature_functions.post_profanity_removal(post_data.copy())
# #display(post_data)
# comments_data = Team_Augury_feature_functions.comment_profanity_removal(comments_data.copy())
# #display(comments_data)

# ### Basic feature addtions for post-only data
# feature_df = Team_Augury_feature_functions.post_basic_features(post_data.copy())
# ### Basic feature additions for comments (avg's)
# feature_df = feature_df.merge( Team_Augury_feature_functions.comment_basic_features(comments_data.copy()) , how='left',on=['batch_id','post_id'])
# ### VADER sentiment for posts
# feature_df = feature_df.merge( Team_Augury_feature_functions.post_sentiment_func(post_data.copy()) , how='left',on=['post_id', 'post_text'])
# ### VADER sentiment for comments (avg)
# feature_df = feature_df.merge( Team_Augury_feature_functions.comment_sentiment_func(comments_data.copy()) , how='left',on=['batch_id','post_id'])

# ### SBERT sentence transform for posts
# feature_df = feature_df.merge( Team_Augury_feature_functions.post_sentence_transform_func(post_data.copy()) , how='left',on=['post_id'])

# ### WARNING: Comments SBERT featurization can take over 20-30 minutes
# ### SBERT sentence transform for comments (avg's)
# feature_df = feature_df.merge( Team_Augury_feature_functions.comment_sentence_transform_func(comments_data.copy()) , how='left',on=['batch_id','post_id'])
# ### load a backup of Comments SBERT as a workaround
# # feature_df = feature_df.merge( pd.read_csv("/content/gdrive/Shareddrives/Capstone697/Stub Code/backup_comments_SBERT_features.csv") , how='left',on=['batch_id','post_id'])

# ### above comments SBERT features takes over 20 minutes to run
# ### export to Google Drive
# #df = Team_Augury_feature_functions.comment_sentence_transform_func(comments_data)
# #df.to_csv("/content/gdrive/Shareddrives/Capstone697/Stub Code/backup_comments_SBERT_features.csv", index=False)

# ### Import backup from Google Drive
# #backup_comments_sbert_data = pd.read_csv("/content/gdrive/Shareddrives/Capstone697/Stub Code/backup_comments_SBERT_features.csv")

# ### check for NaN's
# #display(feature_df[feature_df.isna().any(axis=1)])
# feature_df.replace([np.nan], 0, inplace=True) #clean NaN's formed from comments merging
# #print('-- raw feature table --')
# display(feature_df)

feature_df = pd.read_csv("saved_work/backup_features_data.csv")

csv_folder = "saved_work/"

targets = ['popular_hr_3', 'popular_hr_6','popular_hr_24'] #, 'popular_max']
increments = [0, 3, 6]

""" features = ["post_basic", "post_temporal", "comment_basic", "post_sent", "comment_sent", "post_sBERT", "comment_sBERT"]
parameters = []
for p in itertools.product([True,False],repeat=7):
    params = dict(zip(features,p))
    parameters.append(params)

for params in parameters:
  if all(value == False for value in params.values()):
    parameters.remove(params) """

parameters = [{'post_basic': True, 'post_temporal': True, 'comment_basic': True, 'post_sent': True, 'comment_sent': True, 'post_sBERT': True, 'comment_sBERT': True},
              {'post_basic': True, 'post_temporal': True, 'comment_basic': True, 'post_sent': True, 'comment_sent': False, 'post_sBERT': True, 'comment_sBERT': False},
              {'post_basic': True, 'post_temporal': True, 'comment_basic': True, 'post_sent': True, 'comment_sent': True, 'post_sBERT': False, 'comment_sBERT': False},
              {'post_basic': True, 'post_temporal': True, 'comment_basic': True, 'post_sent': False, 'comment_sent': False, 'post_sBERT': False, 'comment_sBERT': False}]


##add intelligence to code here to not have to manually manipulate the csv files
#lsr = pd.read_csv(csv_folder + "last_successful.csv").iloc[:, 1:] #last successful run
#lsr = pd.DataFrame()
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
      X_train, X_test, y_train, y_test = generate_X_y(feature_df, target=target, increment=increment, post_basic=params["post_basic"], post_temporal=params["post_temporal"], comment_basic=params["comment_basic"],
                  post_sent=params["post_sent"], comment_sent=params["comment_sent"], post_sBERT=params["post_sBERT"], comment_sBERT=params["comment_sBERT"])
      
      classifier_tests = test_classifiers(X_train, X_test, y_train, y_test)

      dic_results = {"target":target, "increment":increment}
      dic_results.update(params)
      dic_results.update(classifier_tests)
      results.append(dic_results)
    
      temp_df = pd.DataFrame.from_records(results)
      temp_df.to_csv(csv_folder + "temp.csv")

        #code to adjust the csv files with results