from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from profanity_filter import ProfanityFilter
import spacy  #needed for language profanity filtering

def hr_func(ts): 
    return ts.hour #returns the hour as an integer

def day_func(ts):   ### NON-Production function (just for display purposes)
    return ts.day_name() #returns the day of the week as a name

def day_num_func(ts): 
    return ts.day_of_week #returns the day of the week as a number

def post_basic_features(df):
    df = df.copy()
    ### upvotes/hours (or hot proxy) and comments/hours
    df['upvotes_vs_hours'] = df.apply(lambda x: x.post_upvotes/1 if x.hours_since_created==0 else x.post_upvotes/x.hours_since_created, axis=1)
    df['number_comments_vs_hrs'] = df.apply(lambda x: x.number_comments/1 if x.hours_since_created==0 else x.number_comments/x.hours_since_created, axis=1)
    ### clean those new columns
    df['upvotes_vs_hours'].replace([np.inf, -np.inf], 0, inplace=True)
    df['upvotes_vs_hours'].replace([np.nan], 0, inplace=True)
    df['number_comments_vs_hrs'].replace([np.inf, -np.inf], 0, inplace=True)
    df['number_comments_vs_hrs'].replace([np.nan], 0, inplace=True)
    ### create hour and weekday features
    df['time_hour'] = df.post_created_at.apply(hr_func)
    df['day_of_week'] = df.post_created_at.apply(day_num_func)

    return df

def comment_basic_features(df):
    df = df.copy()
    ### upvotes/hours (or hot proxy) and comments/hours
    df['comment_upvotes_vs_hrs'] = df.apply(lambda x: x.comment_upvotes/1 if x.hours_since_created==0 else x.comment_upvotes/x.hours_since_created, axis=1)
    ### clean those new columns
    df['comment_upvotes_vs_hrs'].replace([np.inf, -np.inf], 0, inplace=True)
    df['comment_upvotes_vs_hrs'].replace([np.nan], 0, inplace=True)
    ### create hour and weekday features
    df['time_hour'] = df.comment_created_at.apply(hr_func)
    df['day_of_week'] = df.comment_created_at.apply(day_num_func)

    merge_df = df.copy().groupby(['batch_id','post_id'])[['comment_upvotes_vs_hrs','comment_author_karma','time_hour','day_of_week']].mean().reset_index()
    merge_df = merge_df.rename(
        {'comment_upvotes_vs_hrs':'avg_comment_upvotes_vs_hrs','comment_author_karma':'avg_comment_author_karma','time_hour':'avg_comment_time_hour', 'day_of_week':'avg_comment_day_of_week'},
        axis=1)

    return merge_df 


def post_sentiment_func(df):
    ### Functions for creating VADER sentiment for posts and comments
    ### based on 682 Social Media Assignment4 and this blog
    ### https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664

    ### set the sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    ### Function to extract only the 'compound' or aggregate sentiment
    def compound_sentiment_func(text):
        all_sentiment = sid.polarity_scores(text)
        compound_sentiment = all_sentiment['compound']
        return compound_sentiment

    merge_df = df.copy()
    merge_df  = merge_df[['post_id','post_text']].drop_duplicates()
    merge_df['post_sentiment'] = merge_df['post_text'].apply(compound_sentiment_func)
    return merge_df


def comment_sentiment_func(df):
    ### Functions for creating VADER sentiment for posts and comments
    ### based on 682 Social Media Assignment4 and this blog
    ### https://towardsdatascience.com/sentimental-analysis-using-vader-a3415fef7664

    ### set the sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    
    ### Function to extract only the 'compound' or aggregate sentiment
    def compound_sentiment_func(text):
        all_sentiment = sid.polarity_scores(text)
        compound_sentiment = all_sentiment['compound']
        return compound_sentiment

    merge_df = df.copy()
    merge_df  = merge_df[['comment_id','comment_text']].drop_duplicates()
    merge_df['comment_sentiment'] = merge_df['comment_text'].apply(compound_sentiment_func)

    df = df.copy()
    df = df.merge(merge_df, how='left', on=['comment_id', 'comment_text'])
    avg_df = df.copy().groupby(['batch_id','post_id'])[['comment_sentiment']].mean().reset_index()
    avg_df = avg_df.rename({'comment_sentiment':'avg_comment_sentiment'}, axis=1)

    return avg_df


def post_sentence_transform_func(df):
    df = df.copy()
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def bert_array(text):
        if text == np.nan or text == "" or text == "NaN":
            print ("text is nan")
            bert_array = np.zeros(1)
        else:
            bert_array = model.encode(str(text))   
        
        return bert_array

    merge_df = df.copy()
    merge_df = merge_df[['post_id','post_text']].drop_duplicates()
    merge_df['post_sbert'] = merge_df['post_text'].apply(bert_array)

    df2 = merge_df.copy().explode('post_sbert')
    df2['number'] = df2.groupby('post_id').cumcount()+1
    df2['number'] = df2['number'].astype(str).apply(lambda x: x.zfill(3))
    df2['label'] = 'post_sbert_'
    df2['col_name'] = df2['label'] + df2['number']
    df2.drop(['number','label'], axis=1,inplace=True)
    df2 = df2.pivot(index=['post_id','post_text'],columns='col_name',values='post_sbert').reset_index()
    #display(df2)
    df2.drop(['post_text'], axis=1,inplace=True)
    
    return df2


def comment_sentence_transform_func(df):
    df = df.copy()
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def bert_array(text):
        if text == np.nan or text == "" or text == "NaN":
            print ("text is nan")
            bert_array = np.zeros(1)
        else:
            bert_array = model.encode(str(text))   
        
        return bert_array

    merge_df = df.copy()
    merge_df  = merge_df[['comment_id','comment_text']].drop_duplicates()
    merge_df['comment_sbert'] = merge_df['comment_text'].apply(bert_array)
    
    df2 = merge_df.copy().explode('comment_sbert')
    df2['number'] = df2.groupby('comment_id').cumcount()+1
    df2['number'] = df2['number'].astype(str).apply(lambda x: x.zfill(3))
    df2['label'] = 'avg_comment_sbert_'
    df2['col_name'] = df2['label'] + df2['number']
    df2.drop(['number','label'], axis=1,inplace=True)
    
    cols_to_avg = list(df2.col_name.unique())
    df2 = df2.pivot(index=['comment_id','comment_text'],columns='col_name',values='comment_sbert').reset_index()
    
    avg_df = df.copy()
    avg_df = avg_df.merge(df2, how='left', on=['comment_id','comment_text'])
    avg_df = avg_df.copy().groupby(['batch_id','post_id'])[cols_to_avg].mean().reset_index()
    return avg_df


def post_profanity_removal(df):
    #import spacy
    #nlp = spacy.load('en')
    #pf = ProfanityFilter(nlps={'en': nlp}) # set the filter
    pf = ProfanityFilter() # set the filter
    
    def filter_profanity_func(text):
        return pf.censor(text)
    
    df = df.copy()

    df2 = df.copy()[['post_id','post_text']].drop_duplicates()
    df2['new_col'] = df2['post_text'].apply(filter_profanity_func) 

    output_df = df.copy()
    output_df = output_df.merge(df2, how='left',on=['post_id','post_text']).rename({'post_text':'delete_col'},axis=1).drop(['delete_col'],axis=1).rename({'new_col':'post_text'},axis=1) #merge and rearrange
    cols = list(df.columns)
    output_df = output_df.copy()[cols] #puts columns back in order
    return output_df


def comment_profanity_removal(df):
    
    pf = ProfanityFilter() # set the filter
    
    def profanity_filter(text):
        return pf.censor(text)
    
    df = df.copy()

    df2 = df.copy()[['comment_id','comment_text']].drop_duplicates()
    df2['new_col'] = df2['comment_text'].apply(profanity_filter) 

    output_df = df.copy()
    output_df = output_df.merge(df2, how='left',on=['comment_id','comment_text']).rename({'comment_text':'delete_col'},axis=1).drop(['delete_col'],axis=1).rename({'new_col':'comment_text'},axis=1) #merge and rearrange
    cols = list(df.columns) #puts columns back in order
    output_df = output_df.copy()[cols] #puts columns back in order
    return output_df



def standard_scale_column(df1, df2, column_list):
    scaler = StandardScaler() #set scaler
    df1[column_list] = scaler.fit_transform(df1[column_list]) #convert X_train
    df2[column_list] = scaler.transform(df2[column_list]) # convert X_test
    return df1, df2

def generate_features_csv(post_data, comments_data, csv_filename):

    print("generating features csv [basic]...")
    ### Start by cleaning profanity from posts and comments
    post_data = post_profanity_removal(post_data.copy())
    comments_data = comment_profanity_removal(comments_data.copy())

    ### Basic feature addtions for post-only data
    feature_df = post_basic_features(post_data.copy())
    
    ### Basic feature additions for comments (avg's)
    feature_df = feature_df.merge(comment_basic_features(comments_data.copy()) , how='left',on=['batch_id','post_id'])

    
    print("generating features csv [Vader]...")
    ### VADER sentiment for posts
    feature_df = feature_df.merge(post_sentiment_func(post_data.copy()) , how='left',on=['post_id', 'post_text'])

    ### VADER sentiment for comments (avg)
    feature_df = feature_df.merge(comment_sentiment_func(comments_data.copy()) , how='left',on=['batch_id','post_id'])

    print("generating features csv [sBERT posts]...")
    ### SBERT sentence transform for posts
    feature_df = feature_df.merge(post_sentence_transform_func(post_data.copy()) , how='left',on=['post_id'])

    print("generating features csv [sBERT comments]...")
    ### SBERT sentence transform for comments (avg's)
    feature_df = feature_df.merge(comment_sentence_transform_func(comments_data), how='left',on=['batch_id','post_id'])


    ### check for NaN's
    feature_df.replace([np.nan], 0, inplace=True) #clean NaN's formed from comments merging
    feature_df.to_csv(csv_filename, index=False)
    
    return feature_df


def feature_to_x_y(feature_df):
    df = feature_df.copy()
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

    return X, y