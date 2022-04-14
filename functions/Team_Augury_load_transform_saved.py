# Function to load the saved features and transform them as standard process

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_and_preprocess(apply_transformer = False ):
    '''Loads a stored version of pre-processed data and returns an output identical to what is fed to clf
    Takes one argument apply_transfomer, default = False resulting in 776 features, if True it applies pipeline including 
    one hot which results in 805 features '''
    
    rnd_state = 42
    # This file should be generated from the baseline features csv file
    print("Loading features...")
    feature_df = pd.read_csv("saved_work/backup_features_data.csv")

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

    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())])

    categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', categories='auto'))])

    preprocessor = ColumnTransformer(
                transformers=[
                    ('numerical', numeric_transformer, numeric_features),
                    ('categorical', categorical_transformer, categorical_features)],
                    remainder='passthrough')
    
    if apply_transformer:
        X = preprocessor.fit_transform(X)
    
    print ('data loaded and pre_processed')

    return X, y

    
