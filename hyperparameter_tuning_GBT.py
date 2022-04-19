# This program was built to run an exhaustive GridSearchCV on the third of the selected models: GradientBoostingClassifier. LogisticRegression and SVC are tuned in their specific files
# All features will be used for the models.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

import warnings
warnings.filterwarnings('ignore')  ### BE CAREFUL USING THIS :) Supressing the warning that some LR are NaN

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

#print("lenght y: ", len(y)) #1674
#print("sum y: ", sum(y)) #367

numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', categories='auto'))])

preprocessor = ColumnTransformer(
            transformers=[
                ('numerical', numeric_transformer, numeric_features),
                ('categorical', categorical_transformer, categorical_features)], remainder='passthrough')

# Moved to StratifiedKFold due to imbalanced dataset https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/

#Scoring metrics
scoring = {'acc': 'accuracy', 'f1': 'f1'}

""" gbc_params = {
            "clf__learning_rate":stats.uniform(0.01, 0.15),
            "clf__n_estimators":stats.randint(50, 200),
            "clf__max_depth":stats.randint(2, 8),
            "clf__min_samples_split":stats.uniform(0.01, 0.15),
            "clf__min_samples_leaf":stats.randint(1, 10),
            "clf__max_features":stats.uniform(0.1, 1)#,
            #"clf__subsample":stats.uniform(0.1, 1)
            } """

gbc_params = {
            "clf__learning_rate":[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
            "clf__n_estimators":[50,75,100,125,150,175,200, 250],
            "clf__max_depth":[3,5,8],
            "clf__max_features":["log2","sqrt"],
            "clf__subsample":[0.5,0.75,1.0],
            }

pipe = Pipeline(steps=[('preprocessor', preprocessor), ("clf", GradientBoostingClassifier(random_state=rnd_state))])

print("Initiating randomized grid search on GBT...")
#results = RandomizedSearchCV(estimator=pipe, param_distributions=gbc_params, cv=5, scoring=scoring, refit="f1", n_jobs=-1, return_train_score=True, verbose=1, random_state=rnd_state, n_iter=200).fit(X, y)
results = GridSearchCV(estimator=pipe, param_grid=gbc_params, cv=5, scoring=scoring, refit="f1", n_jobs=-1, return_train_score=True, verbose=1).fit(X, y)   
results = pd.DataFrame(results.cv_results_)
results.columns = [col.replace('param_clf__', '') for col in results.columns]
results.to_csv("saved_work/hp_tuning_RAW_GBT.csv", index=False)
# clean up
it_params = ["learning_rate", "n_estimators", "max_depth", "max_features", "subsample"]
it_params_noC = ["n_estimators", "max_depth", "max_features", "subsample"]
tot_params = it_params + ["mean_train_acc", "mean_test_acc", "mean_train_f1", "mean_test_f1"]
results = results[tot_params]
results = results.melt(id_vars=it_params, var_name= "Score", value_name='Result')
results["model_param"] =results["learning_rate"].astype(str) + " / " + results["n_estimators"].astype(str) + " / " + results["max_depth"].astype(str) + " / " + results["max_features"].astype(str) + " / " + results["subsample"].astype(str)
results['type'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_train_f1')), "train", "test")
results['Score'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_test_acc')), "Accuracy", "F1")
results = results[["Result", "model_param", "Score", "type", "learning_rate", "n_estimators", "max_depth", "max_features", "subsample"]]
#remove overfitting
ovr_fit = results[(results["Result"]>=0.99) & (results["type"]=="train")] #with small dataset, better err on the side of caution
ovr_fit = list(set(ovr_fit['model_param']))
#remove dismal f1 score (<0.2)
undr_fit = results[(results["Result"]<0.2) & (results["Score"]=="F1")]
undr_fit = list(set(undr_fit['model_param']))
remove_params = list(undr_fit + ovr_fit)
results = results.drop(results[results.model_param.isin(remove_params)].index)
#save
results.to_csv("saved_work/hp_tuning_GBT.csv", index=False)