# This program was built to run an exhaustive GridSearchCV on the 3 selected models: LogisticRegression, SVC and GradientBoostingClassifier
# All features will be used for the models.

import psycopg2
from functions.Team_Augury_feature_functions import generate_features_csv
import functions.Team_Augury_SQL_func
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt

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
                ('categorical', categorical_transformer, categorical_features)])

# Moved to StratifiedKFold due to imbalanced dataset https://machinelearningmastery.com/cross-validation-for-imbalanced-classification/
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state= rnd_state)

#Scoring metrics
scoring = {'acc': 'accuracy', 'f1': 'f1'}

# Set classifiers
classifiers = [
            GradientBoostingClassifier(random_state=rnd_state)
            ]

gbc_params = {
            "learning_rate":[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
            "n_estimators":[100, 225, 350, 500]}

#1st pass
        #gbc_params = {"learning_rate":[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
        #               "n_estimators":[100, 225, 350, 500]}

#2nd pass
            #gbc_params = {"learning_rate":[0.005, 0.01, 0.02, 0.03, 0.4],
            #              "n_estimators":[50, 75, 100, 125, 150]}


    """ "max_depth":[3, 4, 5],
    "min_samples_split":np.linspace(0.1, 0.5, 8),
    "min_samples_leaf":np.linspace(0.1, 0.5, 8),
    "max_features":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
} """

parameters = gbc_params

for i in range(0, 6):
    print("Initiating grid search on " + classifiers[0].__class__.__name__ + "...")
    pipe_grid = GridSearchCV(estimator=classifiers[0], param_grid=parameters[0], cv=cv, scoring=scoring, refit="f1", n_jobs=-1, return_train_score=True, verbose=2).fit(X, y)  
    results = pd.DataFrame(pipe_grid.cv_results_)
    if i == 0: #logistic regression
        it_params = ["param_C", "param_penalty", "param_solver"] 
        it_params_noC = ["param_penalty", "param_solver"]
    elif i == 1: #svc linear
        it_params = ["param_C", "param_gamma", "param_kernel"]
        it_params_noC = ["param_gamma", "param_kernel"]
    elif i == 2: #svc rbf
        it_params = ["param_C", "param_gamma", "param_kernel"]
        it_params_noC = ["param_gamma", "param_kernel"]
    elif i == 3: #svc poly
        it_params = ["param_C", "param_gamma", "param_kernel", "param_coef0", "param_degree"]
        it_params_noC = ["param_gamma", "param_kernel", "param_coef0", "param_degree"]
    elif i == 4: #svc sigmoid
        it_params = ["param_C", "param_gamma", "param_kernel", "param_coef0", "param_degree"]
        it_params_noC = ["param_gamma", "param_kernel", "param_coef0"]



    tot_params = it_params + ["mean_train_acc", "mean_test_acc", "mean_train_f1", "mean_test_f1"]
    results = results[tot_params]
    results = results.melt(id_vars=it_params, var_name= "Score", value_name='Result')

    if i == 0 or i == 1 or i == 2:
        results["model_param"] = results[it_params_noC[0]] + " " + results[it_params_noC[1]]
    elif i == 3:
        results["model_param"] = results[it_params_noC[0]] + " " + results[it_params_noC[1]] + " " + results[it_params_noC[2]] + " " + results[it_params_noC[3]]
    elif i == 4:
        results["model_param"] = results[it_params_noC[0]] + " " + results[it_params_noC[1]] + " " + results[it_params_noC[2]]]
        
    results['type'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_train_f1')), "train", "test")
    results['Score'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_test_acc')), "Accuracy", "F1")
    results["C"] = results["param_C"].astype(str)

    sns.set(rc={'figure.figsize':(16,16)})
    g = sns.FacetGrid(data=results, row="model_param", col="Score").map_dataframe(sns.lineplot, x="C", y="Result", hue="type") 
    g.add_legend()
    g.fig.set_size_inches(18.5, 12.5)
    plt.savefig("saved_work/hp_tuning_" + str(i)+ classifiers[0].__class__.__name__ + ".png")
    plt.clf()
    results.to_csv("saved_work/hp_tuning_" + str(i) + classifiers[0].__class__.__name__ + ".csv")