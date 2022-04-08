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
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#Scoring metrics
scoring = {'acc': 'accuracy', 'f1': 'f1'}

# Set classifiers
classifiers = [
            LogisticRegression(random_state = 42),
            SVC(random_state=42),
            GradientBoostingClassifier(random_state=42)
            ]

lr_params = {"C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
             "max_iter":[100], #****** CHANGE THIS FOR FINAL RUN, 2000 seems good
             "solver":["liblinear", "lbfgs"],
             "penalty":["l1", "l2"]        
             }

svc_params = {
    "C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "kernel":["linear", "rbf", "poly", "sigmoid"],
    'gamma': [1,0.1,0.01,0.001],
    "max_iter":[100]  #****** CHANGE THIS FOR FINAL RUN >20000 seems good
}

gbc_params = {
    "learning_rate":[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "n_estimators":[100, 225, 350, 500],
    "max_depth":[3, 4, 5],
    "min_samples_split":np.linspace(0.1, 0.5, 8),
    "min_samples_leaf":np.linspace(0.1, 0.5, 8),
    "max_features":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
}

parameters = [lr_params, svc_params, gbc_params]


# Linear Regression
print("Initiating grid search on " + classifiers[0].__class__.__name__ + "...")
pipe_grid = GridSearchCV(estimator=classifiers[0], param_grid=parameters[0], cv=cv, scoring=scoring, refit="f1", n_jobs=-1, return_train_score=True, verbose=0).fit(X, y)
results = pd.DataFrame(pipe_grid.cv_results_)
results = results[results['split0_train_f1'].notna()] #remove the runs that could not run: lbfgs with l1 which don't work together
results = results[["param_C", "param_penalty", "param_solver", "mean_train_acc", "mean_test_acc", "mean_train_f1", "mean_test_f1"]]
results = results.melt(id_vars=["param_C", "param_penalty", "param_solver"], var_name= "Score", value_name='Result')
results["model_param"] = results["param_solver"] + " " + results["param_penalty"]
results['type'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_train_f1')), "train", "test")
results['Score'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_test_acc')), "Accuracy", "F1")
results["C"] = results["param_C"].astype(str)

sns.set(rc={'figure.figsize':(16,16)})
g = sns.FacetGrid(data=results, row="model_param", col="Score").map_dataframe(sns.lineplot, x="C", y="Result", hue="type") 
g.add_legend()
plt.savefig("saved_work/hp_tuning_" + classifiers[0].__class__.__name__ + ".png")
plt.clf()
results.to_csv("saved_work/hp_tuning_" + classifiers[0].__class__.__name__ + ".csv")

# SVC Classifier

print("Initiating grid search on " + classifiers[1].__class__.__name__ + "...")
pipe_grid = GridSearchCV(estimator=classifiers[1], param_grid=parameters[1], cv=cv, scoring=scoring, refit="f1", n_jobs=-1, return_train_score=True, verbose=0).fit(X, y)
results = pd.DataFrame(pipe_grid.cv_results_)
results = results[results['split0_train_f1'].notna()] #remove the runs that could not run: lbfgs with l1 which don't work together

results = results[["param_C", "param_gamma", "param_kernel", "mean_train_acc", "mean_test_acc", "mean_train_f1", "mean_test_f1"]]
results = results.melt(id_vars=["param_C", "param_gamma", "param_kernel"], var_name= "Score", value_name='Result')
results["model_param"] = results["param_kernel"] + " " + results["param_gamma"].astype(str)
results['type'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_train_f1')), "train", "test")
results['Score'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_test_acc')), "Accuracy", "F1")
results["C"] = results["param_C"].astype(str)

sns.set(rc={'figure.figsize':(16,16)})
g = sns.FacetGrid(data=results, row="model_param", col="Score").map_dataframe(sns.lineplot, x="C", y="Result", hue="type") 
g.add_legend()
plt.savefig("saved_work/hp_tuning_" + classifiers[1].__class__.__name__ + ".png")
plt.clf()
results.to_csv("saved_work/hp_tuning_" + classifiers[1].__class__.__name__ + ".csv")


# Gradient Boosting Tree Classifier
# guided by https://machinelearningmastery.com/configure-gradient-boosting-algorithm/ and it's references for tuning

print("Initiating grid search on " + classifiers[2].__class__.__name__ + "...")
pipe_grid = GridSearchCV(estimator=classifiers[2], param_grid=parameters[2], cv=cv, scoring=scoring, refit="f1", n_jobs=-1, return_train_score=True, verbose=1).fit(X, y)
results = pd.DataFrame(pipe_grid.cv_results_)
results = results[results['split0_train_f1'].notna()] #remove the runs that could not run: lbfgs with l1 which don't work together

results = results[["param_learning_rate", "param_n_estimators", "param_min_samples_split", "param_min_samples_leaf", "mean_train_acc", "mean_test_acc", "mean_train_f1", "mean_test_f1"]]
results = results.melt(id_vars=["param_learning_rate", "param_n_estimators", "param_min_samples_split", "param_min_samples_leaf"], var_name= "Score", value_name='Result')
results["model_param"] = results["param_n_estimators"].astype(str) + " " + results["param_min_samples_split"].astype(str) + " " + results["param_min_samples_leaf"].astype(str)
results['type'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_train_f1')), "train", "test")
results['Score'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_test_acc')), "Accuracy", "F1")
results["Learning Rate"] = results["param_learning_rate"].astype(str)

sns.set(rc={'figure.figsize':(16,16)})
g = sns.FacetGrid(data=results, row="model_param", col="Score").map_dataframe(sns.lineplot, x="Learning Rate", y="Result", hue="type") 
g.add_legend()
plt.savefig("saved_work/hp_tuning_" + classifiers[2].__class__.__name__ + ".png")
plt.clf()
results.to_csv("saved_work/hp_tuning_" + classifiers[2].__class__.__name__ + ".csv")

