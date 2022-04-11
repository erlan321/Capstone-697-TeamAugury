# This program was built to run an exhaustive GridSearchCV on the first of the selected models: LogisticRegression. SVC and GradientBoostingClassifier are tuned in their specific files
# All features will be used for the models.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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
            LogisticRegression(random_state = rnd_state),
            ]

lr_params = [
            {"C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "solver":["liblinear"],
            "penalty":["l1", "l2"],
            "max_iter":[5000], #raised to reduce non-convergence errors
            "multi_class":["ovr"]},

            {"C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "solver":["lbfgs"],
            "penalty":["l2"],
            "max_iter":[5000], #raised to reduce non-convergence errors
            "multi_class":["ovr"]}
            ]


# liblinear gave the better result for the solver, especially on the F1 score
# finaly L2 penality was chosen, because L1 ramped up to over fitting the data as soon as C was greater than 0.1
# After results of the above, picked C = 0.01, while the C = 1000 gives slightly better results, keeping C lower helps with regularization
# With these settings the baseline is Accuracy = 0.789 and F1 = 0.306


pipe = Pipeline(steps=[('preprocessor', preprocessor), ("clf", LogisticRegression(random_state = rnd_state, max_iter=10000, multi_class="ovr"))])
parameters = [
            {"clf__C":np.logspace(-5, 10, num=16, base=2),
            "clf__solver":["liblinear"],
            "clf__penalty":["l1", "l2"],
            },

            {"clf__C":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "clf__solver":["lbfgs"],
            "clf__penalty":["l2"],
            }
            ]

results = GridSearchCV(estimator=pipe, param_grid=parameters, cv=5, scoring=scoring, refit="f1", n_jobs=-1, return_train_score=True, verbose=1).fit(X, y)          
results = pd.DataFrame(results.cv_results_)
results.columns = [col.replace('param_clf__', '') for col in results.columns]
results.to_csv("saved_work/hp_tuning_RAW_LogReg.csv", index=False)
# clean up
it_params = ["C", "penalty", "solver"]
it_params_noC = ["gamma", "penalty", "solver"]
tot_params = it_params + ["mean_train_acc", "mean_test_acc", "mean_train_f1", "mean_test_f1"]
results = results[tot_params]
results = results.melt(id_vars=it_params, var_name= "Score", value_name='Result')
results["model_param"] = results["penalty"] + " / " + results["solver"]
results['type'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_train_f1')), "train", "test")
results['Score'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_test_acc')), "Accuracy", "F1")
results = results[["Result", "model_param", "Score", "type", "C", "penalty","solver"]]
#remove overfitting
ovr_fit = results[(results["Result"]>=0.99) & (results["type"]=="train")]
ovr_fit = list(set(ovr_fit['model_param']))
#remove dismal f1 score (<0.2)
undr_fit = results[(results["Result"]<0.2) & (results["Score"]=="F1")]
undr_fit = list(set(undr_fit['model_param']))
remove_params = list(undr_fit + ovr_fit)
results = results.drop(results[results.model_param.isin(remove_params)].index)
#save
results.to_csv("saved_work/hp_tuning_LogReg.csv", index=False)


"""


    print("Initiating grid search on " + classifiers[i].__class__.__name__ + "...")
    pipe_grid = GridSearchCV(estimator=classifiers[i], param_grid=parameters[i], cv=cv, scoring=scoring, refit="f1", n_jobs=-1, return_train_score=True, verbose=2).fit(X, y)  
    results = pd.DataFrame(pipe_grid.cv_results_)
    results.to_csv("saved_work/hp_tuning_RAW" + str(i) + classifiers[i].__class__.__name__ + ".csv")

    if i == 0: #logistic regression
        it_params = ["param_C", "param_penalty", "param_solver"] 
        it_params_noC = ["param_penalty", "param_solver"]

    tot_params = it_params + ["mean_train_acc", "mean_test_acc", "mean_train_f1", "mean_test_f1"]
    results = results[tot_params]
    results = results.melt(id_vars=it_params, var_name= "Score", value_name='Result')

    if i == 0 :
        results["model_param"] = results[it_params_noC[0]] + " " + results[it_params_noC[1]]
        
    results['type'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_train_f1')), "train", "test")
    results['Score'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_test_acc')), "Accuracy", "F1")
    results["C"] = results["param_C"].astype(str)

    sns.set(rc={'figure.figsize':(16,16)})
    g = sns.FacetGrid(data=results, row="model_param", col="Score").map_dataframe(sns.lineplot, x="C", y="Result", hue="type") 
    g.add_legend()
    g.fig.set_size_inches(18.5, 12.5)
    plt.savefig("saved_work/hp_tuning_" + str(i)+ classifiers[i].__class__.__name__ + ".png")
    plt.clf()
    results.to_csv("saved_work/hp_tuning_" + str(i) + classifiers[i].__class__.__name__ + ".csv")"""