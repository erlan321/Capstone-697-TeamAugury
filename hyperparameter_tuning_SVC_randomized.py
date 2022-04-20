# This program was built to run an alternate RandomizedSearchCV on the SVC model
# All features will be used for the models.

from inspect import classify_class_attrs
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from functions.Team_Augury_feature_functions import feature_to_x_y
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from scipy import stats

import warnings
warnings.filterwarnings('ignore')  ### BE CAREFUL USING THIS :) Supressing the warning that some LR are NaN

rnd_state = 42
# This file should be generated from the baseline features csv file
print("Loading features...")
feature_df = pd.read_csv("saved_work/backup_features_data.csv")

X, y = feature_to_x_y(feature_df)

numeric_features = ['number_comments_vs_hrs','post_author_karma', 'avg_comment_upvotes_vs_hrs','avg_comment_author_karma']
categorical_features = ['time_hour', 'day_of_week']


numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', categories='auto'))])

preprocessor = ColumnTransformer(
            transformers=[
                ('numerical', numeric_transformer, numeric_features),
                ('categorical', categorical_transformer, categorical_features)], remainder="passthrough")


#Scoring metrics
scoring = {'acc': 'accuracy', 'f1': 'f1'}

# rbf  
print("Initiating randomized grid search on rbf SVC...")
pipe = Pipeline(steps=[('preprocessor', preprocessor), ("clf", SVC(class_weight="balanced", max_iter=-1, cache_size=2000, tol=0.01, random_state=rnd_state))])


parameters = {"clf__C":stats.loguniform(2**-5, 2**15), 
            "clf__kernel":["rbf", "linear", "poly", "sigmoid"],
            'clf__gamma': stats.loguniform(2**-15, 2**3)}


results = RandomizedSearchCV(estimator=pipe, param_distributions=parameters, cv=5, scoring=scoring, refit="f1", n_jobs=-1, return_train_score=True, verbose=2, random_state=rnd_state, n_iter=250).fit(X, y)
cv_res = results
results = pd.DataFrame(results.cv_results_)
results.columns = [col.replace('param_clf__', '') for col in results.columns]
results.to_csv("saved_work/hp_tuning_RAW_SVC_rand.csv", index=False)
# clean up
it_params = ["C", "gamma", "kernel"]
it_params_noC = ["gamma", "kernel"]
tot_params = it_params + ["mean_train_acc", "mean_test_acc", "mean_train_f1", "mean_test_f1"]
results = results[tot_params]
results = results.melt(id_vars=it_params, var_name= "Score", value_name='Result')
results["model_param"] =results["C"].astype(str) + " / " + results["gamma"].astype(str)
results['type'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_train_f1')), "train", "test")
results['Score'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_test_acc')), "Accuracy", "F1")
results = results[["Result", "model_param", "Score", "type", "C", "gamma", "kernel"]]
#remove overfitting
ovr_fit = results[(results["Result"]>=0.99) & (results["type"]=="train")]
ovr_fit = list(set(ovr_fit['model_param']))
#remove dismal f1 score (<0.2)
undr_fit = results[(results["Result"]<0.2) & (results["Score"]=="F1")]
undr_fit = list(set(undr_fit['model_param']))
remove_params = list(undr_fit + ovr_fit)
results = results.drop(results[results.model_param.isin(remove_params)].index)
#save
results.to_csv("saved_work/hp_tuning_SVC_rand.csv", index=False)
print(cv_res.best_params_)
print(cv_res.best_score_)