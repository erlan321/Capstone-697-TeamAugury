# This program was built to run an exhaustive GridSearchCV on the SVC model
# All features will be used for the models.

from inspect import classify_class_attrs
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from functions.Team_Augury_feature_functions import feature_to_x_y

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


# Linear  
print("Initiating randomized grid search on linear SVC...")
pipe = Pipeline(steps=[('preprocessor', preprocessor), ("clf", SVC(class_weight="balanced", max_iter=100000, cache_size=2000, random_state=rnd_state))])

parameters = {"clf__C":np.logspace(-5, 15, num=21, base=2), 
            "clf__kernel":["linear"],
            'clf__gamma': np.logspace(-15, 3, num=19, base=2)}
results = GridSearchCV(estimator=pipe, param_grid=parameters, cv=5, scoring=scoring, refit="f1", n_jobs=-1, return_train_score=True, verbose=1).fit(X, y)                                   
results = pd.DataFrame(results.cv_results_)
results.columns = [col.replace('param_clf__', '') for col in results.columns]
results.to_csv("saved_work/hp_tuning_RAW_SVC_linear.csv", index=False)
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
results.to_csv("saved_work/hp_tuning_SVC_linear.csv", index=False)


# rbf  
print("Initiating randomized grid search on rbf SVC...")
pipe = Pipeline(steps=[('preprocessor', preprocessor), ("clf", SVC(class_weight="balanced", max_iter=-1, cache_size=2000, random_state=rnd_state))])
parameters = {"clf__C":np.logspace(-5, 15, num=21, base=2), 
            "clf__kernel":["rbf"],
            'clf__gamma': np.logspace(-15, 3, num=19, base=2)}
results = GridSearchCV(estimator=pipe, param_grid=parameters, cv=5, scoring=scoring, refit="f1", n_jobs=-1, return_train_score=True, verbose=1).fit(X, y)                                   
results = pd.DataFrame(results.cv_results_)
results.columns = [col.replace('param_clf__', '') for col in results.columns]
results.to_csv("saved_work/hp_tuning_RAW_SVC_rbf.csv", index=False)
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
results.to_csv("saved_work/hp_tuning_SVC_rbf.csv", index=False)


# sigmoid  
print("Initiating randomized grid search on sigmoid SVC...")
pipe = Pipeline(steps=[('preprocessor', preprocessor), ("clf", SVC(class_weight="balanced", max_iter=-1, cache_size=2000, random_state=rnd_state))])
parameters = {"clf__C":np.logspace(-5, 15, num=21, base=2), 
            "clf__kernel":["sigmoid"],
            'clf__gamma': np.logspace(-15, 3, num=19, base=2)}
results = GridSearchCV(estimator=pipe, param_grid=parameters, cv=5, scoring=scoring, refit="f1", n_jobs=-1, return_train_score=True, verbose=1).fit(X, y)                                   
results = pd.DataFrame(results.cv_results_)
results.columns = [col.replace('param_clf__', '') for col in results.columns]
results.to_csv("saved_work/hp_tuning_RAW_SVC_sigmoid.csv", index=False)
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
results.to_csv("saved_work/hp_tuning_SVC_sigmoid.csv", index=False) 


# poly  
print("Initiating randomized grid search on poly SVC...")
pipe = Pipeline(steps=[('preprocessor', preprocessor), ("clf", SVC(class_weight="balanced", max_iter=100000, cache_size=2000, random_state=rnd_state))])
parameters = {"clf__C":np.logspace(-5, 15, num=21, base=2), 
            "clf__kernel":["poly"],
            "clf__degree": [3, 4, 5],
            'clf__gamma': np.logspace(-15, 3, num=19, base=2)}
results = GridSearchCV(estimator=pipe, param_grid=parameters, cv=5, scoring=scoring, refit="f1", n_jobs=-1, return_train_score=True, verbose=1).fit(X, y)                                   
results = pd.DataFrame(results.cv_results_)
results.columns = [col.replace('param_clf__', '') for col in results.columns]
results.to_csv("saved_work/hp_tuning_RAW_SVC_poly.csv", index=False)
# clean up
it_params = ["C", "gamma", "kernel", "degree"]
it_params_noC = ["gamma", "kernel", "degree"]
tot_params = it_params + ["mean_train_acc", "mean_test_acc", "mean_train_f1", "mean_test_f1"]
results = results[tot_params]
results = results.melt(id_vars=it_params, var_name= "Score", value_name='Result')
results["model_param"] =results["C"].astype(str) + " / " + results["gamma"].astype(str) + " / " + results["degree"].astype(str) + " / " + results["kernel"].astype(str)
results['type'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_train_f1')), "train", "test")
results['Score'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_test_acc')), "Accuracy", "F1")
results = results[["Result", "model_param", "Score", "type", "C", "gamma", "degree", "kernel"]]
#remove overfitting
ovr_fit = results[(results["Result"]>=0.99) & (results["type"]=="train")]
ovr_fit = list(set(ovr_fit['model_param']))
#remove dismal f1 score (<0.2)
undr_fit = results[(results["Result"]<0.2) & (results["Score"]=="F1")]
undr_fit = list(set(undr_fit['model_param']))
remove_params = list(undr_fit + ovr_fit)
results = results.drop(results[results.model_param.isin(remove_params)].index)
#save
results.to_csv("saved_work/hp_tuning_SVC_poly.csv", index=False)