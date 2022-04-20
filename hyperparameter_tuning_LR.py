# This program was built to run an exhaustive GridSearchCV the LogisticRegression model
# All features will be used for the models.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
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


pipe = Pipeline(steps=[('preprocessor', preprocessor), ("clf", LogisticRegression(random_state = rnd_state, max_iter=100000, multi_class="ovr"))])
parameters = {"clf__C":np.logspace(-5, 10, num=16, base=2),
            "clf__solver":["liblinear", "lbfgs"],
            "clf__penalty":["l1", "l2"],
            }

results = GridSearchCV(estimator=pipe, param_grid=parameters, cv=5, scoring=scoring, refit="f1", n_jobs=-1, return_train_score=True, verbose=1).fit(X, y)          
results = pd.DataFrame(results.cv_results_)
results.columns = [col.replace('param_clf__', '') for col in results.columns]
results.to_csv("saved_work/hp_tuning_RAW_LogReg.csv", index=False)
# clean up
it_params = ["C", "penalty", "solver"]
it_params_noC = ["penalty", "solver"]
tot_params = it_params + ["mean_train_acc", "mean_test_acc", "mean_train_f1", "mean_test_f1"]
results = results[tot_params]
results = results.melt(id_vars=it_params, var_name= "Score", value_name='Result')
results["model_param"] = results["penalty"] + " / " + results["solver"] + " / " + results["C"].astype(str)
results['type'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_train_f1')), "train", "test")
results['Score'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_test_acc')), "Accuracy", "F1")
results = results[["Result", "model_param", "Score", "type", "C", "penalty","solver"]]
#remove NANs
results = results[results['Result'].notna()]
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

