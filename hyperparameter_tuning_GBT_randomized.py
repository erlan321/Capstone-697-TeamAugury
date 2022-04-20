# This program was built to run an exhaustive RandomizedSearchCV on the GBDT model
# All features will be used for the models.

from importlib import import_module
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from functions.Team_Augury_feature_functions import feature_to_x_y
from scipy import stats

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
                ('categorical', categorical_transformer, categorical_features)], remainder='passthrough')

#Scoring metrics
scoring = {'acc': 'accuracy', 'f1': 'f1'}

gbc_params = {
            "clf__learning_rate":stats.uniform(0.01, 0.15),
            "clf__n_estimators":stats.randint(50, 200),
            "clf__max_depth":stats.randint(2, 8),
            "clf__min_samples_split":stats.uniform(0.01, 0.15),
            "clf__min_samples_leaf":stats.randint(1, 10),
            "clf__max_features":stats.uniform(0.1, 1),
            "clf__subsample":stats.uniform(0, 1)
            }

pipe = Pipeline(steps=[('preprocessor', preprocessor), ("clf", GradientBoostingClassifier(random_state=rnd_state))])

print("Initiating randomized grid search on GBT...")
results = RandomizedSearchCV(estimator=pipe, param_distributions=gbc_params, cv=5, scoring=scoring, refit="f1", n_jobs=-1, return_train_score=True, verbose=1, random_state=rnd_state, n_iter=250).fit(X, y)
cv_res = results
results = pd.DataFrame(results.cv_results_)
results.columns = [col.replace('param_clf__', '') for col in results.columns]
results.to_csv("saved_work/hp_tuning_RAW_GBT_rand.csv", index=False)
# clean up
it_params = ["learning_rate", "n_estimators", "max_depth", "max_features", "subsample", "min_samples_split", "min_samples_leaf"]
it_params_noC = ["n_estimators", "max_depth", "max_features", "subsample", "min_samples_split", "min_samples_leaf"]
tot_params = it_params + ["mean_train_acc", "mean_test_acc", "mean_train_f1", "mean_test_f1"]
results = results[tot_params]
results = results.melt(id_vars=it_params, var_name= "Score", value_name='Result')
results["model_param"] =results["learning_rate"].astype(str) + " / " + results["n_estimators"].astype(str) + " / " + results["max_depth"].astype(str) + " / " + results["max_features"].astype(str) + " / " + results["subsample"].astype(str) + " / " + results["min_samples_split"].astype(str) + " / " + results["min_samples_leaf"].astype(str)
results['type'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_train_f1')), "train", "test")
results['Score'] = np.where(((results['Score']== 'mean_train_acc') | (results['Score']== 'mean_test_acc')), "Accuracy", "F1")
results = results[["Result", "model_param", "Score", "type", "learning_rate", "n_estimators", "max_depth", "max_features", "subsample", "min_samples_split", "min_samples_leaf"]]
#remove overfitting
ovr_fit = results[(results["Result"]>=0.99) & (results["type"]=="train")] #with small dataset, better err on the side of caution
ovr_fit = list(set(ovr_fit['model_param']))
#remove dismal f1 score (<0.2)
undr_fit = results[(results["Result"]<0.2) & (results["Score"]=="F1")]
undr_fit = list(set(undr_fit['model_param']))
remove_params = list(undr_fit + ovr_fit)
results = results.drop(results[results.model_param.isin(remove_params)].index)
#save
results.to_csv("saved_work/hp_tuning_GBT_rand.csv", index=False)
print(cv_res.best_params_)
print(cv_res.best_score_)