import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

df = pd.read_csv("saved_work/backup_features_data.csv")

y_column = ["popular_hr_3"]
popular_hr_3_threshold = 10

#test popular 3 hr
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
numeric_features = []
numeric_features.extend(['number_comments_vs_hrs','post_author_karma'])
numeric_features.extend(['avg_comment_upvotes_vs_hrs','avg_comment_author_karma'])

### columns that will need a onehotencode scaler
categorical_features = X_col_post_temporal

X_columns = []
X_columns.extend(X_col_post_basic)
X_columns.extend(X_col_post_sentiment)
X_columns.extend(X_col_post_temporal)
X_columns.extend(X_col_post_sbert)
X_columns.extend(X_col_comment_basic)
X_columns.extend(X_col_comment_sentiment)
X_columns.extend(X_col_comment_sbert)

df = df[df.hours_since_created==0]

X = df[X_columns]
y = df[y_column].values.ravel()

numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', categories='auto'))])

preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)])

classifier = LogisticRegression(random_state = 42)

clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", classifier)]
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {'acc': 'accuracy', 'f1': 'f1'}

scores = cross_validate(clf, X, y, scoring=scoring, return_train_score=True, cv=cv, n_jobs=-1)

print (classifier)
acc = round(np.mean(scores["test_acc"]),3)
f1 = round(np.mean(scores["test_f1"]),3)
print ("Accuracy: {}    F1: {}".format(acc, f1))
