import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier


numeric_features = []
categorical_features = []
rnd_state = 42



    # Bring X and y from validation data

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', categories='auto'))])

    preprocessor = ColumnTransformer(transformers=[('numerical', numeric_transformer, numeric_features),
                    ('categorical', categorical_transformer, categorical_features)], remainder='passthrough')

    scoring = {'acc': 'accuracy', 'f1': 'f1'}

    gbc_params = {
                "clf__learning_rate":[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                "clf__n_estimators":[50,75,100,125,150,175,200, 250],
                "clf__max_depth":[3,5,8],
                "clf__max_features":["log2","sqrt"],
                "clf__subsample":[0.5,0.75,1.0],
                }

    pipe = Pipeline(steps=[('preprocessor', preprocessor), ("clf", GradientBoostingClassifier(random_state=rnd_state))])

    results = GridSearchCV(estimator=pipe, param_grid=gbc_params, cv=5, scoring=scoring, refit="f1", n_jobs=-1,
                        return_train_score=True, verbose=2).fit(X, y)   
    results = pd.DataFrame(results.cv_results_)

    # Clean up the dataframe and export to csv