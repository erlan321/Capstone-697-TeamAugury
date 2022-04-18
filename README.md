# Capstone-697-TeamAugury
Capstone Project for SIADS 697-698

EDA
* EDA.ipynb : contains most of the retained EDA analyses
* dow.ipynb : quick verification on day of week / hour of day correlation
* feat_imp.ipynb : feature importance calculation based on models
* svm_feat_imp.ipynb: feature importance of SVM model

Supervised Learning Pipeline
* baseline_models.py : calculates the baseline models results (both train/test) for model selection in the Hyperparameter tuning phase
* hyperparameter_tuning_* : python files to generate the csv results of the gridsearched cross-validation per model

Supporting functions (Team_Augury_...)
* blog_praw_functions.py: functions to be used in the streamlit blog post
* feature_functions.py: functions for feature extraction from raw SQL exports
* Iterate.py: function for the baseline_models.py iteration
* load_transform_saved.py: loads a stored version of pre-processed data and returns an output identical to what is fed to clf
* SQL_func.py: returns raw data from SQL using timestamps as entrants

Final
* final_model.py : generates the final model as a pickled file based on the hyperparameter tuning
* final_results.py : featurizes the unseen data and predicts wit the final model reporting accuracy and f1 scores

Miscellaneous
* vanilla_models_pkl.py: creates pickled files of the vanilla (non-hpt) models

Folders
* blog_assets: files for the streamlit blog post
* functions: see above
* models: saved trained models
* saved_work: storage for csv files