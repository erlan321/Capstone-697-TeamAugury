import pandas as pd
import numpy as np  
from datetime import datetime 
from functions.Team_Augury_load_transform_saved import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import pickle

# import itertools
# from tqdm import tqdm


X_train, y_train = load_and_preprocess()
print ('Testing load, length of X, y:', X_train.shape, y_train.shape)

#load pkl'd clf

filename = "models/GradientBoostingClassifier_vanilla_model.sav"
GB_loaded = pickle.load(open(filename, 'rb'))
result = GB_loaded.score(X_train, y_train)
print('Testing model load: Accuracy result for GBT:' , result)
