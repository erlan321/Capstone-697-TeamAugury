import pandas as pd
import numpy as np  
from datetime import datetime 
from functions.Team_Augury_load_transform_saved import load_and_preprocess

# import itertools
# from tqdm import tqdm

X, y = load_and_preprocess()

print ('Testing load, length of X, y:', X.shape, y.shape)