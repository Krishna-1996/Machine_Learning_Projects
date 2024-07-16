# 1. Import necessary Libraries:
import pandas as pd
import numpy as np
from matplotlib.pyplot import plt
import seaborn as sns
from sklearn import preprocessing
import tensorflow as tf
import keras


df = pd.read_csv('E:/Machine Learning/Machine Learning/Machine_Learning_Projects/Machine_Learning_Projects/1. IPL Score Prediction/ipl_data.csv')
print(df.head())