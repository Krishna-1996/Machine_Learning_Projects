# Database: D:/MSc. Project DeepFake Detection Datasets/Celeb-DF-v1

# %%

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import random
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# %%
print("Hello Wolrd")
# %%
