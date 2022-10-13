import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

def get_df(file, header=None):
    df = pd.read_csv(file)
    return df