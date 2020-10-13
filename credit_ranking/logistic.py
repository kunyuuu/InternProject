import os
import numpy as np
import pandas as pd
from  scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\longcredit\Projects\InternProject\credit_ranking")
pd.set_option('display.max_columns', None)

#data cleaning and preprocessing
accepts = pd.read_csv('accepts.csv', skipinitialspace = True)
accepts = accepts.dropna(axis = 0, how = 'any')

#cross table
cross_table = pd.crosstab(accepts.used_ind,accepts.bad_ind, margins=True)
cross_table

