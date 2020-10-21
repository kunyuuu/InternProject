import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
from woe import WoE
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,recall_score,classification_report
from sklearn.metrics import roc_curve, auc


os.chdir(r"C:\Users\longcredit\Projects\InternProject\credit_ranking")

################################################################
### data preprocessing                                       ###
### 1. split variables                                       ###
### 2. drop none, nomalization                               ###
### 3. knn for reject.csv to get bad_int prediction          ###
### 4. modify the variables and data cleaning                ###
### 5. woe/iv transformtion                                  ###
################################################################

accepts = pd.read_csv('accepts.csv')
rejects = pd.read_csv('rejects.csv')
accepts_x = accepts[["tot_derog","age_oldest_tr","rev_util","fico_score","ltv"]]
accepts_y = accepts['bad_ind']
rejects_x = rejects[["tot_derog","age_oldest_tr","rev_util","fico_score","ltv"]]

# define the fill-none function
def Myfillna_median(df):
    for i in df.columns:
        median = df[i].median()
        df[i].fillna(value=median, inplace=True)
    return df
accepts_x_filled=Myfillna_median(accepts_x)
rejects_x_filled=Myfillna_median(rejects_x)

#nomalization
accepts_x_norm = pd.DataFrame(Normalizer().fit_transform(accepts_x_filled))
accepts_x_norm.columns = accepts_x_filled.columns
rejects_x_norm = pd.DataFrame(Normalizer().fit_transform(rejects_x_filled))
rejects_x_norm.columns = rejects_x_filled.columns

#use KNN model to predict 'bad_int' in reject.cvs
neigh = KNeighborsClassifier(n_neighbors=5, weights='distance')
neigh.fit(accepts_x_norm, accepts_y) 
rejects['bad_ind'] = neigh.predict(rejects_x_norm)

#combain the two dataset
rejects_res = rejects[rejects['bad_ind'] == 0].sample(1340)
rejects_res = pd.concat([rejects_res, rejects[rejects['bad_ind'] == 1]], axis = 0)
data = pd.concat([accepts.iloc[:, 2:-1], rejects_res.iloc[:,1:]], axis = 0)

# convert the catagorical variable to 0/1
bankruptcy_dict = {'N':0, 'Y':1}
data.bankruptcy_ind = data.bankruptcy_ind.map(bankruptcy_dict)

# drop outliers
# need apply to every countinues variables
year_min = data.vehicle_year.quantile(0.1)
year_max = data.vehicle_year.quantile(0.99)
data.vehicle_year = data.vehicle_year.map(lambda x: year_min if x <= year_min else x)
data.vehicle_year = data.vehicle_year.map(lambda x: year_max if x >= year_max else x)
data.vehicle_year = data.vehicle_year.map(lambda x: 2018 - x) #把年份改成距现在的时间

data.drop(['vehicle_make'], axis = 1, inplace = True)
data_filled=Myfillna_median(data)

X = data_filled[['age_oldest_tr', 'bankruptcy_ind', 'down_pyt', 'fico_score',
       'loan_amt', 'loan_term', 'ltv', 'msrp', 'purch_price', 'rev_util',
       'tot_derog', 'tot_income', 'tot_open_tr', 'tot_rev_debt',
       'tot_rev_line', 'tot_rev_tr', 'tot_tr', 'used_ind', 'veh_mileage',
       'vehicle_year']]
y = data_filled['bad_ind']

#random forest
clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(X,y)
importances = list(clf.feature_importances_)
importances_order = importances.copy()
importances_order.sort(reverse=True)
cols = list(X.columns)
col_top = []
for i in importances_order[:9]:
    col_top.append((i,cols[importances.index(i)]))
col = [i[1] for i in col_top]

#WoE/IV
warnings.filterwarnings("ignore")
print(data_filled.head())
iv_c = {}
for i in col:
    try:
        iv_c[i] = WoE(v_type='c').fit(data_filled[i],data_filled['bad_ind']).optimize().iv 
    except:
        print(i)
    
pd.Series(iv_c).sort_values(ascending=False)

WOE_c = data_filled[col].apply(lambda col:WoE(v_type='c',qnt_num=5).fit(col,data_filled['bad_ind']).optimize().fit_transform(col,data_filled['bad_ind']))

################################################################
### logistic regression                                      ###
### 1. get the train/test set                                ###
### 2. construct model                                       ###
### 3. do the prediction                                     ###
################################################################

X = WOE_c
y = data_filled['bad_ind']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

lr = LogisticRegression(C = 1, penalty = 'l2')
lr.fit(X_train,y_train.values.ravel())
y_pred = lr.predict(X_test.values)

#confusion matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



