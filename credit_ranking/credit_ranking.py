import os
import io
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import warnings
import itertools
import dataprep.eda as eda
import graphviz
from woe import WoE
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix,recall_score,classification_report
from sklearn.metrics import roc_curve, auc

#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')
#os.chdir(r"C:\Users\longcredit\Projects\InternProject\credit_ranking")

################################################################
### data preprocessing                                       ###
### 1. split variables                                       ###
### 2. drop none, nomalization                               ###
### 3. knn for reject.csv to get bad_int prediction          ###
### 4. modify the variables and data cleaning                ###
### 5. woe/iv transformation                                 ###
################################################################

# define the fill-none function
def Myfillna_median(df):
    for i in df.columns:
        median = df[i].median()
        df[i].fillna(value=median, inplace=True)
    return df

def data_processing(accepts):
    data = accepts.dropna(axis = 0, how = 'any')

    #convert the catagorical variable to 0/1
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

    X = data_filled[['age_oldest_tr', 'bankruptcy_ind', 'fico_score',
       'loan_amt', 'loan_term', 'ltv',
       'tot_income', 'tot_open_tr',
       'tot_rev_tr', 'tot_tr', 'used_ind',
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
    for i in importances_order[:5]:
        col_top.append((i,cols[importances.index(i)]))
        col = [i[1] for i in col_top]

    #WoE/IV
    warnings.filterwarnings("ignore")
    iv_c = {}
    for i in col:
        try:
            iv_c[i] = WoE(v_type='c').fit(data_filled[i],data_filled['bad_ind']).optimize().iv 
        except:
            print(i)
    
    pd.Series(iv_c).sort_values(ascending=False)

    WOE_c = data_filled[col].apply(lambda col:WoE(v_type='c',qnt_num=5).fit(col,data_filled['bad_ind']).optimize().fit_transform(col,data_filled['bad_ind']))
    
    return WOE_c, data_filled

################################################################
### model selection                                          ###
### 1. get the train/test set                                ###
### 2. construct logistic model                              ###
### 3. construct Decison tree model                          ###
################################################################

def logistic(X_train, X_test, y_train, y_test):
    lr = LogisticRegressionCV(multi_class="ovr",fit_intercept=True,Cs=10,cv=3,penalty="l2",
                              solver="lbfgs",tol=0.01,class_weight = 'balanced')
    #lr = LogisticRegression(C = 2.0, class_weight = 'balanced')
    lr.fit(X_train,y_train)
    print("Training score:%f" % (lr.score(X_train, y_train)))
    print("Testing score:%f" % (lr.score(X_test, y_test)))
    y_pred = lr.predict(X_test)
    return y_pred

def decision_tree(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(class_weight='balanced', criterion = 'entropy')
    dt.fit(X_train,y_train)
    y_pred = dt.predict(X_test)
    print("Training score:%f" % (dt.score(X_train, y_train)))
    print("Testing score:%f" % (dt.score(X_test, y_test)))
    export_graphviz(dt,out_file='tree.dot',class_names=['1','0'],impurity=False,filled=True)
    with open('tree.dot') as f:
        dot_graph = f.read()
    dot = graphviz.Source(dot_graph)
    dot.view()
    return y_pred

################################################################
### Model Evaluation                                         ###
### 1. get confusion matrix                                  ###
### 2. compute and plot ROC/AUC                              ###
### 3. plot TP/FP curve and ks curve                         ###
################################################################

#plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def evaluation(y_test, y_pred):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test,y_pred)
    np.set_printoptions(precision=2)

    print("TPR(Recall): ", cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1]))
    print("FPR: ", cnf_matrix[1,0]/(cnf_matrix[1,0]+cnf_matrix[1,1]))
    print("Precision: ", cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[1,0]))

    #Plot non-normalized confusion matrix
    class_names = [0,1]
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
    plt.show()

    #model evaluation
    fpr,tpr,threshold = roc_curve(y_test,y_pred, drop_intermediate=False) #compute TP&FP  
    roc_auc = auc(fpr,tpr) ##compute auc
    print("AUC: ", roc_auc)
  
    plt.figure()  
    lw = 2  
    plt.figure(figsize=(10,10))  
    plt.plot(fpr, tpr, color='darkorange',  
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) #x-axis: FP, y-axis: TP
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  
    plt.xlim([0.0, 1.0])  
    plt.ylim([0.0, 1.05])  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title('Receiver operating characteristic example')  
    #plt.legend(loc="lower right")  
    plt.show()

    plt.figure()
    plt.plot(1 - threshold, tpr, label='tpr') # ks曲线要按照预测概率降序排列，所以需要1-threshold镜像
    plt.plot(1 - threshold, fpr, label='fpr')
    plt.plot(1 - threshold, tpr-fpr,label='KS')
    plt.xlabel('score')
    plt.title('KS Curve')
    plt.figure(figsize=(20,20))
    #plt.legend(loc='upper left', shadow=True, fontsize='x-large')
    plt.show()


if __name__ == "__main__":
    #data preprocessing part
    accepts = pd.read_csv('accepts.csv')
    #rejects = pd.read_csv('rejects.csv')
    X, data_filled = data_processing(accepts)
    y = data_filled['bad_ind']
    print(X)
    #eda.plot(data_filled).show_browser()
    #eda.plot_correlation(X,).show_browser()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

    #model selection part
    print("Please Select a Model: Logistic/Decision Tree\n")
    model = input("Enter Your Selection ('L' for logistic, 'D' for Desicion Tree): ")
    while(True):
        if(model.lower() == "l"):
            y_pred = logistic(X_train, X_test, y_train, y_test)
            break
        elif(model.lower() == "d"):
            y_pred = decision_tree(X_train, X_test, y_train, y_test)
            break
        else:
            print("Plese Enter a Valid model!")
            model = input("Enter Your Selection ('L' for logistic, 'D' for Desicion Tree): ")

    #model evaluation part
    evaluation(y_test, y_pred)
    
'''
	accepts_x = accepts[["tot_derog","age_oldest_tr","rev_util","fico_score","ltv"]]
    accepts_y = accepts['bad_ind']
    rejects_x = rejects[["tot_derog","age_oldest_tr","rev_util","fico_score","ltv"]]

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
'''


