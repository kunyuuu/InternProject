import os
import numpy as np
import pandas as pd
from  scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

#os.chdir(r"C:\Users\longcredit\Projects\InternProject\credit_ranking")
pd.set_option('display.max_columns', None)

'''data cleaning and preprocessing'''
accepts = pd.read_csv('accepts.csv', skipinitialspace = True)
accepts = accepts.dropna(axis = 0, how = 'any')

'''cross table'''
cross_table = pd.crosstab(accepts.bankruptcy_ind,accepts.bad_ind, margins=True)
#cross_table = pd.crosstab(accepts.used_ind,accepts.bad_ind, margins=True)
print("cross table: \n", cross_table)

def percConvert(ser):
    return ser/float(ser[-1])

#cross_table.apply(percConvert, axis=1)

'''chi squre'''
print('''chisq = %6.4f 
p-value = %6.4f
dof = %i 
expected_freq = %s'''  %stats.chi2_contingency(cross_table.iloc[:2, :2]))

'''logictic regression'''
train = accepts.sample(frac=0.7, random_state=1234).copy()
test = accepts[~ accepts.index.isin(train.index)].copy()
print(' trainning set: %i \n test set: %i' %(len(train), len(test)))

lg = smf.glm('bad_ind ~ age_oldest_tr', data=train, family=sm.families.Binomial()).fit()
print(lg.summary())


train['proba'] = lg.predict(train)
test['proba'] = lg.predict(test)
test['prediction'] = (test['proba'] > 0.3).astype('int') #threshold

pd.crosstab(test.bad_ind, test.prediction, margins=True) #confusion matrix

acc = sum(test['prediction'] == test['bad_ind']) /np.float(len(test))
print('The accurancy is %.2f' %acc)

for i in np.arange(0.02, 0.3, 0.02):
    prediction = (test['proba'] > i).astype('int')
    confusion_matrix = pd.crosstab(prediction,test.bad_ind,
                                   margins = True)
    precision = confusion_matrix.loc[0, 0] /confusion_matrix.loc['All', 0]
    recall = confusion_matrix.loc[0, 0] / confusion_matrix.loc[0, 'All']
    Specificity = confusion_matrix.loc[1, 1] /confusion_matrix.loc[1,'All']
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('threshold: %s, precision: %.2f, recall:%.2f ,Specificity:%.2f , f1_score:%.2f'%(i, precision, recall, Specificity,f1_score))

fpr_test, tpr_test, th_test = metrics.roc_curve(test.bad_ind, test.proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train.bad_ind, train.proba)

plt.figure(figsize=[3, 3])
plt.plot(fpr_test, tpr_test, 'b--')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.show()
