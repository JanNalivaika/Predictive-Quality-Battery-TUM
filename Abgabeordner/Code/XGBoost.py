#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ## Load Data

# In[2]:


#df = pd.read_excel(r'C:\Users\Simon\Desktop\Python\Practice2_KIiPE\Datensatz_Batteriekontaktierung.xlsx', sep=';') #Datensatz von Simon
df = pd.read_excel(r'C:\Users\Simon\Desktop\Python\Practice2_KIiPE\S1_DN_relabeled.xlsx', sep=';') #Datensatz für Vergleichbarkeit
#del df['LWMID_1'] 
#del df['LWMID_2']
df2 = pd.read_excel(r'C:\Users\Simon\Desktop\Python\Practice2_KIiPE\s2dn.xlsx', sep=';', header=None)
df.head()


# In[3]:


##Datensatz Simon
#y = df.iloc[:,2] #Label
#X = df.iloc[:,4:116] #Only Sensor1
#X2 = df.iloc[:,228:340] #Only Sensor2
#X1 = df.iloc[:,4:228] #Only Sensor1 & Sensor1dn
#X1 = df.iloc[:,1:4] #Only Sensor1dn Labels
#X = df.iloc[:,116:228] #Only Sensor1dn
#X2 = df2 #Only Sensor2dn (MATLAB)
#X = X1.merge(X2, how='inner', left_index=True, right_index=True) #Sensor1dn mit labels

# X = df.iloc[:,4:340] #All
# X = X.merge(df2, how='inner', left_index=True, right_index=True) #Sensor1dn & Sensor2dn mit labels


# In[4]:


##Datensatz für Vergleichbarkeit
y = df.iloc[:,0] #Label
X = df.iloc[:,3:115] #Features

##Lubricant
#y_wd40 = df.iloc[:,1]
#y_gleitmo = df.iloc[:,2]
#y_lubricant = y_wd40 + y_gleitmo
#y = y_lubricant


# ## Statistical Metrics

# In[5]:


## MEAN:
X_mean = X.mean(axis=1)
#X = pd.concat([X_mean, X_mean], axis=1) #XGBoost Minimum Dimension = 2!

##STD:
X_std = X.std(axis=1)
#X = pd.concat([X_std, X_std], axis=1) #XGBoost Minimum Dimension = 2!

##MEAN & STD:
X = pd.concat([X_mean, X_std], axis=1) #XGBoost Minimum Dimension = 2!

##RMS:
#rms = []
#for j in range(len(X)):
#    x_col = X.iloc[j,:]
#    rms_t = np.sqrt(np.mean(x_col**2))
#    rms.append(rms_t)
#
#df_rms = pd.DataFrame(rms) 
#X = pd.concat([df_rms, df_rms], axis=1, keys=['test', 'test1']) #XGBoost Minimum Dimension = 2!


# ## Train-/Test split

# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=None, random_state=None) #stratify?
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, train_size=None, random_state=None)


# ## PCA

# In[7]:


#pca = PCA(.95)
#pca.fit(X_train)
#pca.n_components_
#X_train = pca.transform(X_train)
#X_test = pca.transform(X_test)


# ## Classification - fit & predict

# In[8]:


xgbc = xgb.XGBClassifier(n_estimators=600,
                        learning_rate=0.1,
                        subsample=0.6,
                        min_child_weight=1,
                        max_depth= 4,
                        gamma= 1,
                        colsample_bytree= 0.8)


# In[9]:


xgbc.fit(X_train, y_train)


# In[10]:


start = time.time()
pred = xgbc.predict(X_test)
end = time.time()


# In[11]:


accuracy_score(y_test, pred)


# In[12]:


print("Time for prediction in s: ", end - start)


# ## Cross Validation

# In[13]:


xgbc = xgb.XGBClassifier()
kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(xgbc, X, y, cv=kfold)


# In[14]:


print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# ## Hyperparameters - Non Exhaustive Grid Search

# In[15]:


#params = {
#        'min_child_weight': [1, 5, 10],
#        'gamma': [0.7, 1, 1.5, 2, 5],
#        'subsample': [0.4, 0.6, 0.8],
#        'learning_rate': [0.01, 0.1],
#        'colsample_bytree': [0.8, 1.0],
#        'max_depth': [3, 4, 5]
#        }


# In[16]:


#xgbc = xgb.XGBClassifier(n_estimators=600, silent=True)


# In[17]:


#folds = 5
#param_comb = 20

#skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

#random_search = RandomizedSearchCV(xgbc, param_distributions=params, n_iter=param_comb, scoring='roc_auc', cv=skf.split(X_train,y_train), verbose=3, random_state=1001)

#random_search.fit(X_train, y_train)


# In[18]:


#print(random_search.best_params_)


# In[19]:


#print(random_search.best_score_ * 2 - 1)


# ## False Positive/Negative

# In[20]:


diff = pred - y_test
correct = 0
fp = 0
fn = 0
for j in diff:
    if j == 0:
        correct += 1
    if j == 1:
        fp += 1
    if j == -1:
        fn += 1


# In[21]:


print("#Correct: ", correct)
print("#False Positive: ", fp)
print("#False Negative: ", fn)


# In[22]:


print("Share False Positive: ", fp*100/(correct+fp+fn))
print("Share False Negative: ", fn*100/(correct+fp+fn))


# ## Appendix: Standarddeviation (Not Needed! See Accuracy CV!)

# In[23]:


#score_array = []
#for train_index, test_index in kfold.split(X, y):
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]
#    
#    xgbc.fit(X_train, y_train)
#    pred_t = xgbc.predict(X_test)
#    score = accuracy_score(y_test, pred_t)
#    score_array.append(score)
#
#standard_dev = np.std(score_array)
#print("Standard deviation: ", standard_dev)


# In[ ]:




