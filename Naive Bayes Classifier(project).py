#!/usr/bin/env python
# coding: utf-8

# In[219]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
get_ipython().run_line_magic('matplotlib', 'inline')


# In[220]:


import warnings
warnings.filterwarnings("ignore")


# In[221]:


df=pd.read_csv(r"D:\NIT\JANUARY\8 JAN(NAIVE BAYES ALGORITHM)\8th\project\adult.csv")


# In[222]:


df


# In[223]:


df.shape


# In[224]:


df.head()


# In[225]:


df.info()


# In[226]:


df.columns


# In[227]:


categorical_columns = df.select_dtypes(include=['object']).columns


# In[228]:


categorical_columns


# In[229]:


categorical = [var for var in df.columns if df[var].dtype=='O']


# In[230]:


print(len(categorical))
print("The categorical variables are :\n",categorical)


# In[231]:


df[categorical].head()


# In[232]:


df[categorical].isnull().sum()


# In[233]:


for var in categorical:
    print(df[var].value_counts())


# In[234]:


# view frequency distribution of categorical variables

for var in categorical: 
    
    print(df[var].value_counts()/np.float32(len(df)))


# In[235]:


df.workclass.unique()


# In[236]:


df.workclass.nunique()


# In[237]:


df.workclass.value_counts()


# In[238]:


for var in categorical: 
    print(df[var].replace('?', np.NaN, inplace=True))


# In[239]:


df.workclass.value_counts()


# In[240]:


df.occupation.unique()


# In[241]:


df[categorical].isnull().sum()


# In[242]:


# check for cardinality in categorical variables

for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')


# In[243]:


numerical_columns = df.select_dtypes(include=['number']).columns


# In[244]:


numerical_columns


# In[245]:


numerical = [var for var in df.columns if df[var].dtype!='O']
print(len(numerical))
print("The numerical variables are :\n",numerical)


# In[246]:


df[numerical].head()


# In[247]:


df[numerical].isnull().sum()


# In[248]:


X = df.drop(['income'], axis=1)

y = df['income']


# In[249]:


X.head(2)


# In[250]:


y.head(2)


# In[251]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# In[252]:


X_train, X_test, y_train, y_test


# In[253]:


X_train.shape, X_test.shape


# In[254]:


y_train.shape, y_test.shape


# In[255]:


X_train.dtypes


# In[256]:


categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical


# In[257]:


numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical


# In[258]:


X_train[categorical].isnull().sum()


# In[259]:


X_train


# In[260]:


X_train[categorical].isnull().mean()


# In[261]:


X_test[categorical].isnull().sum()


# In[262]:


y_train.isnull().sum()


# In[263]:


y_test.isnull().sum()


# In[264]:


X_train[categorical].columns



# In[265]:


X_train[numerical].columns


# In[266]:


print(X_train.columns)


# In[267]:


### Impute missing values with mode
for col in ["workclass","occupation","native.country"]:
    X_train[col].fillna(X_train[col].mode()[0],inplace=True)


# In[268]:


X_train.isnull().sum()


# In[269]:


# check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()


# In[270]:


### Impute missing values with mode
for col in ["workclass","occupation","native.country"]:
    X_test[col].fillna(X_test[col].mode()[0],inplace=True)


# In[271]:


X_test[categorical].isnull().sum()


# In[272]:


categorical


# In[273]:


X_train[categorical].head(3)


# In[274]:


pip install category_encoders


# In[275]:


import category_encoders as ce


# In[276]:


encoder = ce.OneHotEncoder(['workclass', 'education', 'marital_status', 'occupation', 'relationship', 
                                 'race', 'sex', 'native_country'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)


# In[ ]:





# In[ ]:





# In[205]:


'''from sklearn.preprocessing import LabelEncoder
categorical_features = categorical

for feature in categorical_features:
    le = LabelEncoder()
    X_train[feature] = le.fit_transform(X_train[feature])
    X_test[feature] = le.transform(X_test[feature])'''


# In[277]:


X_train.head(3)


# In[278]:


X_test.head(3)


# In[279]:


categorical


# In[280]:


X_train.shape


# In[281]:


X_test.head()


# In[282]:


X_test.shape


# In[283]:


X_train.columns


# In[284]:


from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# In[285]:


X_train = pd.DataFrame(X_train)


# In[286]:


X_test = pd.DataFrame(X_test)


# In[287]:


X_train


# In[288]:


X_test


# In[289]:


X_train.head()


# In[290]:


X_train[:5]


# In[296]:


# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB


# instantiate the model
gnb = GaussianNB()


# fit the model
gnb.fit(X_train, y_train)


# In[297]:


y_pred = gnb.predict(X_test)

y_pred


# In[298]:


from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print("Model accuracy score:",ac) 


# In[299]:


bias = gnb.score(X_train, y_train)
bias

variance = gnb.score(X_test, y_test)
variance
print('Training set score: ',bias)
print("Test set score: ",variance)


# In[300]:


y_test.value_counts()


# In[301]:


null_accuracy = (7407/(7407+2362))
print("Null accuracy score:",null_accuracy)


# In[302]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[303]:


sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')


# In[304]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[305]:


TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]


# In[306]:


classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)


# In[307]:


classification_accuracy


# In[308]:


classification_error = (FP + FN) / float(TP + TN + FP + FN)
classification_error


# In[309]:


# print precision score

precision = TP / float(TP + FP)
precision


# In[310]:


recall = TP / float(TP + FN)
recall


# In[311]:


true_positive_rate = TP / (TP + FN)


# In[312]:


true_positive_rate


# In[313]:


false_positive_rate = FP / float(FP + TN)

false_positive_rate


# In[314]:


specificity = TN / (TN + FP)
specificity


# In[315]:


# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = gnb.predict_proba(X_test)[0:20]

y_pred_prob


# In[316]:


y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50K', 'Prob of - >50K'])

y_pred_prob_df


# In[317]:


gnb.predict_proba(X_test)[0:10, 1]


# In[318]:


y_test


# In[319]:


gnb.predict_proba(X_test)[0:10]


# In[320]:


gnb.predict_proba(X_test)[0:5, 1]


# In[321]:


## Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = gnb, X = X_train, y = y_train,cv = 10)


# In[322]:


print("Accuracy:",(accuracies.mean()*100))


# In[323]:


variance


# In[324]:


bias


# In[ ]:




