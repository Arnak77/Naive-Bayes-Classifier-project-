
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization


import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv(r"D:\NIT\JANUARY\8 JAN(NAIVE BAYES ALGORITHM)\8th\project\adult.csv")

df

df.shape

df.head()

df.info()

df.columns

categorical_columns = df.select_dtypes(include=['object']).columns


categorical_columns

categorical = [var for var in df.columns if df[var].dtype=='O']


print(len(categorical))
print("The categorical variables are :\n",categorical)


df[categorical].head()

df[categorical].isnull().sum()

for var in categorical:
    print(df[var].value_counts())

# view frequency distribution of categorical variables

for var in categorical: 
    
    print(df[var].value_counts()/np.float32(len(df)))


df.workclass.unique()

df.workclass.nunique()

df.workclass.value_counts()

for var in categorical: 
    print(df[var].replace('?', np.NaN, inplace=True))

df.workclass.value_counts()


df.occupation.unique()

df[categorical].isnull().sum()

# check for cardinality in categorical variables

for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')

numerical_columns = df.select_dtypes(include=['number']).columns

numerical_columns

numerical = [var for var in df.columns if df[var].dtype!='O']
print(len(numerical))
print("The numerical variables are :\n",numerical)

df[numerical].head()


df[numerical].isnull().sum()

X = df.drop(['income'], axis=1)

y = df['income']

X.head(2)

y.head(2)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

X_train, X_test, y_train, y_test

X_train.shape, X_test.shape

y_train.shape, y_test.shape


X_train.dtypes

categorical = [col for col in X_train.columns if X_train[col].dtypes == 'O']

categorical

numerical = [col for col in X_train.columns if X_train[col].dtypes != 'O']

numerical

X_train[categorical].isnull().sum()

X_train

X_train[categorical].isnull().mean()

X_test[categorical].isnull().sum()

y_train.isnull().sum()

y_test.isnull().sum()

X_train[categorical].columns



X_train[numerical].columns

print(X_train.columns)


### Impute missing values with mode
for col in ["workclass","occupation","native.country"]:
    X_train[col].fillna(X_train[col].mode()[0],inplace=True)

X_train.isnull().sum()

# check missing values in categorical variables in X_test

X_test[categorical].isnull().sum()

### Impute missing values with mode
for col in ["workclass","occupation","native.country"]:
    X_test[col].fillna(X_test[col].mode()[0],inplace=True)


X_test[categorical].isnull().sum()

categorical

X_train[categorical].head(3)



import category_encoders as ce

encoder = ce.OneHotEncoder(['workclass', 'education', 'marital_status', 'occupation', 'relationship', 
                                 'race', 'sex', 'native_country'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)





'''from sklearn.preprocessing import LabelEncoder
categorical_features = categorical

for feature in categorical_features:
    le = LabelEncoder()
    X_train[feature] = le.fit_transform(X_train[feature])
    X_test[feature] = le.transform(X_test[feature])'''


X_train.head(3)

X_test.head(3)

categorical

X_train.shape

X_test.head()

X_test.shape

X_train.columns

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

X_train = pd.DataFrame(X_train)

X_test = pd.DataFrame(X_test)

X_train

X_test

X_train.head()

X_train[:5]

# train a Gaussian Naive Bayes classifier on the training set
from sklearn.naive_bayes import GaussianNB


# instantiate the model
gnb = GaussianNB()


# fit the model
gnb.fit(X_train, y_train)


y_pred = gnb.predict(X_test)

y_pred



from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print("Model accuracy score:",ac) 

bias = gnb.score(X_train, y_train)
bias

variance = gnb.score(X_test, y_test)
variance
print('Training set score: ',bias)
print("Test set score: ",variance)


y_test.value_counts()

null_accuracy = (7407/(7407+2362))
print("Null accuracy score:",null_accuracy)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)


classification_accuracy

classification_error = (FP + FN) / float(TP + TN + FP + FN)
classification_error

# print precision score

precision = TP / float(TP + FP)
precision

recall = TP / float(TP + FN)
recall

true_positive_rate = TP / (TP + FN)

true_positive_rate

false_positive_rate = FP / float(FP + TN)

false_positive_rate

specificity = TN / (TN + FP)
specificity

# print the first 10 predicted probabilities of two classes- 0 and 1

y_pred_prob = gnb.predict_proba(X_test)[0:20]

y_pred_prob

y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50K', 'Prob of - >50K'])

y_pred_prob_df

gnb.predict_proba(X_test)[0:10, 1]

y_test

gnb.predict_proba(X_test)[0:10]

gnb.predict_proba(X_test)[0:5, 1]



## Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = gnb, X = X_train, y = y_train,cv = 10)

print("Accuracy:",(accuracies.mean()*100))

variance

bias

