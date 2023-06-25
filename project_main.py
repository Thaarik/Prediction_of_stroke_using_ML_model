#!/usr/bin/env python
# coding: utf-8

# # Stroke Prediction Data Analytics using Logistic regression and kNN classification.

# ### 1. Importing necessary libraries for the analysis

# In[531]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.simplefilter('ignore',category=UserWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# ### 2. Importing and loading the data from the csv containing dataset of every patient record

# In[532]:


data = pd.read_csv('healthcare-dataset-stroke-data.csv')
data.columns #features and traget value (stroke) in the dataset


# #### 2.1. Data Exploration

# In[533]:


data.shape #size of the dataset


# In[534]:


data.head() # first 5 row 


# In[535]:


data.tail() # last 5 rows


# In[536]:


data.info() # information of attributes


# In[537]:


data.describe() 


# ### 3. Analysis of Data and Data Preprocessing

# In[538]:


df = data.copy() #create a copy of the data for preprocessing
#id column contains unique values of all patients. So this column is dropped
df = df.drop(['id'],axis=1)


# #### 3.1. Find the null values in the columns and process it

# In[539]:


df.isnull().sum()


# In[540]:


# bmi column has 201 null values
#fill the missing value in numerical variable with the mean value
df['bmi']=df['bmi'].fillna(df['bmi'].mean())


# In[541]:


df.isnull().sum()


# #### 3.2. Dividing the columns into numerical and categorical 

# In[542]:


num = ['age','avg_glucose_level','bmi']
df_num = df[num]
cat = ['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status']
df_cat = df[['gender','hypertension','heart_disease','ever_married','work_type','Residence_type','smoking_status']]


# #### 3.3. Distribution of all numerical value (df_num)

# In[543]:


for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show()


# #### 3.4. Relation between Numerical valued feature and target variable (Stroke)

# In[544]:


for i, num_feature in enumerate(num):
    plt.figure(i,figsize=(6, 4), dpi=80)
    sns.scatterplot(x=num_feature, y='stroke', data=df)
    plt.title('Relation between {} and Stroke Variable'.format(num_feature))
    plt.xlabel(num_feature)
    plt.ylabel('Stroke')
    plt.show()


# In[545]:


#compare the stroke across, numerical values columns
pd.pivot_table(df,index = 'stroke', values=['age','avg_glucose_level','bmi'])


# #### 3.5. Check correlations between numerical values

# In[546]:


# to check correlations in numerical value
print(df_num.corr())
sns.heatmap(df_num.corr(),annot=True,cmap="BrBG")


# #### 3.6. Distribution of all continuous variable in numerical data 

# In[547]:


#distribution of continuous variable
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(16,4))
df['avg_glucose_level'].plot(kind='hist',bins=50,ax=axes[0],xlabel="Avg glucose level",title="Histogram")
df['avg_glucose_level'].plot(kind='density', color='r', ax=axes[1], title='Density Plot')
df['avg_glucose_level'].plot(kind='box', ax=axes[2], ylabel='avg_glucose_level',
                           xlabel='', title='Boxplot')
plt.show()


# In[548]:


#distribution of continuous variable
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(16,4))
df['bmi'].plot(kind='hist',bins=50,ax=axes[0],xlabel="BMI",title="Histogram")
df['bmi'].plot(kind='density', color='r', ax=axes[1], title='Density Plot')
df['bmi'].plot(kind='box', ax=axes[2], ylabel='BMI',
                           xlabel='', title='Boxplot')
plt.show()


# In[549]:


#distribution of continuous variable
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(16,4))
df['age'].plot(kind='hist',bins=50,ax=axes[0],xlabel="age",title="Histogram")
df['age'].plot(kind='density', color='r', ax=axes[1], title='Density Plot')
df['age'].plot(kind='box', ax=axes[2], ylabel='age',
                           xlabel='', title='Boxplot')
plt.show()


# #### 3.7 Distribution of all categorical values (df_cat)

# In[550]:


#distribution for all categorical values (df_cat)
# x gives 0 and 1 at x axis
# y gives count value 
for i in df_cat.columns:
    sns.barplot(x=df_cat[i].value_counts().index ,y=df_cat[i].value_counts()).set_title(i)
    plt.show()


# In[551]:


#remove 'Other' in gender column
print(df["gender"].value_counts())
df = df.drop(df[df["gender"]=="Other"].index.values,axis=0)
print(df["gender"].value_counts())
df.shape


# #### 3.8. Relation between Categorical valued feature and target variable (Stroke)

# In[552]:


for feature in cat:
    ct = pd.crosstab(df[feature], df['stroke'], normalize='index')
    ct.plot(kind='bar', stacked=True,figsize=(8,4))
    plt.title(f'Relation between {feature} and Stroke Variable')
    plt.xlabel(feature)
    plt.ylabel('Proportion')
    plt.xticks(rotation=0)
    plt.show()


# #### 3.9. Analysis on Traget Variable

# In[553]:


print('Values and counts for stroke are: \n{}'.format(df['stroke'].value_counts()))
print('Values and normalised counts for stroke are: \n{}'.format(df['stroke'].value_counts(normalize=True)))


# In[554]:


df['stroke'].value_counts().plot.bar(figsize=(5,4))
plt.show()


# In[555]:


df_original = df.copy()


# #### 3.10. Normalizing the numerical value for better modelling 

# In[556]:


#adjusting distribution of continuous variables by normalization
df['avg_glucose_level_log']=np.log(df['avg_glucose_level'])
df['bmi_log']=np.log(df['bmi'])

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(16,4))
df['avg_glucose_level'].plot(kind='hist', bins=50, ax=axes[0], title='Raw variable distribution')
df['avg_glucose_level_log'].plot(kind='hist', bins=50, ax=axes[1], title='Log variable distribution')
plt.show()


# In[557]:


fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(16,4))
df['bmi'].plot(kind='hist', bins=50, ax=axes[0], title='Raw variable distribution')
df['bmi_log'].plot(kind='hist', bins=50, ax=axes[1], title='Log variable distribution')
plt.show()


# #### 3.11. Coverting the categorical variables into integer ones using LabelEncoder

# In[558]:


#converting categorical variables to integer ones. Label Encoder is used in this process
from sklearn.preprocessing import LabelEncoder
labelEncode = LabelEncoder() 
df['gender'] = labelEncode.fit_transform(df['gender'])
print('\nGender is converted to integer:\n{}'.format(df['gender'].value_counts()))
df['ever_married'] = labelEncode.fit_transform(df['ever_married'])
print('\nMarriage Status is converted to integer:\n{}'.format(df['ever_married'].value_counts()))
df['work_type'] = labelEncode.fit_transform(df['work_type'])
print('\nwork_type is converted to integer:\n{}'.format(df['work_type'].value_counts()))
df['Residence_type'] = labelEncode.fit_transform(df['Residence_type'])
print('\nResidence_type is converted to integer:\n{}'.format(df['Residence_type'].value_counts()))
df['smoking_status'] = labelEncode.fit_transform(df['smoking_status'])
print('\nsmoking_status is converted to integer:\n{}'.format(df['smoking_status'].value_counts()))


# ### 4. Feature Selection

# In[559]:


#generates correlation heatmap between all features
corr = df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr,annot=True,cmap="BrBG")
plt.title("Correlation heatmap between all features")
plt.show()


# In[560]:


#drop bmi and avg glucose level because of high correlation coefficient
cols = ['avg_glucose_level','bmi']
df = df.drop(columns=cols,axis=1)


# In[561]:


corr = df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr,annot=True,cmap="BrBG")
plt.title("Correlation heatmap between all features after removing avg_glucose_level and bmi features")
plt.show()


# In[562]:


# Preparation of feature variable and target variable
X = df.drop(['stroke'],axis=1)
print(X)
y = df['stroke']
print('\nShape of X and y before resampling - X: {},y :{}\n'.format(X.shape,y.shape))
y.value_counts().plot.bar(figsize=(5,4))
plt.title('count of target variable before resampling')
plt.show()


# In[563]:


#now check how strongly the remaining features are associated with target 
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
chi2 = SelectKBest(score_func = chi2, k = 'all').fit(X,y)
chi2_sorted = pd.Series(data=chi2.scores_, index=X.columns).sort_values()
ftest = SelectKBest(score_func = f_classif, k = 'all').fit(X,y)
ftest_sorted = pd.Series(data=ftest.scores_, index=X.columns).sort_values()
mitest = SelectKBest(score_func = mutual_info_classif, k = 'all').fit(X,y)
mitest_sorted = pd.Series(data=mitest.scores_, index=X.columns).sort_values()


# In[564]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
plt.subplots_adjust(wspace=0.8)
chi2_sorted.plot(kind='barh', ax=axes[0], title='chi2 score')
ftest_sorted.plot(kind='barh', ax=axes[1], title='F Test score')
plt.show()


# ### 5. Prediction using Scikit-Learn

# In[565]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, auc, roc_curve, classification_report


# #### 5.1 Resampling the imbalance dataset with SMOTE technique

# In[566]:


from imblearn.over_sampling import SMOTE #imblearn is derived from sklearn library
# Create SMOTE object
smote = SMOTE(random_state=42)
# resampling the imbalanced dataset
X, y = smote.fit_resample(X, y)
print('\nShape of X and y after resampling - X: {},y :{}\n'.format(X.shape,y.shape))
y.value_counts().plot.bar(figsize=(5,4))
plt.title('count of target variable after resampling')
plt.show()


# #### 5.2. Prediction model with 10 fold cross validation function creation

# In[567]:


def train_and_validate(model,X,y):
    print('\nResults from {} algorithm:'.format(model))
    #splitting the dataset into train and test data
    X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    #prediction
    predicted_model = model.fit(X_train,y_train)
    print('\nAccuracy on training data is {:.3f}\n'.format(model.score(X_train,y_train)))
    #checking the cross validation score for 10 folds
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    print('Mean cross-validation accuracy is {:.3f} with SD {:.3f}'.format(np.mean(scores),np.std(scores)))
    y_pred=model.predict(X_test)
    print('Accuracy on test data is {:.3f}\n'.format(accuracy_score(y_true=y_test, y_pred=y_pred)))
    print(classification_report(y_test,y_pred))
    print('Test data metrics: accuracy={:.3f}, f1={:.3f}, precision={:.3f}, recall={:.3f}'
          .format(accuracy_score(y_true=y_test, y_pred=y_pred),
                  f1_score(y_true=y_test, y_pred=y_pred),
                  precision_score(y_true=y_test, y_pred=y_pred),
                  recall_score(y_true=y_test, y_pred=y_pred)))
    confusionmatrix = confusion_matrix(y_true=y_test, y_pred = y_pred)
    plt.figure(figsize=(4,4))
    print(confusionmatrix)
    ax = sns.heatmap(confusionmatrix,annot=True,fmt="d",cmap="BrBG",xticklabels=['N','Y'], cbar=False,yticklabels=['N','Y'],square=True,linewidths=8.0)
    ax.set_xlabel('Predicted Stroke')
    ax.set_ylabel('Actual Stroke')
    plt.show()
    
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)  
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return predicted_model


# #### 5.3 Logistic Regression Model

# In[568]:


from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
train_and_validate(LogisticRegression(), X,y)


# #### 5.4 kNN classification model

# In[569]:


from sklearn.neighbors import KNeighborsClassifier
train_and_validate(KNeighborsClassifier(), X, y)


# Accuracy of kNN classification model is much higher than Logistic regression model. So, we need to try hyperparameter tuning and PCA reduction on Logistic regression model.

# #### 5.5 choosing best hyperparameter model for logistic regression model

# In[570]:


# hyperparameter
from sklearn.model_selection import GridSearchCV
# create logistic regression object
logreg_model = LogisticRegression(solver='liblinear') # liblinear allows both l1 and l2 regularization
# specify hyperparameter grid
parameters = {'C': [0.01, 0.1, 1, 10, 100],'penalty': ['l1', 'l2']}
# perform grid search with 10-fold cross validation
grid_search = GridSearchCV(logreg_model, parameters, cv=10)
X_train,X_test,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
grid_search.fit(X_train, y_train)
print("Best hyperparameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# #### 5.6 PCA on Logistic regression model

# In[571]:


# PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
df = df_original.copy()
cols = ['age','gender','hypertension','heart_disease','ever_married','avg_glucose_level','bmi','work_type','Residence_type','smoking_status']
le = LabelEncoder()                       # initialise the necessary function taken from the LabelEncoder library
for col in cols:                          # iterate over all variables in cols
    df[col] = le.fit_transform(df[col])   # convert categorical values into integer values
X = df.drop(['stroke'],axis=1)
y = df['stroke']
print(X.head())
scaler = MinMaxScaler()       # Normalizing the feature variable
scaler.fit(X)
X = scaler.transform(X)


# In[572]:


from sklearn.decomposition import PCA
pca = PCA(n_components=10).fit(X)
plt.plot(range(1,11), np.cumsum(pca.explained_variance_ratio_)) #plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xticks(range(1,11))  
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()


# In[573]:


pca = PCA(n_components=10).fit(X)
X_reduc = pca.transform(X)
print('\nPCA reduces features from {} to {}'.format(X.shape, X_reduc.shape))
learnt_model = train_and_evaluate(LogisticRegression(), X_reduc, y)


# After hyperparameter tuning and PCA feature reduction on Logistic regression model, kNN still outperforms the Logistic regression model
