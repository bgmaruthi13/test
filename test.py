#!/usr/bin/env python
# coding: utf-8

# # Regression

# ## Importing the libraries

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing the dataset

# In[ ]:


df = pd.read_csv('50_Startups.csv')
df.head(5)


# ##  Info & describe

# In[ ]:


df.info()


# In[ ]:


df.describe()


# ## Statistical Analysis

# In[ ]:


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr() ,annot=True)
plt.show()


# In[ ]:


plt.figure(figsize=(10, 8))
sns.pairplot(data=df)
plt.show()


# ## Outlier Treatment

# In[ ]:


plt.figure(figsize=(10, 8))
sns.boxplot(data=df)
plt.show()


# In[ ]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

lb = Q1 - 1.5 * IQR
ub = Q3 + 1.5 * IQR
df = df[~(df < lb) | (df > ub).any(axis=1)]
df.shape


# ## Column Types

# In[ ]:


num_cols = df.select_dtypes(include=[float, int]).columns
cat_cols = df.select_dtypes(include='object').columns
print(num_cols, cat_cols)
df = pd.concat((df[cat_cols], df[num_cols]), axis=1)
df.head()


# ## Taking care of missing data

# In[ ]:


df.isna().sum()


# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df[num_cols] = imputer.fit_transform(df[num_cols])


# ## Encoding categorical data

# In[ ]:


df = pd.get_dummies(df, columns=cat_cols, drop_first=True).astype(int)
df.head(4)


# ## Split

# In[ ]:


from sklearn.model_selection import train_test_split
X = df.drop(columns='Profit', axis=1)
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## Train

# In[ ]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# ## Predicting the Test set results

# In[ ]:


y_pred = regressor.predict(X_test)


# ## Evaluating the Model Performance

# In[ ]:


# Evaluating the Model Performance
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# ## Feature Selection

# In[ ]:


import statsmodels.api as sm
x_opt = sm.add_constant(X)
display(x_opt.head(4))

OLS_reg = sm.OLS(y, x_opt).fit()
OLS_reg.summary()


# In[ ]:


OLS_reg = sm.OLS(y, x_opt.drop('State_New York', axis=1)).fit()
OLS_reg.summary()


# In[ ]:


OLS_reg = sm.OLS(y, x_opt.drop(columns=['State_New York', 'State_Florida'], axis=1)).fit()
OLS_reg.summary()


# In[ ]:


OLS_reg = sm.OLS(y, x_opt.drop(columns=['State_New York', 'State_Florida', 'Administration'], axis=1)).fit()
OLS_reg.summary()


# In[ ]:


OLS_reg = sm.OLS(y, x_opt.drop(columns=['State_New York', 'State_Florida', 'Administration', 'Marketing Spend'], axis=1)).fit()
OLS_reg.summary()


# ## Regularization 

# In[ ]:


# https://www.dataquest.io/blog/regularization-in-machine-learning/


# In[ ]:


print(f"Linear Regression-Training set score: {regressor.score(X_train, y_train):.2f}")
print(f"Linear Regression-Test set score: {regressor.score(X_test, y_test):.2f}")


# In[ ]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.7).fit(X_train, y_train)
print(f"Ridge Regression-Training set score: {ridge.score(X_train, y_train):.2f}")
print(f"Ridge Regression-Test set score: {ridge.score(X_test, y_test):.2f}")


# In[ ]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0).fit(X_train, y_train)
print(f"Lasso Regression-Training set score: {lasso.score(X_train, y_train):.2f}")
print(f"Lasso Regression-Test set score: {lasso.score(X_test, y_test):.2f}")
print(f"Number of features: {sum(lasso.coef_ != 0)}")


# In[ ]:


from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.07, l1_ratio=0.01).fit(X_train, y_train)
print(f"Elastic Net-Training set score: {elastic_net.score(X_train, y_train):.2f}")
print(f"Elastic Net-Test set score: {elastic_net.score(X_test, y_test):.2f}")
print(f"Number of features: {sum(elastic_net.coef_ != 0)}")


# ## Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)


# ## Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
r2_score(y_test, y_pred)


# In[ ]:


#GridSearchCV


# In[ ]:


from numpy import mean
from numpy import absolute
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

cv = KFold(n_splits=10, random_state=1, shuffle=True)
model = LinearRegression()
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
mean(absolute(scores))


# ##  Assumptions

# In[ ]:


#https://www.kaggle.com/code/tawfikelmetwally/assumptions-of-linear-regression-model


# In[ ]:





# ### Linearity

# In[ ]:





# ### Mean of residuals should be equal zero.

# In[ ]:


residuals = y_train - y_pred

mean_residuals = np.mean(residuals)
print("Mean of Residuals {}".format(mean_residuals))


# ### Normality of residuals

# In[ ]:


fig = plt.figure()
sns.distplot(residuals , bins=20)
fig.suptitle('Error Terms', fontsize = 20)    
plt.xlabel('Errors', fontsize = 18)
plt.show()


# ### Error Term should be independent to each other

# In[ ]:


plt.scatter(y_pred , residuals)
plt.axhline(y=0,color="red" ,linestyle="--")
plt.show()


# In[ ]:





# ### Multicollinearity

# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

columns= X.columns
# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(columns))]
  
vif_data


# In[ ]:


cro


# In[ ]:





# In[ ]:





# In[ ]:




