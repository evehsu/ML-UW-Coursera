
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn import linear_model
from math import log, sqrt
import functools


# In[13]:

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}


# In[19]:

training = pd.read_csv("/Users/wxu/workspace/regression/week3/wk3_kc_house_train_data.csv",sep=",",header=0,dtype = dtype_dict)
validation = pd.read_csv("/Users/wxu/workspace/regression/week3/wk3_kc_house_valid_data.csv",sep=",",header=0,dtype = dtype_dict)
testing = pd.read_csv("/Users/wxu/workspace/regression/week3/wk3_kc_house_test_data.csv",sep=",",header=0,dtype = dtype_dict)
house = pd.read_csv("/Users/wxu/workspace/regression/week3/kc_house_data.csv",sep=",",header= 0,dtype = dtype_dict)


# In[15]:

house["sqft_living_sqrt"] = house["sqft_living"].apply(sqrt)
house['sqft_lot_sqrt'] = house['sqft_lot'].apply(sqrt)
house["bedrooms_square"] = house["bedrooms"]*house["bedrooms"]
house["floors_square"]=house["floors"]*house["floors"]


# In[16]:

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']


# In[17]:

model_all = linear_model.Lasso(alpha=5e2,normalize=True)
model_all_fit = model_all.fit(house[all_features],house["price"])


# In[87]:

print model_all_fit.coef_
print all_features


# In[20]:

testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']


# In[31]:

def select_lambda_rss(lambdaList):
    rss = [0]*len(lambdaList)
    for i in range(len(lambdaList)):
        
        curr_model = linear_model.Lasso(alpha = lambdaList[i],normalize=True)
        curr_model_fit = curr_model.fit(training[all_features],training["price"])
        curr_rss = sum((curr_model_fit.predict(validation[all_features])-validation["price"])**2)
        rss[i] = curr_rss
    return rss
    


# In[35]:

myLambda = np.logspace(1, 7, num=13)
print myLambda


# In[33]:

myRss = select_lambda_rss(myLambda)
print myRss
print min(myRss)


# In[38]:

select_model = linear_model.Lasso(alpha = 10,normalize=True)
select_model_fit = select_model.fit(training[all_features],training["price"])
test_rss = sum((select_model_fit.predict(testing[all_features])-testing["price"])**2)
print test_rss


# In[39]:

np.count_nonzero(select_model_fit.coef_) + np.count_nonzero(select_model_fit.intercept_)


# In[41]:

def select_lambda_fixed(lambdaList):
    nonzero = [0]*len(lambdaList)
    for i in range(len(lambdaList)):
        
        curr_model = linear_model.Lasso(alpha = lambdaList[i],normalize=True)
        curr_model_fit = curr_model.fit(training[all_features],training["price"])
        curr_nonzero = np.count_nonzero(curr_model_fit.coef_) + np.count_nonzero(curr_model_fit.intercept_)
        nonzero[i] = curr_nonzero
    return nonzero
    


# In[42]:

myLambda2 = np.logspace(1, 4, num=20)
myNonzero = select_lambda_fixed(myLambda2)


# In[43]:

myNonzero


# In[48]:


a = [a for (a,b) in zip(myLambda2,myNonzero) if b==7]
print a
print myLambda2[8]


# In[52]:

l1_penalty_min = max([a for (a,b) in zip(myLambda2,myNonzero) if b==10])
l1_penalty_max = min([a for (a,b) in zip(myLambda2,myNonzero) if b==6])
print l1_penalty_min
print l1_penalty_max


# In[54]:

myLambda3 = np.linspace(l1_penalty_min,l1_penalty_max,20)


# In[58]:

def select_lambda_rss_fixed(lambdaList):
    nonzero = [0]*len(lambdaList)
    rss = [0]*len(lambdaList)
    for i in range(len(lambdaList)):
        
        curr_model = linear_model.Lasso(alpha = lambdaList[i],normalize=True)
        curr_model_fit = curr_model.fit(training[all_features],training["price"])
        curr_nonzero = np.count_nonzero(curr_model_fit.coef_) + np.count_nonzero(curr_model_fit.intercept_)
        nonzero[i] = curr_nonzero
        curr_rss = sum((curr_model_fit.predict(validation[all_features])-validation["price"])**2)
        rss[i] = curr_rss
        theList = zip(rss,nonzero,myLambda3)
    return theList


# In[62]:

mytuple = select_lambda_rss_fixed(myLambda3)
filterMytuple = [elem for elem in mytuple if elem[1]==7]
print filterMytuple


# In[69]:

selectTuple = [elem for elem in filterMytuple if elem[0]==min(map(lambda x: x[0],filterMytuple))]


# In[70]:

print selectTuple


# In[71]:

final_model = linear_model.Lasso(alpha = 156.10909673930755,normalize=True)
final_model_fit = final_model.fit(training[all_features],training["price"])
final_model_fit.coef_


# In[86]:

a = final_model_fit.coef_
final_feature = [feature for feature,index in zip(all_features,range(len(a))) if index in list(np.where(a!=0)[0])]
print final_feature


# In[84]:

print len(a)
print len(all_features)

