
# coding: utf-8

# In[12]:

import pandas as pd
import numpy as np
import math
from sklearn import linear_model
import functools


# In[2]:

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 
              'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 
              'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 
              'id':str, 'sqft_lot':int, 'view':int}


# In[4]:

train = pd.read_csv("/Users/wxu/workspace/regression/week4/wk3_kc_house_train_data.csv",sep=",",header=0,dtype = dtype_dict)
valid = pd.read_csv("/Users/wxu/workspace/regression/week4/wk3_kc_house_valid_data.csv",sep=",",header=0,dtype = dtype_dict)
test = pd.read_csv("/Users/wxu/workspace/regression/week4/wk3_kc_house_test_data.csv",sep=",",header=0,dtype = dtype_dict)
house = pd.read_csv("/Users/wxu/workspace/regression/week3/kc_house_data.csv",sep=",",header= 0,dtype = dtype_dict)
shuffle = pd.read_csv("/Users/wxu/workspace/regression/week4/wk3_kc_house_train_valid_shuffled.csv",sep = ",",header=0,dtype = dtype_dict)

set1 = pd.read_csv("/Users/wxu/workspace/regression/week4/wk3_kc_house_set_1_data.csv",sep=",",header=0,dtype = dtype_dict)
set2 = pd.read_csv("/Users/wxu/workspace/regression/week4/wk3_kc_house_set_2_data.csv",sep=",",header=0,dtype = dtype_dict)
set3 = pd.read_csv("/Users/wxu/workspace/regression/week4/wk3_kc_house_set_3_data.csv",sep=",",header=0,dtype = dtype_dict)
set4 = pd.read_csv("/Users/wxu/workspace/regression/week4/wk3_kc_house_set_4_data.csv",sep=",",header= 0,dtype = dtype_dict)


# In[ ]:

house = house.sort_values(["sqft_living","price"])


# In[21]:

l2_small_penalty = 1.5e-5


# In[15]:

def take_power(x,i):
    poly = x**i
    return poly
def polynomial_dataframe(feature, degree): # feature is pandas.Series type
    # assume that degree >= 1
    # initialize the dataframe:
    poly_dataframe = pd.DataFrame()
    poly_dataframe["power_1"] = feature
    # and set poly_dataframe['power_1'] equal to the passed feature
    
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1):
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # assign poly_dataframe[name] to be feature^power; use apply(*)
            poly_dataframe[name] = poly_dataframe['power_1'].apply(functools.partial(take_power,i=power))
    return poly_dataframe


# In[23]:

poly15_data = polynomial_dataframe(feature = house['sqft_living'],degree=15)


# In[25]:

#quiz1
model = linear_model.Ridge(alpha = l2_small_penalty,normalize = True)
model.fit(poly15_data,house["price"]).coef_


# In[26]:

#quiz2
l2_small_penalty = 1e-9


# In[27]:

poly15_set1 = polynomial_dataframe(set1["sqft_living"],degree=15)
model_set = linear_model.Ridge(alpha = l2_small_penalty,normalize = True)
model_set.fit(poly15_set1,set1["price"]).coef_


# In[28]:

poly15_set2 = polynomial_dataframe(set2["sqft_living"],degree=15)
model_set.fit(poly15_set2,set2["price"]).coef_


# In[29]:

poly15_set3 = polynomial_dataframe(set3["sqft_living"],degree=15)
model_set.fit(poly15_set3,set3["price"]).coef_


# In[30]:

poly15_set4 = polynomial_dataframe(set4["sqft_living"],degree=15)
model_set.fit(poly15_set4,set4["price"]).coef_


# In[31]:

#quiz3
l2_small_penalty = 1.23e2
model_set = linear_model.Ridge(alpha = l2_small_penalty,normalize = True)
print model_set.fit(poly15_set1,set1["price"]).coef_
print model_set.fit(poly15_set2,set2["price"]).coef_
print model_set.fit(poly15_set3,set3["price"]).coef_
print model_set.fit(poly15_set4,set4["price"]).coef_


# In[38]:

n=len(shuffle)
k = 10

for i in xrange(k):
    start = n*i/k
    end = n*(i+1)/k-1
    print i ,(start,end)


# In[52]:

#quiz4
def k_fold_cv(k,l2_penalty,data):
    n = len(data)
    my_model = linear_model.Ridge(alpha = l2_penalty,normalize=True)
    validError = [0]*k
    for i in xrange(k):
        start = n*i/k
        end = n*(i+1)/k
        curr_train = data[0:start].append(data[end:])
        curr_test = data[start:end]
        curr_train_x = polynomial_dataframe(curr_train["sqft_living"],degree=15)
        curr_train_y = curr_train["price"]
        curr_test_x = polynomial_dataframe(curr_test["sqft_living"],degree=15)
        curr_test_y = curr_test["price"]
        curr_fit = my_model.fit(curr_train_x,curr_train_y)
        validError[i] = np.mean((curr_fit.predict(curr_test_x) - curr_test_y) ** 2)
    return(np.mean(validError))


# In[56]:

select_penalty = [0]*13
penalty_candidate = np.logspace(3, 9, num=13)
for i in xrange(13):
    penalty = penalty_candidate[i]
    select_penalty[i] = k_fold_cv(k=10,l2_penalty=penalty,data=shuffle)


# In[57]:

select_penalty


# In[58]:

final_model_x = polynomial_dataframe(train["sqft_living"],degree=15)
final_model_y = train["price"]
final_model = linear_model.Ridge(alpha = 1000,normalize=True).fit(final_model_x,final_model_y)
whole_test_x = polynomial_dataframe(test["sqft_living"],degree=15)
whole_test_y = test["price"]
final_rss = sum((final_model.predict(whole_test_x)-whole_test_y)**2)
print final_rss

