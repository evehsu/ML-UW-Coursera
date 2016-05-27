
# coding: utf-8

# In[6]:

import pandas as pd
import numpy as np
from sklearn import linear_model
import math
import functools


# In[3]:

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 
              'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 
              'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 
              'id':str, 'sqft_lot':int, 'view':int}


# In[27]:

training = pd.read_csv("/Users/wxu/workspace/regression/week3/wk3_kc_house_train_data.csv",sep=",",header=0,dtype = dtype_dict)
validation = pd.read_csv("/Users/wxu/workspace/regression/week3/wk3_kc_house_valid_data.csv",sep=",",header=0,dtype = dtype_dict)
test = pd.read_csv("/Users/wxu/workspace/regression/week3/wk3_kc_house_test_data.csv",sep=",",header=0,dtype = dtype_dict)
house = pd.read_csv("/Users/wxu/workspace/regression/week3/kc_house_data.csv",sep=",",header= 0,dtype = dtype_dict)

set1 = pd.read_csv("/Users/wxu/workspace/regression/week3/wk3_kc_house_set_1_data.csv",sep=",",header=0,dtype = dtype_dict)
set2 = pd.read_csv("/Users/wxu/workspace/regression/week3/wk3_kc_house_set_2_data.csv",sep=",",header=0,dtype = dtype_dict)
set3 = pd.read_csv("/Users/wxu/workspace/regression/week3/wk3_kc_house_set_3_data.csv",sep=",",header=0,dtype = dtype_dict)
set4 = pd.read_csv("/Users/wxu/workspace/regression/week3/wk3_kc_house_set_4_data.csv",sep=",",header= 0,dtype = dtype_dict)


# In[5]:

def take_power(x,i):
    poly = x**i
    return poly


# In[31]:

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


# In[38]:

sales = house.sort_values(['sqft_living','price'])


# In[47]:

poly1_data = polynomial_dataframe(sales['sqft_living'], 1)
poly1_data['price'] = sales["price"]
model1_X = poly1_data[["power_1"]]
model1_Y = poly1_data[["price"]]
poly2_data = polynomial_dataframe(sales['sqft_living'], 2)
poly2_data['price'] = sales["price"]
model2_X = poly2_data[["power_1","power_2"]]
model2_Y = poly2_data[["price"]]
poly3_data = polynomial_dataframe(sales['sqft_living'], 3)
poly3_data['price'] = sales["price"]
model3_X = poly3_data[["power_1","power_2","power_3"]]
model3_Y = poly3_data[["price"]]


# In[49]:

model1 = linear_model.LinearRegression().fit(model1_X,model1_Y)
model2 = linear_model.LinearRegression().fit(model2_X,model2_Y)
model3 = linear_model.LinearRegression().fit(model3_X,model3_Y)


# In[50]:

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
poly1_data['power_1'], model1.predict(model1_X),'-')


# In[52]:

plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
poly1_data['power_1'], model2.predict(model2_X),'-')


# In[53]:

plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
poly1_data['power_1'], model3.predict(model3_X),'-')


# In[56]:

poly15_data  = polynomial_dataframe(sales["sqft_living"], degree=15)
model15_X = poly15_data 
poly15_data["price"] = sales["price"]
model15_Y = poly15_data["price"]


# In[57]:

model15 = linear_model.LinearRegression().fit(model15_X,model15_Y)
print ('Coefficients: \n', model15.coef_)


# In[58]:

poly15_set1  = polynomial_dataframe(set1["sqft_living"], degree=15)
model15_X_set1 = poly15_set1 
poly15_set1["price"] = set1["price"]
model15_Y_set1 = poly15_set1["price"]
model15_set1 = linear_model.LinearRegression().fit(model15_X_set1,model15_Y_set1)
print ('Coefficients: \n', model15_set1.coef_)


# In[59]:

poly15_set2  = polynomial_dataframe(set2["sqft_living"], degree=15)
model15_X_set2 = poly15_set2 
poly15_set2["price"] = set2["price"]
model15_Y_set2 = poly15_set2["price"]
model15_set2 = linear_model.LinearRegression().fit(model15_X_set2,model15_Y_set2)
print ('Coefficients: \n', model15_set2.coef_)


# In[60]:

poly15_set3  = polynomial_dataframe(set3["sqft_living"], degree=15)
model15_X_set3 = poly15_set3 
poly15_set3["price"] = set3["price"]
model15_Y_set3 = poly15_set3["price"]
model15_set3 = linear_model.LinearRegression().fit(model15_X_set3,model15_Y_set3)
print ('Coefficients: \n', model15_set3.coef_)


# In[62]:

poly15_set4  = polynomial_dataframe(set4["sqft_living"], degree=15)
model15_X_set4 = poly15_set4 
poly15_set4["price"] = set4["price"]
model15_Y_set4 = poly15_set4["price"]
model15_set4 = linear_model.LinearRegression().fit(model15_X_set4,model15_Y_set4)
print ('Coefficients: \n', model15_set4.coef_)


# In[63]:

plt.plot(poly15_set1['power_1'],poly15_set1['price'],'.',
poly15_set1['power_1'], model15_set1.predict(model15_X_set1),'-')


# In[64]:

plt.plot(poly15_set2['power_1'],poly15_set2['price'],'.',
poly15_set2['power_1'], model15_set2.predict(model15_X_set2),'-')


# In[65]:

plt.plot(poly15_set3['power_1'],poly15_set3['price'],'.',
poly15_set3['power_1'], model15_set3.predict(model15_X_set3),'-')


# In[66]:

plt.plot(poly15_set4['power_1'],poly15_set4['price'],'.',
poly15_set4['power_1'], model15_set4.predict(model15_X_set4),'-')


# In[70]:

valid_rss = [0]*15

for i in range(1,16):
    poly_train  = polynomial_dataframe(training["sqft_living"],degree = i)
    poly_valid = polynomial_dataframe(validation["sqft_living"],degree = i)
    model_X_curr = poly_train
    test_X_curr = poly_valid
    poly_train['price'] = training["price"]
    poly_valid['price'] = validation["price"]
    model_Y_curr = poly_train['price']
    test_Y_curr = poly_valid['price']
    curr_model = linear_model.LinearRegression().fit(model_X_curr,model_Y_curr)
    valid_rss[i-1] = np.mean((curr_model.predict(test_X_curr) - test_Y_curr) ** 2)
valid_rss


# In[71]:

test_X = polynomial_dataframe(test["sqft_living"],degree = 2)
test_Y = test["price"]


# In[72]:

final_train_X = polynomial_dataframe(training["sqft_living"],degree = 2)
final_train_Y = training["price"]
final_model = linear_model.LinearRegression().fit(final_train_X,final_train_Y)
test_rss = np.mean((final_model.predict(test_X) - test_Y) ** 2)
print test_rss


# In[75]:

print len(training)
print len(validation)
print len(test_X)


# In[1]:

61137591012*2217

