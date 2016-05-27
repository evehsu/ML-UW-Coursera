
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
from sklearn import linear_model
import math


# In[2]:

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 
              'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float,
              'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}


# In[3]:

housePrice = pd.read_csv("/Users/wxu/workspace/regression/kc_house_data.csv",sep=",",header=0,dtype = dtype_dict)
housePrice_train = pd.read_csv("/Users/wxu/workspace/regression/kc_house_train_data.csv",sep=",",header=0,dtype = dtype_dict)
housePrice_test = pd.read_csv("/Users/wxu/workspace/regression/kc_house_test_data.csv",sep=",",header=0,dtype = dtype_dict)


# In[4]:

input_sqft = housePrice_train["sqft_living"]
input_bedroom = housePrice_train["bedrooms"]
output_y = housePrice_train["price"]
input_sqft_test = housePrice_test["sqft_living"]
input_bedroom_test = housePrice_test["bedrooms"]
output_y_test = housePrice_test["price"]


# In[14]:

def simple_regression(input_feature,output):
    # input_feature: the single predictor
    # output_feature: y/target/lable
    # return the slope and intercept
    
    assert (len(input_feature) == len(output)),"length different between predictor and target"
    
    x_mean  = np.mean(input_feature)
    y_mean = np.mean(output)
    xy_mean = np.mean(map(lambda x,y:x*y,input_feature,output))
    xSquare_mean = np.mean(map(lambda x:x**2,input_feature))
    
    slope = (xy_mean-x_mean*y_mean)/(xSquare_mean -x_mean*x_mean)
    intercept = y_mean-slope*x_mean
    
    return slope,intercept


# In[6]:

def get_regression_prediction(input_feature,intercept,slope):
    output = slope*input_feature+intercept
    return output


# In[11]:

def get_RSS(input_feature,output,intercept,slope):
    predict = map(lambda x: x*slope,input_feature)
    predict = map(lambda x: x+intercept,predict)
    rssList = map(lambda x,y: x-y,predict,output)
    rss = sum(map(lambda x: x**2,rssList))
    return rss


# In[12]:

def inverse_regression_predictions(output,intercept,slope):
    estimate_input = (output-intercept)/slope
    return estimate_input


# In[15]:

#quiz1
sqftSlope,sqftIntercept = simple_regression(input_sqft,output_y)
predPrice2650 = get_regression_prediction(2650,sqftIntercept,sqftSlope)
print predPrice2650


# In[16]:

#quiz2
rss = get_RSS(input_sqft,output_y,sqftIntercept,sqftSlope)


# In[17]:

rss


# In[18]:

#quiz3
estimate_sqft = inverse_regression_predictions(800000,sqftIntercept,sqftSlope)
print estimate_sqft


# In[19]:

#quiz4
bdrSlope,bdrIntercept = simple_regression(input_bedroom,output_y)
rssSqft = get_RSS(input_sqft_test,output_y_test,sqftIntercept,sqftSlope)
rssBedroom  = get_RSS(input_bedroom_test,output_y_test,bdrIntercept,bdrSlope)
print rssSqft,rssBedroom


# In[ ]:



