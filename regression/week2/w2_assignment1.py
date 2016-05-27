
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import math
from sklearn import linear_model


# In[2]:

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 
              'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float,
              'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}


# In[10]:

housePrice_train = pd.read_csv("/Users/wxu/workspace/regression/week2/kc_house_train_data.csv",sep=",",header=0,dtype = dtype_dict)
housePrice_test = pd.read_csv("/Users/wxu/workspace/regression/week2/kc_house_test_data.csv",sep=",",header=0,dtype = dtype_dict)


# In[11]:

housePrice_train["bedrooms_squared"] = housePrice_train["bedrooms"]*housePrice_train["bedrooms"]
housePrice_train["bed_bath_rooms"] = housePrice_train["bedrooms"]*housePrice_train["bathrooms"]
housePrice_train["log_sqft_living"] = np.log(housePrice_train["sqft_living"])
housePrice_train["lat_plus_long"] = housePrice_train["lat"]+housePrice_train["long"]

housePrice_test["bedrooms_squared"] = housePrice_test["bedrooms"]*housePrice_test["bedrooms"]
housePrice_test["bed_bath_rooms"] = housePrice_test["bedrooms"]*housePrice_test["bathrooms"]
housePrice_test["log_sqft_living"] = np.log(housePrice_test["sqft_living"])
housePrice_test["lat_plus_long"] = housePrice_test["lat"]+housePrice_test["long"]


# In[14]:

# quiz 1
print np.mean(housePrice_test["bedrooms_squared"])
print np.mean(housePrice_test["bed_bath_rooms"])
print np.mean(housePrice_test["log_sqft_living"])
print np.mean(housePrice_test["lat_plus_long"])


# In[5]:

housePrice_train_X_model1 = housePrice_train[["sqft_living","bedrooms","bathrooms","lat","long"]]
housePrice_train_X_model2 = housePrice_train[["sqft_living","bedrooms","bathrooms","lat","long","bed_bath_rooms"]]
housePrice_train_X_model3 = housePrice_train[["sqft_living","bedrooms","bathrooms","lat","long","bed_bath_rooms",
                                             "bedrooms_squared","log_sqft_living","lat_plus_long"]]
housePrice_train_y = housePrice_train["price"]

housePrice_test_X_model1 = housePrice_test[["sqft_living","bedrooms","bathrooms","lat","long"]]
housePrice_test_X_model2 = housePrice_test[["sqft_living","bedrooms","bathrooms","lat","long","bed_bath_rooms"]]
housePrice_test_X_model3 = housePrice_test[["sqft_living","bedrooms","bathrooms","lat","long","bed_bath_rooms",
                                             "bedrooms_squared","log_sqft_living","lat_plus_long"]]
housePrice_test_y = housePrice_test["price"]


# In[6]:

model1 = linear_model.LinearRegression().fit(housePrice_train_X_model1,housePrice_train_y)
print('Coefficients: \n', model1.coef_)
model2 = linear_model.LinearRegression().fit(housePrice_train_X_model2,housePrice_train_y)
print('Coefficients: \n', model2.coef_)
model3 = linear_model.LinearRegression().fit(housePrice_train_X_model3,housePrice_train_y)


# In[7]:

# RSS
model1_testing_rss = np.mean((model1.predict(housePrice_test_X_model1) - housePrice_test_y) ** 2)
model2_testing_rss = np.mean((model2.predict(housePrice_test_X_model2) - housePrice_test_y) ** 2)
model3_testing_rss = np.mean((model3.predict(housePrice_test_X_model3) - housePrice_test_y) ** 2)
print model1_testing_rss
print model2_testing_rss
print model3_testing_rss


# In[9]:

housePrice_train["incpt"]= np.array([1]*len(housePrice_train.index))
housePrice_test["incpt"]= np.array([1]*len(housePrice_test.index))
housePrice_train_X_model1 = housePrice_train[["incpt","sqft_living","bedrooms","bathrooms","lat","long"]]
housePrice_train_X_model2 = housePrice_train[["incpt","sqft_living","bedrooms","bathrooms","lat","long","bed_bath_rooms"]]
housePrice_train_X_model3 = housePrice_train[["incpt","sqft_living","bedrooms","bathrooms","lat","long","bed_bath_rooms",
                                             "bedrooms_squared","log_sqft_living","lat_plus_long"]]
housePrice_train_y = housePrice_train["price"]

housePrice_test_X_model1 = housePrice_test[["incpt","sqft_living","bedrooms","bathrooms","lat","long"]]
housePrice_test_X_model2 = housePrice_test[["incpt","sqft_living","bedrooms","bathrooms","lat","long","bed_bath_rooms"]]
housePrice_test_X_model3 = housePrice_test[["incpt","sqft_living","bedrooms","bathrooms","lat","long","bed_bath_rooms",
                                             "bedrooms_squared","log_sqft_living","lat_plus_long"]]
housePrice_test_y = housePrice_test["price"]


# In[13]:

dataframe=housePrice_train
features = ["sqft_living","bedrooms","bathrooms","lat","long"]
output = ["price"]


# In[63]:

def get_numpy_data(dataframe, features, output):
    
    dataframe['constant']=1
    features = ['constant'] + features
    features_matrix = dataframe[features].as_matrix()
    output_array = dataframe.as_matrix([output])
    
    return (features_matrix, output_array)


# In[131]:

def predict_outcome(feature_matrix,weights):
    predictions = np.dot(feature_matrix,weights)
    return(predictions)


# In[132]:

def feature_derivative(error, feature):
    derivative = 2*sum(error*feature)
    return(derivative)


# In[136]:

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    curr_weights = np.array(initial_weights)
    
    while not converged:
        
        predictions = predict_outcome(feature_matrix,curr_weights)
        error = predictions-output
        
        gradient_sum_squares = 0
        
        for i in range(len(curr_weights)):
            ith_feature = feature_matrix[:,i]
            ith_feature.shape = (len(ith_feature),1)
            ith_derivative = feature_derivative(error,ith_feature)
            gradient_sum_squares = gradient_sum_squares+ith_derivative**2
            curr_weights[i] = curr_weights[i]-step_size*ith_derivative
        
        gradient_magnitude = math.sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(curr_weights)
        


# In[134]:

simple_features = ['sqft_living']
my_output= 'price'
(simple_feature_matrix, output) = get_numpy_data(housePrice_train, simple_features, my_output)
initial_weights =[[-47000.], [1.]]
step_size = 7e-12
tolerance = 2.5e7


# In[137]:

#quiz1
simple_weights = regression_gradient_descent(simple_feature_matrix, output,initial_weights, step_size, tolerance)

#quiz2
(test_simple_feature_matrix, test_output) = get_numpy_data(housePrice_test, simple_features, my_output)


# In[140]:

test_pred = predict_outcome(feature_matrix=test_simple_feature_matrix,weights=simple_weights)


# In[141]:

print test_pred[0]


# In[142]:

simple_rss = np.mean((test_pred - test_output) ** 2)
print simple_rss


# In[146]:

model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(housePrice_train, model_features,my_output)
initial_weights = np.array([[-100000.], [1.], [1.]])
step_size = 4e-12
tolerance = 1e9


# In[148]:

second_weights = regression_gradient_descent(feature_matrix, output,initial_weights, step_size, tolerance)


# In[150]:

(test_sec_feature_matrix, test_output) = get_numpy_data(housePrice_test, model_features, my_output)
test_pred2 = predict_outcome(feature_matrix=test_sec_feature_matrix,weights=second_weights)


# In[152]:

#quiz3
print test_pred2[0]
print test_pred[0]
print test_output[0]


# In[153]:

#quiz4
second_rss = np.mean((test_pred2 - test_output) ** 2)
print second_rss

