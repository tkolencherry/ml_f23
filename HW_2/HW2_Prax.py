#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from random import seed
import requests
import io


# In[2]:


#df = pd.read_csv ("C:/Users/teris/ml_f23/HW_2/hotel_booking.csv")

url = "https://raw.githubusercontent.com/tkolencherry/ml_f23/main/HW_2/hotel_booking.csv"
file = requests.get(url).content
df = pd.read_csv(io.StringIO(file.decode('utf-8')))
df.head()


# In[3]:


df.corr()['is_canceled']


# In[4]:


class NN_scratch(object): 
    
    #f(x): Initialize
    #Input: Activation Mode - "sigmoid", "relu", or "tanh"
    #Purpose: Initialize the class and specify which activation function to use in the NN
    def __init__(self, activation_mode, df): 
        self.n_inputs = 8 #number of predictors 
        self.n_hidden = 8 #hidden layer with 4 neurons
        self.n_outputs = 1 #number of outputs - since this is a classification problem we want 1 class
        self.mode = activation_mode
        self.df = df
        np.random.seed(1)
        
        #set of weights to go from the input layer to the hidden layer (10Xn matrix)
        self.w_inner = np.random.randn(self.n_inputs, self.n_hidden)
        
        #set of weights to go from hidden layer to output layer (nX1 matrix)
        self.w_outer = np.random.randn(self.n_hidden, self.n_outputs)
    
    #f(x): Pre-Process and Split 
    def xy_split (self): 
        # Creating dummy variables from one column:
        hotel_type_dict = {"Resort Hotel" : 1, "City Hotel":0}
        df = self.df.replace({'hotel':hotel_type_dict})
        df_dummies = pd.get_dummies(df, columns=['deposit_type'])
        dataset = df_dummies[['hotel', 'lead_time', 'is_repeated_guest', 'previous_cancellations', 'total_of_special_requests', 'deposit_type_No Deposit', 'deposit_type_Non Refund', 'deposit_type_Refundable', 'is_canceled']]
        dataset = dataset.head(3000)
        dataset_x = dataset.drop('is_canceled', axis = 1)
        dataset_y = dataset['is_canceled']
        
        x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size =.20, random_state =42)
        
        return x_train, x_test, y_train, y_test
    
    #f(x): Forward Propogation
    #Input: x-values 
    #Purpose: Complete a forward pass through the neural network
    #Output: the output value after the pass (the class prediction)
    def _fwd_prop(self, X): #input the starting x values and then find what we call netj by taking dot product
        self.net_inner = np.dot(self.w_inner.T, X.T)
        self.net_outer = np.dot(self.w_outer.T, X.T)
        
        if(self.mode == "sigmoid"):
            self.activ_inner = self._sigmoid(self.net_inner)
            self.activ_outer = self._sigmoid(self.net_outer)
        elif(self.mode == "relu"): 
            self.activ_inner = self._relu(self.net_inner)
            self.activ_outer = self._sigmoid(self.net_outer) 
        elif(self.mode == "tanh"):
            self.activ_inner = self._tanh(self.net_inner)
            self.activ_outer = self._sigmoid(self.net_outer)
        return self.activ_outer 
    
    #ACTIVATION FUNCTIONS
    #Input: w.T *X - the weighted predictors
    #Purpose: non-linear activation functions help us solve more complex classification problems
    #Output: neuron output to next layer
    def _sigmoid(self, net):
        return 1.0/(1+np.exp(-net)) 
    
    #have to make this leaky otherwise this isn't very effective
    def _relu (self, net): 
        return np.maximum(0.01, np.array(net))
    
    def _tanh(self, net):
        return (np.nan_to_num((np.exp(net) - np.exp(-net)))/np.nan_to_num((np.exp(net) + np.exp(-net))))
    
    
    #f(x): Loss Function
    #Input: Predicted Y Values and Observed Y Values
    #Purpose: We need an optimization function - earlier we used MSE, but here we should use the log error since MSE isn't an appropriate error calculation for binary classification
    
    def _loss(self, predict, y): #we need to have optimization in this assignment and so we need something to optimize
        n = len(y) #grab the number of observations
        log_prob = np.nan_to_num(np.multiply(np.log(predict), y)) + np.nan_to_num(np.multiply((1-y), np.log(1-predict)))
        loss = - np.sum(log_prob) / n
        return loss
    #f(x): Back Propogation Function 
    #Input: X and Y values 
    #Purpose: We make a backwards pass from the output all the way back to the beginning and update the weights as we go 
    def _back_prop(self, X, y):
        predict = self._fwd_prop(X)
        n = X.shape[0]
        resid = predict - y
        if(self.mode == "sigmoid"):
            delta_outer = np.multiply(resid, self._sigmoid_prime(self.net_outer))
            delta_inner = delta_outer*self.w_outer*self._sigmoid_prime(self.net_inner)
        elif(self.mode == "relu"):
            delta_outer = np.multiply(resid, self._sigmoid_prime(self.net_outer))
            delta_inner = delta_outer*self.w_outer*self._relu_prime(self.net_inner)
        elif(self.mode == "tanh"):
            delta_outer = np.multiply(resid, self._tanh_prime(self.net_outer))
            delta_inner = delta_outer*self.w_outer*self._tanh_prime(self.net_inner)
            
        self.dw2 = (1/n)*np.sum(np.multiply(self.activ_inner, delta_outer), axis = 1).reshape(self.w_outer.shape)
        self.dw1 = (1/n)*np.dot(X.T, delta_inner.T) #calculate the inner back propogation value for dz and then update
        
    #ACTIVATION DERIVATIVES
    def _relu_prime(self, net):
        val = (net>0) *1
        return val
        
    def _sigmoid_prime(self, net):
        return self._sigmoid(net)*(1-self._sigmoid(net))
    
    def _tanh_prime(self, net):
        return 1 - (self._tanh(net))**2
    
    #f(x): Update Weights
    #Input: Learning Rate
    #Purpose: Update the starting weights using the propogation rule
    def _update_wt(self, learning_rate = .001): 
        self.w_inner = self.w_inner - learning_rate*self.dw1 #update using propogation rule
        self.w_outer = self.w_outer - learning_rate*self.dw2 #update using propogation rule
    
    #f(x): Train
    #Input: Beginning X values, observed y values, number of passes
    #Purpose: Make a bunch of passes to train the neural network
    def train(self, X, y, n_epoch = 10): 
        for i in range(n_epoch):
            y_hat = self._fwd_prop(X)
            loss = self._loss(y_hat,y)
            self._back_prop(X,y)
            self._update_wt()
            if i%3 == 0: 
                print("loss: ", loss)
                
     #the decision boundaries are different for each activation function  
    
    #f(x) Predict
    #Input: Starting x values
    #Purpose: Once we've trained the model, make some predictions using a decision boundary
    def predict(self, X):
        y_hat = self._fwd_prop(X)
        if(self.mode == "sigmoid"):
            y_hat = [1 if i >= 0.5 else 0 for i in y_hat.T]
        elif(self.mode == "relu"): 
            y_hat = [1 if i>= 0.5 else 0 for i in y_hat.T]
        elif(self.mode == "tanh"): 
            y_hat = [1 if i >= 0.5 else 0 for i in y_hat.T]
        return np.array(y_hat)
    
    #f(x): Score
    #Input: predictions and observations 
    #Purpose: Score the model to see how well it's doing
    def score(self, predict, y): 
        correct = np.sum(predict ==y)
        return correct/(len(y))
        
    


# In[5]:


nn = NN_scratch("tanh",df)
x_train, x_test, y_train, y_test = nn.xy_split()
nn.train(x_train, y_train.values)
pred_y_train = nn.predict(x_train)
pred_y_test = nn.predict(x_test)

accuracy_train = nn.score(pred_y_train, y_train.values)
accuracy_test = nn.score(pred_y_test, y_test.values)
print('predictions.head:',  pred_y_test[1:10])
print('observed.head:', y_test.values[1:10])
print('training error:', 1- accuracy_train)
print('estimated test error:', 1- accuracy_test)


# ### Works Cited 
# Fixing the Relu Derivative Array Issue 
# https://stackoverflow.com/questions/46411180/implement-relu-derivative-in-python-numpy
# 
# Fixing the Max vs Maximum Issue for Relu
# https://stackoverflow.com/questions/44957704/passing-relu-function-to-all-element-of-a-numpy-array
# 
# Nitty Gritty Logic for the Neural Network - First Attempt and Background 
# https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
# 
# Simplified Way to Code Neural Network 
# https://medium.com/@qempsil0914/implement-neural-network-without-using-deep-learning-libraries-step-by-step-tutorial-python3-e2aa4e5766d1
# 
# What Loss Function for Binary Classification
# https://www.analyticsvidhya.com/blog/2021/03/binary-cross-entropy-log-loss-for-binary-classification/#:~:text=In%20this%20article%2C%20we%20will,used%20for%20binary%20classification%20problems
# 
# Decision Boundaries for Different Classifiers
# https://medium.com/analytics-vidhya/activation-functions-in-neural-network-55d1afb5397a
# 
# Tanh
# https://datascience.stackexchange.com/questions/109547/why-does-using-tanh-worsen-accuracy-so-much
# 
# Small Learning Rate for Tanh and Relu
# https://stats.stackexchange.com/questions/324896/training-loss-increases-with-time
# 
# Kaggle Dataset
# https://www.kaggle.com/datasets/mojtaba142/hotel-booking/
