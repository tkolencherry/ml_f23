#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import requests
import io
from sklearn.model_selection import train_test_split 

#Part 2 libraries used
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error


# ### Pre-Processing

# In[2]:


url = "https://raw.githubusercontent.com/tkolencherry/ml_f23/main/HW_1/garments_worker_productivity.csv"
file = requests.get(url).content
prod_data = pd.read_csv(io.StringIO(file.decode('utf-8')))
prod_data.head()


# In[3]:


prod_data.describe()


# In[4]:


prod_data.info()
#both info and describe because some of the data is categorial
#there are missing entries from wip (the number of unfinished iterms for products)


# In[5]:


prod_data.loc[prod_data["wip"] == 0]

#since there are no rows where the works in progress are zero, it seems acceptable to
#fill in zeroes for the null values


# In[6]:


prod_data = prod_data.fillna(0)
prod_data.info()


# In[7]:


#now I want to check the quarter column
prod_data.quarter.value_counts()
#how are there five quarters? 


# In[8]:


prod_data.loc[prod_data.quarter == "Quarter5"]


# In[9]:


prod_data.loc[prod_data.quarter == "Quarter1"].date.value_counts()


# In[10]:


#change all of the Q5 entries to Q1 when we map
q_dict = {"Quarter1":1, "Quarter2":2, "Quarter3":3, "Quarter4":4, "Quarter5":1}

prod_data = prod_data.replace({'quarter':q_dict})
prod_data.head()


# In[11]:


prod_data.department.value_counts()


# In[12]:


prod_data.loc[prod_data.department == "finishing"]


# In[13]:


prod_data.loc[prod_data.department == "finishing "]


# In[14]:


#Sewing is 1 and Finishing is 2
dep_dict = {"sweing":1, "finishing ": 2, "finishing":2}
prod_data = prod_data.replace({'department':dep_dict})
prod_data.department.value_counts()


# In[15]:


prod_data.day.value_counts()


# In[16]:


#Sewing is 1 and Finishing is 2
day_dict = {"Monday":1, "Tuesday": 2, "Wednesday":3, "Thursday": 4, "Saturday": 5, "Sunday": 6,}
prod_data = prod_data.replace({'day':day_dict})
prod_data.day.value_counts()


# In[17]:


prod_data.corr() 


# In[18]:


prod_data.corr().actual_productivity.sort_values(ascending = False)

#let's pick our indicators - targeted_productivity, no_of_style_change, idle_men, quarter, and team
#our response variable is actual_productivity
#didn't select smv because it has such high correlation with factors already selected


# In[19]:


df = prod_data[["quarter", "targeted_productivity", "no_of_style_change", "idle_men", "team", "actual_productivity"]]
df.describe()


# In[20]:


x_quarter = df.quarter
x_targeted_productivity = df.targeted_productivity
x_no_of_style_change = df.no_of_style_change
x_idle_men = df.idle_men
x_team = df.team

y = df.actual_productivity


# In[21]:


plt.scatter( x_quarter, y, s =5, label = "quarter")
plt.scatter( x_targeted_productivity,y, s =5, label = "targeted productivity")
plt.scatter(x_no_of_style_change, y,  s =5, label = "# of style changes")
plt.scatter(x_idle_men, y, s =5, label = "idle men")
plt.scatter(x_team, y,  s =5, label = "team size")
plt.legend(fontsize = 15)
plt.xlabel('Predictors', fontsize =15)
plt.ylabel("Actual Productivity Score", fontsize = 15)
plt.legend()
plt.show()

#if there is time, would be good to show a scatter plot but with the conditional means 
# E[Y|Quarter = 1] vs E[Y|Quarter = 2]


# ### Gradient Descent Class

# In[22]:


class GSD_5:
    p1 = np.arange(0.000001,0.00001999,.0000001)
    
    def __init__(self, df, response_variable, n_pred,mode = "large"):
        self.df = df
        self.response = response_variable
        self.mode = mode
        self.npred = n_pred
        self.weights = []
        
    def xy_split (self):
        df_x = self.df.drop(self.response, axis = 1)
        df_y = self.df[self.response]

        x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, random_state = 42, test_size = 0.2)
    
        x_test = np.c_[np.ones(x_test.count()[0]), x_test]
        x_train = np.c_[np.ones(x_train.count()[0]), x_train]
    
        return x_train, x_test, y_train, y_test
    
    def gradient(self, x,y,w):
        if self.npred == 5:
            y_hat = np.dot(w,x.T) #y_hat = w0(1) + w1x1 + w2x2 +...
            residual = y_hat - y
            residual = residual.values
            x_0 = x[:, :1]
            x_1 = x[:, 1:2]
            x_2 = x[:, 2:3]
            x_3 = x[:, 3:4]
            x_4 = x[:, 4:5]
            x_5 = x[:, 5:6]
        #dot = np.dot(x_0.T,residual) - used for debugging
            return residual.mean(), np.dot(x_0.T,residual).mean(), np.dot(x_1.T,residual).mean(), np.dot(x_2.T,residual).mean(), np.dot(x_3.T,residual).mean(), np.dot(x_4.T,residual).mean(), np.dot(x_5.T,residual).mean()
        elif self.npred == 3: 
            y_hat = np.dot(w,x.T) #y_hat = w0(1) + w1x1 + w2x2 +...
            residual = y_hat - y
            residual = residual.values
            x_0 = x[:, :1]
            x_1 = x[:, 1:2]
            x_2 = x[:, 2:3]
            x_3 = x[:, 3:4]

            return residual.mean(), np.dot(x_0.T,residual).mean(), np.dot(x_1.T,residual).mean(), np.dot(x_2.T,residual).mean(), np.dot(x_3.T,residual).mean()
        elif self.npred == 1:
            y_hat = np.dot(w,x.T) #y_hat = w0(1) + w1x1 + w2x2 +...
            residual = y_hat - y
            residual = residual.values
            x_0 = x[:, :1]
            x_1 = x[:, 1:2]
            
            return residual.mean(), np.dot(x_0.T,residual).mean(), np.dot(x_1.T,residual).mean(),
            
    def gd(self, gradient, x, y, start, learn_rate=0.1, n_iter = 50, tolerance = .01):
        vector = start
        diff = 0
        for i in range(n_iter):
        
            diff = learn_rate*np.array(self.gradient(x,y,vector))  

            if np.all(np.abs(diff) <= tolerance):
                break
            else:
                if(self.npred == 5):
                    vector -= diff[1:7] #the first column is the mean residuals, the rest are the partial derivatives w/ respect to x_i
                elif(self.npred == 3):
                    vector -= diff[1:5]
                elif(self.npred == 1):
                    vector -= diff[1:3]
        return vector

    #function for returning assessment measures
    def lr_assessment (self, weights,x_train, y_train, x_observed, y_test): 
        train_yhat = []
        test_yhat = []
        n_train = len(y_train)
        n_test = len(y_test)
    
        #use our suggested weights to generate a y-hat array for train values
        for subset in x_train:
            if self.npred ==5: 
                temp_yhat = weights[0] + weights[1]*subset[1] + weights[2]*subset[2] + weights[3]*subset[3] + weights[4]*subset[4] + weights[5]*subset[5]
            elif self.npred ==3: 
                temp_yhat = weights[0] + weights[1]*subset[1] + weights[2]*subset[2] + weights[3]*subset[3] 
            elif self.npred ==1: 
                temp_yhat = weights[0] + weights[1]*subset[1]
            
            train_yhat.append(temp_yhat)
    
        #use our suggested weights to generate a y-hat array for test values
        for subset in x_test:
            if self.npred ==5: 
                temp_yhat = weights[0] + weights[1]*subset[1] + weights[2]*subset[2] + weights[3]*subset[3] + weights[4]*subset[4] + weights[5]*subset[5]
            elif self.npred ==3: 
                temp_yhat = weights[0] + weights[1]*subset[1] + weights[2]*subset[2] + weights[3]*subset[3] 
            elif self.npred ==1: 
                temp_yhat = weights[0] + weights[1]*subset[1]
                
            test_yhat.append(temp_yhat)
            
        #then calculate the training MSE
        train_SE = ((train_yhat-y_train).values)**2
        train_MSE = train_SE.sum()/n_train


        #then calculate the training MSE
        test_SE = ((test_yhat-y_test).values)**2
        test_MSE = test_SE.sum()/n_test
    
        return train_MSE, test_MSE, train_yhat, test_yhat
    
    def loop_learning(self, x_train, x_test, y_train, y_test,  seed, upper, learn_array, n_iter, threshold = 0.1):
    #generate random weights (seed = 175)
        self.min_test = 1000000
        self.min_p = 1000000
        test_MSEs = []
        train_MSEs = []
        train_r2_adjs = []
        test_r2_adjs = []
        np.random.seed(seed)
        
        if self.mode == "large":
            temp_weights = np.random.randint(0,upper,(self.npred+1))*1.0 # we need w0, w1, w2, w3, w4, w5, w6
            beg_wt = temp_weights.copy()
        else: 
            temp_weights = np.random.rand(self.npred +1)
            temp_weights = temp_weights.astype(float)
            beg_wt = temp_weights.copy()

#n = 100
        for i in learn_array:
            sugg_weights = self.gd(self.gradient, x_train, y_train, temp_weights, i, n_iter, threshold)
            train_MSE, test_MSE, train_yhat, test_yhat = self.lr_assessment(sugg_weights,x_train, y_train, x_test, y_test) 
            train_MSEs.append(train_MSE)
            test_MSEs.append(test_MSE)
            
            if test_MSE < self.min_test: 
                self.min_test = test_MSE
                self.min_p = i
        print("****MODEL PARAMETERS**** \n", )
        print("Weights")
        for l in range(self.npred+1): 
            print("w_",l," : ", beg_wt[l], "------->", sugg_weights[l])
            
        print("Iterations : ", n_iter, "\n Threshold : ", threshold, "\n Number of Predictors : ", self.npred)
        print("\n Minimum Test MSE",self.min_test, " for Learning Rate = ",self.min_p) 
        plt.plot(self.p1,train_MSEs, label = "Training MSE")
        plt.plot(self.p1,test_MSEs, label = "Test MSE")
        plt.title("Training vs Test Errors for Varying Learning Rates")
        plt.legend(loc = "upper right")
        plt.xlabel("Learning Rate")
        plt.ylabel("Error Rate")
        plt.show()
        
        self.min_test = 1000000
        self.min_p = 1000000
        return
        
    def regression(self, x_train, x_test, y_train, y_test, learn_array, learning_type, max_iter, threshold):
        train_mses = []
        test_mses = []
        min_test = 1000
        min_p = 1000

        for i in learn_array:
            model = SGDRegressor(alpha=i, eta0=0.001, learning_rate = learning_type, max_iter = max_iter, tol = threshold)
            model.fit(x_train, y_train)
            y_train_predict = model.predict(x_train)
            train_mse = mean_squared_error(y_train, y_train_predict)
            y_test_predict = model.predict(x_test)
            test_mse = mean_squared_error(y_test, y_test_predict)
            train_mses.append(train_mse)
            test_mses.append(test_mse)
            if test_mse < min_test: 
                min_test = test_mse
                min_p = i

        print("WEIGHT ESTIMATION")
        for l in range(self.npred+1): 
            print("w_",l," : ", model.coef_[l])
   
        print("\n Minimum Test MSE",min_test, " for Learning Rate = ", min_p) 
        plt.plot(self.p1,train_mses, label = "Training MSE")
        plt.plot(self.p1,test_mses, label = "Test MSE")
        plt.title("Training vs Test Errors for Varying Learning Rates")
        plt.legend(loc = "upper right")
        plt.xlabel("Learning Rate")
        plt.ylabel("Error Rate")
        plt.show()
        
    def residual_plot(self, weights, x_train, x_test, y_train, y_test): 
        train_yhat = []
        test_yhat = []
        #use our suggested weights to generate a y-hat array for train values
        for subset in x_train:
            if self.npred ==5: 
                temp_yhat = weights[0] + weights[1]*subset[1] + weights[2]*subset[2] + weights[3]*subset[3] + weights[4]*subset[4] + weights[5]*subset[5]
            elif self.npred ==3: 
                temp_yhat = weights[0] + weights[1]*subset[1] + weights[2]*subset[2] + weights[3]*subset[3] 
            elif self.npred ==1: 
                temp_yhat = weights[0] + weights[1]*subset[1]
            
            train_yhat.append(temp_yhat)
    
        #use our suggested weights to generate a y-hat array for test values
        for subset in x_test:
            if self.npred ==5: 
                temp_yhat = weights[0] + weights[1]*subset[1] + weights[2]*subset[2] + weights[3]*subset[3] + weights[4]*subset[4] + weights[5]*subset[5]
            elif self.npred ==3: 
                temp_yhat = weights[0] + weights[1]*subset[1] + weights[2]*subset[2] + weights[3]*subset[3] 
            elif self.npred ==1: 
                temp_yhat = weights[0] + weights[1]*subset[1]
                
            test_yhat.append(temp_yhat)
        
        train_resids = (y_train - train_yhat)
        test_resids = (y_test - test_yhat)
            
        plt.scatter(train_yhat, train_resids) 
        plt.title("Residual Plot for Model Training")
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel("Predicted Y")
        plt.ylabel("Residuals")
        plt.show()
        
        plt.scatter(test_yhat, test_resids) 
        plt.title("Residual Plot for Model Testing")
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel("Predicted Y")
        plt.ylabel("Residuals")
        plt.show()


# ### Part 1
# #### Experimenting with Varying Learning Rates, Iterations, and Thresholds

# In[23]:


temp1 = GSD_5(df,"actual_productivity",5)
x_train, x_test, y_train, y_test = temp1.xy_split()
temp1.loop_learning(x_train, x_test, y_train, y_test, 42, 8, temp1.p1, 100)


# In[24]:


temp1.loop_learning(x_train, x_test, y_train, y_test, 42, 20, temp1.p1, 100)


# In[25]:


temp1.loop_learning(x_train, x_test, y_train, y_test, 175, 8, temp1.p1, 100)


# In[26]:


temp1.loop_learning(x_train, x_test, y_train, y_test, 13, 20, temp1.p1, 100)


# In[27]:


temp1.loop_learning(x_train, x_test, y_train, y_test, 99, 20, temp1.p1, 100)


# In[28]:


temp1.loop_learning(x_train, x_test, y_train, y_test, 99, 20, temp1.p1, 100,.1)


# In[29]:


temp1.loop_learning(x_train, x_test, y_train, y_test, 99, 20, temp1.p1, 100, .0001)


# In[30]:


temp1.loop_learning(x_train, x_test, y_train, y_test, 99, 20, temp1.p1, 10000,.0001)


# In[31]:


temp1.loop_learning(x_train, x_test, y_train, y_test, 99, 20, temp1.p1, 100000, .00001)


# #### Experimenting with Lower Starting Weights

# In[32]:


temp2 = GSD_5(df,"actual_productivity", 5, "party time")
temp2.loop_learning(x_train, x_test, y_train, y_test, 99, 20, temp1.p1, 100, .01)


# In[33]:


temp2.loop_learning(x_train, x_test, y_train, y_test, 99, 20, temp1.p1, 100, .00001)


# In[34]:


temp2.loop_learning(x_train, x_test, y_train, y_test, 99, 20, temp1.p1, 10000,.0001)


# In[35]:


temp2.loop_learning(x_train, x_test, y_train, y_test, 99, 20, temp1.p1, 100000, .00001)


# In[36]:


temp2.residual_plot([0.36874433840601695,-0.010558652720481124, 0.6219027836157457, -0.039362528753501866, -0.009114088967741867, -0.008742295191248606],x_train, x_test, y_train, y_test)


# #### Scaling Predictors

# In[69]:


df.boxplot()


# In[ ]:


df.idle_men.value_counts()


# In[ ]:


idle_outliers = df.loc[df.idle_men > 20]
idle_outliers


# In[70]:


temp_df = df.drop([650,654,818,822,841,843,882,1046,1085])
temp_df.loc[temp_df.idle_men > 20]


# In[71]:


print("Correlations Before Removing Outliers in idle_men Predictor \n",df.corr().actual_productivity)
print("\n\n")
print("Correlations After Removing Outliers in idle_men Predictor \n",temp_df.corr().actual_productivity)


# #### Taking out Weak Predictors

# In[37]:


df.corr().drop("actual_productivity")


# In[38]:


#there is a decent positive correlation between style changes and the quarter and targeted productivity. Since targeted productivity has a larger correlation, I'll drop quarter and style changes
adj3_df = df.drop(columns = ["quarter","no_of_style_change"])
adj3_df.head()


# In[39]:


temp3 = GSD_5(adj3_df,"actual_productivity",3, "woohoo")
x_train, x_test, y_train, y_test = temp3.xy_split()
temp3.loop_learning(x_train, x_test, y_train, y_test, 42, 8, temp3.p1, 100000, .00001)


# In[40]:


temp3.residual_plot([0.19966337938691053, 0.8098532191256858, -0.009710500708695136, -0.00841640162667597], x_train, x_test, y_train, y_test)


# In[41]:


adj1_df = df.drop(columns = ["quarter","no_of_style_change","idle_men", "team"])
adj1_df.head()


# In[42]:


temp4 = GSD_5(adj1_df,"actual_productivity",1, "woohoo")
x_train, x_test, y_train, y_test = temp4.xy_split()
temp4.loop_learning(x_train, x_test, y_train, y_test, 42, 8, temp4.p1, 100000, .00001)


# In[43]:


temp4.residual_plot([0.15751497340626586, 0.7902068560712477], x_train, x_test, y_train, y_test)


# ### Part 2
# #### Varying Learning Rates, Iterations, Threshold Values

# In[73]:


temp5 = GSD_5(df,"actual_productivity",5)
x_train, x_test, y_train, y_test = temp5.xy_split()
temp5.regression(x_train, x_test, y_train, y_test, temp5.p1, "invscaling",100, .01)


# In[47]:


temp5.regression(x_train, x_test, y_train, y_test, temp5.p1, "invscaling", 1000, .01)


# In[48]:


temp5.regression(x_train, x_test, y_train, y_test, temp5.p1, "invscaling", 1000, .001)


# In[49]:


temp5.regression(x_train, x_test, y_train, y_test, temp5.p1, "invscaling", 100000, .0001)


# #### Changing the Learning Rate Schedule

# In[50]:


temp5.regression(x_train, x_test, y_train, y_test, temp5.p1, "constant", 100000, .0001)


# In[51]:


temp5.regression(x_train, x_test, y_train, y_test, temp5.p1, "optimal", 100000, .0001)


# In[80]:


temp5.regression(x_train, x_test, y_train, y_test, temp5.p1, "adaptive", 100000, .0001)


# #### Regression with Fewer Predictors

# In[75]:


temp6 = GSD_5(adj3_df,"actual_productivity",3)
x_train, x_test, y_train, y_test = temp6.xy_split()
temp6.regression(x_train, x_test, y_train, y_test, temp6.p1, "adaptive",100000, .0001)


# In[78]:


temp7 = GSD_5(adj1_df,"actual_productivity",1)
x_train, x_test, y_train, y_test = temp7.xy_split()
temp7.regression(x_train, x_test, y_train, y_test, temp7.p1, "invscaling",100000, .0001)


# In[79]:


temp7.regression(x_train, x_test, y_train, y_test, temp7.p1, "invscaling",100000, .5)


# ### Works Cited

# Dataset: 
# https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees
# 
# Gradient Descent Lab from Class: https://colab.research.google.com/drive/1rmblKcJUf0A18GMk7Jx4u6DXnQLf32V6?usp=sharing#scrollTo=QVz-JbxFJXOW
# 
# Gradient Descent in R: (helpful for orienting logic) 
# https://oindrilasen.com/2018/02/compute-gradient-descent-of-a-multivariate-linear-regression-model-in-r/#:~:text=Similar%20to%20the%20Gradient%20Descent%20for%20a%20Univariate,%28i%29%29.%20xj%20%28i%29%20where%20j%20%3D%200%2C1%2C2%E2%80%A6n%20%7D
# 
# Alternate Multiple Linear Regression with Gradient Descent (used generating random weights and using the dot product of A^T * A for squaring the matrix) 
# https://www.kaggle.com/code/rakend/multiple-linear-regression-with-gradient-descent
# 
# R-Squared: 
# https://en.wikipedia.org/wiki/Coefficient_of_determination
# 
# Reading in Hosted File: 
# https://stackoverflow.com/questions/32400867/pandas-read-csv-from-url

# In[ ]:




