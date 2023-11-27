#!/usr/bin/env python
# coding: utf-8

# In[2]:


from pathlib import Path
import pandas as pd
import numpy as np
import re
from copy import deepcopy
import random
import requests
import io


# ## Pre-Processing
# * Drop tweet ID and timestamp
# * Change letters to all lowercase
# * Drop '#' and '@'
# * Drop URLs

# In[3]:


#grab the tweets from the various text files. Since we don't need to build our model based on originating news outlet, I combined the tweets into one text file
#path = r'C:\Users\teris\ml_f23\HW_3\tweets.txt'
#tweets = pd.read_table(path, sep = "|", header = None, on_bad_lines='skip')

url = "https://raw.githubusercontent.com/tkolencherry/ml_f23/main/HW_3/tweets.txt"
file = requests.get(url).content
tweets = pd.read_table(io.StringIO(file.decode('utf-8')), sep = "|", header = None, on_bad_lines = 'skip')

tweets.head()


# In[3]:


#now delete out the tweet id and timestamp
retweets = tweets.drop(columns = [0,1]).drop_duplicates()
retweets = retweets.rename(columns = {0:'ID', 2:'Tweet'})
retweets.head()


# In[4]:


retweets['Clean'] = retweets.apply(lambda row: str(row['Tweet']).lower(),axis=1)
retweets['Clean'] = retweets.apply(lambda row: re.sub("@[A-Za-z0-9_]+","", str(row['Clean'])),axis=1)
retweets['Clean'] = retweets.apply(lambda row: re.sub("#","", str(row['Clean'])),axis=1) #we still want to keep the tagnames, just not the hashtag
retweets['Clean'] = retweets.apply(lambda row: re.sub(r"http\S+","", row['Clean']),axis=1)
retweets['Clean'] = retweets.apply(lambda row: re.sub(r"www.\S+","", row['Clean']),axis=1)


# In[5]:


print(retweets.Tweet[0])
print(retweets.Clean[0])


# In[6]:


#to make my life easier, I'm going to go ahead and pre-process the tweets to form their specific bag of words 
retweets['Clean'] = retweets.apply(lambda row: row['Clean'].split(),axis=1)
print(retweets.Clean[0])


# ## KMeans Algorithm

# In[100]:


retweets_short = retweets.head(1000)
retweets_short.head()


# In[101]:


class kmeansclusters():
    def __init__(self, tweets, k): 
        self.tweets = tweets
        self.n = len(tweets)
        self.k = k
        self.distMatrix = {}
        
        self.clusters = {}
        self.reverse_cluster = {}
        self.cluster_avg = np.zeros(self.k)
        
        self.init_centers = self.centers_setup() #pick the random centers
        #Set up the Jaccard matrix and the Initial Clusters
        self.cluster_setup()
        self.jaccard_matrix()
        
        self.iteration_threshold = 1000
        
    #f(x): jaccard_dist
    #PURPOSE: find the jaccard distance ( 1 - jaccard similarity) between two sets of words
    #OUTPUT: returns a float value between 0 and 1. The closer to 1, the further apart the sets are
    def jaccard_dist(self, setA, setB): 
        setA = set(setA) #intersection and union won't work on lists, must be set format
        setB = set(setB)
        intersection_AB = len(setA.intersection(setB))
        union_AB = len(setA.union(setB))
        if union_AB == 0:
            return 0.0
        return 1 - (intersection_AB/union_AB)
    
    #f(x): jaccard_matrix
    #PURPOSE: create a distance matrix where each tweet is a row and a column. This matrix will help us keep track of which two clusters are closest.
            # This idea is similar to the matrices we used to show which nodes are connected in graphs.
    #OUTPUT: none - the goal is simply to initialize the matrix
    def jaccard_matrix (self):
        for ptA in self.tweets.Clean.index: 
            self.distMatrix[ptA] = {}
            setA = self.tweets.Clean[ptA]
            for ptB in self.tweets.Clean.index: 
                if ptB not in self.distMatrix: 
                    self.distMatrix[ptB] = {}
                setB = self.tweets.Clean[ptB]
                dist_AB = self.jaccard_dist(setA, setB)
                self.distMatrix[ptA][ptB] = dist_AB
                self.distMatrix[ptB][ptA] = dist_AB
    
    #f(x): centers_setup
    #PURPOSE: Randomly choose k tweets to be the initial centers of the cluster using their indices
    #OUTPUT: The k indices of the centers 
    def centers_setup (self): 
        init_centers = self.tweets.sample(self.k, replace = False, weights = None, axis = 0).index
        return np.array(init_centers)
    
    #f(x): cluster_setup
    #PURPOSE: Once the centers have been assigned, initialize the clusters to that center
    #OUTPUT: none
    def cluster_setup (self): 
        
        for tweet in self.tweets.Clean.index: 
            self.reverse_cluster[tweet] = -1 # initially each tweet has no cluster assigned to it
        
        for val in range(self.k):
            self.clusters[val] = {self.init_centers[val]}

            self.reverse_cluster[self.init_centers[val]] = self.init_centers[val]
    #f(x): cluster_update
    #PURPOSE: update the clusters based on who is closes to the new centers
    #OUTPUT: the new clusters and their reverses 
    def cluster_update(self): 
        new_cluster = {}
        new_rev_cluster = {}
        total_cluster_dist = np.zeros(self.k)
        for val in range(self.k): 
            new_cluster[val] = set()
        for ptA in self.tweets.Clean.index: 
            min_distance =  np.inf
            min_cluster = 0
            #Fix this MINIMIZATION
            for idx in range(len(self.init_centers)):                
                ptB = self.init_centers[idx]
                new_dist = self.distMatrix[ptA][ptB]
                    #if the distance for that point to the specified cluster is lower than the current minimum point 
                    #then update the distance and assign it to that cluster instead of using an average metric
                if min_distance > new_dist: 
                    min_distance = new_dist
                    min_cluster = idx
                
            total_cluster_dist[min_cluster] += min_distance
            new_cluster[min_cluster].add(ptA)
            new_rev_cluster[ptA] = min_cluster
            
        for cluster in range(self.k): 
            cluster_length = int(len(self.clusters[cluster]))
            self.cluster_avg[cluster] = total_cluster_dist[cluster]/cluster_length
            
        return new_cluster, new_rev_cluster
    
        #f(x): center_update 
    #PURPOSE: update the centroid of the clusters
    #OUTPUT: the new centroid of the cluster
    def center_update(self): 
        for group_name in self.clusters:
            empty = set()
            if self.clusters[group_name] != empty: 
                length = int(len(self.clusters[group_name]))
                middle = total_cluster_dist[group_name]
                #middle = int(np.median(sorted(self.clusters[group_name])))
                self.init_centers[group_name] = middle
        return
        
    def iteration(self):
        rounds = 1
        new_cluster, new_rev_cluster = self.cluster_update()
        self.clusters = deepcopy(new_cluster)
        self.reverse_cluster = deepcopy(new_rev_cluster)
        while rounds < self.iteration_threshold: 
            new_cluster, new_rev_cluster  = self.cluster_update()
            rounds += 1
            if self.clusters != new_cluster: 
                self.clusters = deepcopy(new_cluster)
                self.reverse_cluster = deepcopy(new_rev_cluster)
                self.center_update()
            else: 
                return 

    def sse_calc(self):
        sum_dist = 0.0
        for center in self.clusters: 
            for pt in self.clusters[center]: 
                tmp_dist = self.distMatrix[center][pt]
                sum_dist += tmp_dist**2
        return sum_dist
    
    def cluster_print(self):
        sse = self.sse_calc()
        print("Sum Squared Distance/SSE: ", sse)
        for cluster in self.clusters: 
            print("Cluster ", cluster + 1, " has ", len(self.clusters[cluster])," tweets.")


# In[102]:


for k in [3,5, 7, 9, 10]:
    print(k, " Clusters: ")
    kmeans = kmeansclusters(retweets_short, k)
    kmeans.iteration()
    kmeans.cluster_print()


# In[ ]:





# ## Works Cited

#  * For loop to read all data files - https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
#  
#  * How to calculate Jaccard similarity - https://www.geeksforgeeks.org/how-to-calculate-jaccard-similarity-in-python/#
#  
#  * Cleaning and Tokenization of texts - https://www.kaggle.com/code/tariqsays/tweets-cleaning-with-python
#  
#  * One implementation of Jaccard k means - https://github.com/findkim/Jaccard-K-Means/blob/master/k-means%2B%2B.py
#  
#  * Another implementation of Jaccard kmeans - https://medium.com/@rishit.dagli/build-k-means-from-scratch-in-python-e46bf68aa875 
#  
#  * How to Improve on the kmeans algorithm in the future - chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://repository.dinus.ac.id/docs/ajar/file_2013-09-26_09:38:06_Catur_Supriyanto,_M.CS__An_efficient_K-Means_Algorithm_integrated_with_Jaccard_Distance_Measure_for_Document_Clustering_-_Shameem,_Raihana_Ferdous_-_2009.pdf

# In[ ]:




