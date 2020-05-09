#!/usr/bin/env python
# coding: utf-8

# ## Word2Vec testings

# ##### Have defined 3 functions for the performing the word2vec and feature selection using the same efficiently, with the flexibility of selecting the threshold for the amount of words to be considered as a deciding factor for the feature selection.

# In[1]:


import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.word2vec import H2OWord2vecEstimator
import random, os, sys
from datetime import datetime
import pandas as pd
import csv
import time
from distutils.util import strtobool
import psutil
import numpy as np
from matplotlib import pyplot as plt
import statistics
import warnings
warnings.filterwarnings("ignore")


# In[2]:


pct_memory=0.5
virtual_memory=psutil.virtual_memory()
min_mem_size=int(round(int(pct_memory*virtual_memory.available)/1073741824,0))
print(min_mem_size)


# In[3]:


h2o.init(strict_version_check=False,max_mem_size=min_mem_size)


# In[4]:


data_path = "data/AmazonReviews.Small.csv"
reviews = h2o.import_file(data_path)


# In[5]:


print(reviews.shape)
print(reviews.head(3))


# ##### Below are the STOP_WORDS that will be useful in the analysis. The location for the same to be downloaded is as given
# https://raw.githubusercontent.com/h2oai/h2o-tutorials/master/h2o-world-2017/nlp/stopwords.csv

# In[6]:


# Stopwords for the testings is included
# Below code tests if Stopping word is present in the local machine, if not then it finds it at the specific GitHub 

data_path = "data/stopwords.csv"
if os.path.isfile(data_path):
    data_path = data_path
else:
    data_path = "https://raw.githubusercontent.com/h2oai/h2o-tutorials/master/h2o-world-2017/nlp/stopwords.csv"

STOP_WORDS = pd.read_csv(data_path, header=0)
STOP_WORDS = list(STOP_WORDS['STOP_WORD'])


# ### Functions
# 
# Few functions that has been build to work with the testings. Below are the list and explanation for the same of it's working. With the input and return explained for the functions
# 
# * <b>w2v_col</b> - This function helps us get the list of all the columns which has the text above a threshold that is given as a input `w_threshold`. And the data frame should be `pandas df` data frame given as an input for it to find it.
# * <b>w2v_col_avg</b> - This function helps us get the list of all the columns which has the text above a threshold average count of characters in the column that is given as an input `w_threshold`. And the data frame should be `pandas df` data frame given as an input for it to find it.
# * <b>tokenize</b> - This function goes through the text string in each columns and tokenize it to words based on the STOP_WORDS that is given to it. This is used and called in the 3rd function below
# * <b>word2vec_func_df</b> - This function helps us get the data frame with all the word converted to vector and aggregated with the `AVERAGE` method. This takes output of first function which is the list of all columns to be tokenized and takes the threshold amount of columns to be vecotrized in the result for each of the columns being considered, also takes the H2O Dataframe as an input for tokenizing and vectoring it. As the output it returns the combined data frame consisting all the words into vectors for our further analysis

# In[7]:


# A function to return the columns that has some threshold amount of words, 
# so that they can be considered for word2vec training and adding as features to the dataset
def w2v_col(df, w_threshold):
    do_w2v=0
    w2v_col=[]
    for col in df.columns:
        do_w2v=0
        try:
            for val in df[col].str.len():
                if val > w_threshold:
                    do_w2v=1
            if do_w2v==1:
                w2v_col.append(col)
        except:
            print("Skipped: "+col)
    print("List of text field above "+str(w_threshold)+": "+str(w2v_col))
    return w2v_col

# Second version of the function considering the average count of characters in the column
def w2v_col_avg(df, w_threshold):
    do_w2v=0
    w2v_col=[]
    for col in df.columns:
        do_w2v=0
        try:
            if int(statistics.mean(df[col].str.len()))>=w_threshold:
                do_w2v=1
                w2v_col.append(col)
                print("Min count of characters = "+ str(min(df['Text'].str.len())) + " in {} column of the dataset".format('Text'))
                print("Max count of characters = "+ str(max(df['Text'].str.len())) + " in {} column of the dataset".format('Text'))
                print("Avg count of characters = "+ str(int(statistics.mean((df['Text'].str.len())))) + " in {} column of the dataset".format('Text'))
                print("***************************")
        except:
            print("Skipped: "+col)
    print("List of text field above "+str(w_threshold)+": "+str(w2v_col))
    return w2v_col

# Function to tockenize the words compared to the stop words
def tokenize(sentences, stop_word = STOP_WORDS):
    tokenized = sentences.tokenize("\\W+")
    tokenized_lower = tokenized.tolower()
    tokenized_filtered = tokenized_lower[(tokenized_lower.nchar() >= 2) | (tokenized_lower.isna()),:]
    tokenized_words = tokenized_filtered[tokenized_filtered.grep("[0-9]",invert=True,output_logical=True),:]
    tokenized_words = tokenized_words[(tokenized_words.isna()) | (~ tokenized_words.isin(STOP_WORDS)),:]
    return tokenized_words

# word2vec training and getting a dataset as a final with all the vectors binded with the original dataset
def word2vec_func_df(arr, vec_size, df):
    for col in arr:
        words = tokenize(df[col].ascharacter())
        model_id= "w2v_"+col+".hex"
        w2v_model = H2OWord2vecEstimator(vec_size = vec_size, model_id = model_id)
        w2v_model.train(training_frame=words)
        models_path='/data'
        h2o.save_model(w2v_model, path = models_path, force = True)
        print(w2v_model.find_synonyms("coffee", count = 10))
        print(w2v_model.find_synonyms("tea", count = 10))
        df_vec = w2v_model.transform(words, aggregate_method = "AVERAGE")
        df_vec.names = [col + s for s in df_vec.names]
        ext_df = df.cbind(df_vec)
        df = ext_df
    return ext_df


# In[8]:


# Used pandas to find the columns that have a threshold amount of words to be considered

df=pd.read_csv("data/AmazonReviews.Small.csv")
w2vcol=w2v_col_avg(df, 100)
w2vcol


# In[9]:


reviews["PositiveReview"] = (reviews["Score"] >= 4).ifelse("1", "0")


# In[10]:


reviews.head()


# We first build a model based on the original dataset to evaluate it's metric over AUC

# In[11]:


train,test,valid = reviews.split_frame(ratios=[.7, .15])


# In[12]:


from h2o.estimators import H2OGradientBoostingEstimator

predictors = ['ProductId', 'UserId', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time']
response = 'PositiveReview'

gbm_baseline = H2OGradientBoostingEstimator(stopping_metric = "AUC", stopping_tolerance = 0.001,
                                            stopping_rounds = 5, score_tree_interval = 10,
                                            model_id = "gbm_baseline.hex"
                                           )
gbm_baseline.train(x = predictors, y = response, 
                   training_frame = train, validation_frame = test
                  )


# In[13]:


print("AUC on Validation Data: " + str(round(gbm_baseline.auc(valid = True), 3)))


# ##### Used the function defined above to get the new features out of the text fields and appending it to the original dataset for further usage

# In[14]:


# Used the function defined above to get the new features out of the texy fields and appending it to the riginal dataset 
# for further usage

ext_reviews=word2vec_func_df(w2vcol, 50, reviews)
ext_reviews


# In[15]:


ext_train,ext_test,ext_valid = ext_reviews.split_frame(ratios=[.7, .15])


# ### We build the model over the dataset consisting of all the word2vec as a features. And test how it improves our AUC

# In[16]:


predictors = predictors + ext_reviews.names
response = 'PositiveReview'

gbm_embeddings = H2OGradientBoostingEstimator(stopping_metric = "AUC", stopping_tolerance = 0.001,
                                              stopping_rounds = 5, score_tree_interval = 10,
                                              model_id = "gbm_embeddings.hex"
                                             )
gbm_embeddings.train(x = predictors, y = response, 
                   training_frame = ext_train, validation_frame = ext_test
                  )


# In[17]:


print("Baseline AUC: " + str(round(gbm_baseline.auc(valid = True), 3)))
print("With Embeddings AUC: " + str(round(gbm_embeddings.auc(valid = True), 3)))


# In[18]:


# h2o.cluster().shutdown()

