#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



def recommend_items(user_input):
    df=pd.read_csv('C:/Users/LENOVO/Desktop/Recommendation project/sample30.csv')


    # Main Dataframe

    # In[3]:


    df.head(3)


    # In[4]:


    df.info()


    # In[5]:


    df.shape


    # In[6]:


    df.isna().sum()


    # In[7]:


    df[df['user_sentiment'].isna()]


    # Fix Sentiment based on ratings

    # In[8]:


    print(df[df['reviews_rating']==5]['user_sentiment'].value_counts())
    print(df[df['reviews_rating']==4]['user_sentiment'].value_counts())
    print(df[df['reviews_rating']==3]['user_sentiment'].value_counts())
    print(df[df['reviews_rating']==2]['user_sentiment'].value_counts())
    print(df[df['reviews_rating']==1]['user_sentiment'].value_counts())


    # In[9]:


    df.loc[df['reviews_rating'] < 3, 'user_sentiment'] = 'Negative'
    df.loc[df['reviews_rating'] > 2.5, 'user_sentiment'] = 'Positive' 


    # In[10]:


    print(df[df['reviews_rating']==5]['user_sentiment'].value_counts())
    print(df[df['reviews_rating']==4]['user_sentiment'].value_counts())


    # **Sentiment Classification**

    # Sentiment Data

    # In[11]:


    df_sentiment=df[['reviews_text','user_sentiment']]


    # In[12]:


    df_sentiment.head(5)


    # In[13]:


    df_sentiment.loc[df_sentiment['user_sentiment'] == 'Positive', 'sentiment'] = 1
    df_sentiment.loc[df_sentiment['user_sentiment'] == 'Negative', 'sentiment'] = 0


    # In[14]:


    df_sentiment.head(-10)


    # In[15]:


    df_sentiment.drop(columns=['user_sentiment'], inplace=True)
    df_sentiment.head(3)


    # In[16]:


    df_sentiment.shape


    # In[17]:


    #Split test and train


    # In[18]:


    


    # In[19]:


    X=df_sentiment['reviews_text']
    Y=df_sentiment['sentiment']


    # In[20]:


    X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.3)


    # In[21]:


    
    vec = TfidfVectorizer()
    vec.fit(X_train)
    x_train=vec.transform(X_train)
    x_test=vec.transform(X_test)


    # **1. Logistic regression
    # 2. Random forest
    # 3. XGBoost
    # 4. Naive Bayes**

    # In[22]:


    
    lr = LogisticRegression()
    lr.fit(x_train, Y_train)


    # In[23]:


    predicted_test=lr.predict(x_test)


    # In[24]:


    


    # In[25]:


    accuracy_score(predicted_test, Y_test)


    # **#Recommendation Part**

    # In[26]:


    df.shape


    # In[27]:


    df.isna().sum()


    # In[28]:


    df=df[~df['reviews_username'].isna()]


    # In[29]:


    df.shape


    # In[30]:


    small_df=df[['id','reviews_username','reviews_rating']]
    small_df.shape


    # In[31]:


    small_df.drop_duplicates(subset=['id', 'reviews_username'], keep='last', inplace=True)


    # In[32]:


    small_df.shape


    # In[33]:


    train, test = train_test_split(small_df, test_size=0.30, random_state=31)


    # In[34]:


    print(train.shape)
    print(test.shape)


    # In[35]:


    train.head(2)


    # In[36]:


    train.nunique()


    # In[37]:


    train.isna().sum()


    # User User Recommendation

    # In[38]:


    df_pivot=train.pivot(index='reviews_username',
        columns='id',
        values='reviews_rating').fillna(0)
    df_pivot.head(5)


    # In[39]:


    df_pivot['AV13O1A8GV-KLJ3akUyj'].value_counts()


    # Create Dummy train

    # In[40]:


    dummy_train = train.copy()


    # In[41]:


    # The movies not rated by user is marked as 1 for prediction. 
    dummy_train['rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)


    # In[42]:


    dummy_train.head(3)


    # In[43]:


    # Convert the dummy train dataset into matrix format.
    dummy_train = dummy_train.pivot(
        index='reviews_username',
        columns='id',
        values='rating'
    ).fillna(1)


    # In[44]:


    dummy_train.head()


    # In[45]:


    dummy_train.shape


    # In[46]:


    df_pivot.shape


    # Similarity Matrix

    # In[47]:


    

    # Creating the User Similarity Matrix using pairwise_distance function.
    user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0
    print(user_correlation)


    # In[48]:


    user_correlation.shape


    # Adjusted Cosine similarity

    # In[49]:


    mean = np.nanmean(df_pivot, axis=1)
    df_subtracted = (df_pivot.T-mean).T


    # In[50]:


    df_subtracted.head()


    # In[51]:


    user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0
    print(user_correlation)


    # In[52]:


    user_correlation[user_correlation<0]=0
    user_correlation


    # In[53]:


    user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
    user_predicted_ratings


    # In[54]:


    user_predicted_ratings.shape


    # In[55]:


    user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
    user_final_rating.head()


    # 

    # In[59]:


#     user_input = (input("Enter your user name"))
#     print(user_input)


    # In[60]:


    d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
    d


    # Mapping with Product name

    # In[61]:


    d = pd.merge(d,df,left_on='id',right_on='id', how = 'left')
    d.head()


    # In[62]:


    d.shape


    # In[63]:


    d['user_sentiment'].value_counts()


    # In[64]:


    UserBasedRecommend=d.drop_duplicates(subset=['id','brand'])


    # In[65]:


    UserBasedRecommend.head()


    # In[66]:


    print("Top 10 recommendations based on similar user")
    print(UserBasedRecommend['name'])


    # Recommend only top 5 positive items based on sentiment classification

    # In[67]:


    UserBasedRecommend['name']


    # In[68]:


    d.head(3)


    # In[69]:


    d.groupby('id').agg({'user_sentiment':['max','count','min']})


    # In[70]:


    top_5prod=d.groupby('id')['user_sentiment'].value_counts().sort_values(ascending=False)[0:5]


    # In[73]:


    top_5prod


    # In[71]:


    top_5name= pd.merge(top_5prod,df,left_on='id',right_on='id', how = 'left')


    # In[72]:


    final_recommendation=str(top_5name.drop_duplicates(subset=['id','brand'])['name'])
    return final_recommendation


# In[ ]:




