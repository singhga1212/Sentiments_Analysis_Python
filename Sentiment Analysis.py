#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


## load data set
SA = pd.read_csv('linkedin-reviews.csv')
SA


# In[3]:


#EDA (Exploratory Data Analysis)

plt.figure(figsize=(9, 5))
sns.countplot(data=SA, x='Rating')

plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel ('Count')


# In[4]:


#pip install -U seaborn


# In[5]:


#Length of review corelates with consumers sentiments
SA['Review Length'] = SA['Review'].apply(len)

# Plotting the distribution of review lengths
plt.figure(figsize=(9, 6))
sns.histplot(SA['Review Length'], bins=50, kde=True)
plt.title('Distribution of Review Lengths')
plt.xlabel('Length of Review')
plt.ylabel('Count')
plt.show()


# In[6]:


#pip install textblob


# In[7]:


from textblob import TextBlob

def textblob_sentiment_analysis(review):
    # Analyzing the sentiment of the review
    sentiment = TextBlob(review).sentiment
    # Classifying based on polarity
    if sentiment.polarity > 0.1:
        return 'Positive'
    elif sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

# Applying TextBlob sentiment analysis to the reviews
SA['Sentiment'] = SA['Review'].apply(textblob_sentiment_analysis)

SA

#The dataset now includes sentiment labels for each review, classified as Positive, Negative, or Neutral based on the polarity score calculated by TextBlob.


# In[8]:


# Analyzing the distribution of sentiments
sentiment_distribution = SA['Sentiment'].value_counts()

# Plotting the distribution of sentiments
plt.figure(figsize=(9, 5))
sns.barplot(x=sentiment_distribution.index, y=sentiment_distribution.values)

plt.title('Distribution of Sentiments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()


# In[9]:


plt.figure(figsize=(10, 5))
sns.countplot(data=SA, x='Rating', hue='Sentiment')
plt.title('Sentiment Distribution Across Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.show()


# In[10]:


#pip install wordcloud


# In[11]:


from wordcloud import WordCloud

# Function to generate word cloud for each sentiment
def generate_word_cloud(sentiment):
    text = ' '.join(review for review in SA[SA['Sentiment'] == sentiment]['Review'])
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {sentiment} Reviews')
    plt.axis('off')
    plt.show()

# Generating word clouds for each sentiment
for sentiment in ['Positive', 'Negative','Neutral']:
    generate_word_cloud(sentiment)

