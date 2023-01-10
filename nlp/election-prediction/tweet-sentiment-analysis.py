## Election Prediction with Tweet Sentiment Analysis ##

## Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

## Importing the dataset
trump_reviews = pd.read_csv('../data/twitter-sentiment/Trumpall2.csv')
print(trump_reviews.head())

biden_reviews = pd.read_csv('../data/twitter-sentiment/Bidenall2.csv')
print(biden_reviews.head())

## Sentiment analysis
from textblob import TextBlob

textblob_trump = TextBlob(trump_reviews['text'][100])
print('Trump: ', textblob_trump.sentiment)
textblob_biden = TextBlob(biden_reviews['text'][100])
print('Biden: ', textblob_biden.sentiment)

def get_polarity(review):
    return TextBlob(review).sentiment.polarity

trump_reviews['polarity'] = trump_reviews['text'].apply(get_polarity)
print(trump_reviews.tail())
biden_reviews['polarity'] = biden_reviews['text'].apply(get_polarity)
print(biden_reviews.tail())

## Add sentiment label into the dataset
trump_reviews['label'] = np.where(trump_reviews['polarity'] > 0, 'positive', 'negative')
trump_reviews['label'][trump_reviews['polarity'] == 0] = 'neutral'
print(trump_reviews.tail())

biden_reviews['label'] = np.where(biden_reviews['polarity'] > 0, 'positive', 'negative')
biden_reviews['label'][biden_reviews['polarity'] == 0] = 'neutral'
print(biden_reviews.tail())

## Data cleaning
trump_temp = trump_reviews[trump_reviews['polarity'] == 0.0000]
trump_cond = trump_reviews['polarity'].isin(trump_temp['polarity'])
trump_reviews = trump_reviews.drop(trump_reviews[trump_cond].index)
print(trump_temp.shape, trump_reviews.shape)

biden_temp = biden_reviews[biden_reviews['polarity'] == 0.0000]
biden_cond = biden_reviews['polarity'].isin(biden_temp['polarity'])
biden_reviews = biden_reviews.drop(biden_reviews[biden_cond].index)
print(biden_temp.shape, biden_reviews.shape)

## Balancing the dataset
np.random.seed(10)
remove_n = len(trump_reviews) - 1000
drop_indices = np.random.choice(trump_reviews.index, remove_n, replace=False)
df_subset_trump = trump_reviews.drop(drop_indices)
print(df_subset_trump.shape)

remove_n = len(biden_reviews) - 1000
drop_indices = np.random.choice(biden_reviews.index, remove_n, replace=False)
df_subset_biden = biden_reviews.drop(drop_indices)
print(df_subset_biden.shape)

## Create a summary plot 
count_trump = df_subset_trump.groupby('label').count()
trump_positive_reviews = (count_trump['polarity'][1] / 1000) * 100
trump_negative_reviews = (count_trump['polarity'][0] / 1000) * 100
print(trump_positive_reviews, trump_negative_reviews)

count_biden = df_subset_biden.groupby('label').count()
biden_positive_reviews = (count_biden['polarity'][1] / 1000) * 100
biden_negative_reviews = (count_biden['polarity'][0] / 1000) * 100
print(biden_positive_reviews, biden_negative_reviews)

politicians = ['Joe Biden', 'Donald Trump']
plot_reviews = [[trump_positive_reviews, biden_positive_reviews],
                [trump_negative_reviews, biden_negative_reviews]]

items = np.arange(len(politicians))
width = .25
fig1, ax = plt.subplots()
plt.bar(items - (width / 2), plot_reviews[0], width, color='green')
plt.bar(items + (width / 2), plot_reviews[1], width, color='red')
plt.xticks(items, politicians)
plt.title('Election Prediction')
plt.xlabel('Sentiments')
plt.ylabel('Politicians')
plt.legend(['Positive', 'Negative'])
plt.xlim(-.5, len(politicians))
plt.show()
