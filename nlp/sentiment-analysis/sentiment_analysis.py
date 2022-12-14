## Sentiment Analysis ##

## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
dataset = pd.read_csv('data/Train.csv')
print(dataset.head())

## Visualize the distribution of the data
fig = plt.figure(figsize=(5, 5))
colors = ['skyblue', 'pink']
positive = dataset[dataset['label'] == 1]
negative = dataset[dataset['label'] == 0]
distributed_data = [len(positive), len(negative)]
plt.pie(distributed_data, 
        labels=['Positive', 'Negative'],
        autopct='%1.1f%%',
        shadow=True,
        colors=colors,
        startangle=45,
        explode=(0, .1))
plt.show()

## Cleaning the texts
import re
from nltk.stem.porter import PorterStemmer

def preprocessing(text):
    text = re.sub('(\<[^>]*\>)', '', text)
    text = re.sub('[^A-Za-z0-9]+', ' ',  text.lower())
    return text

dataset['text'] = dataset['text'].apply(preprocessing)
porter = PorterStemmer()

def tokenizer(text):
    return [word != ' ' for word in text.split()]

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split() if word != ' ']

## Creating the WordCloud
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud

nltk.download('stopwords')
stopwords = stopwords.words('english')

positive = dataset[dataset['label'] == 1]['text']
negative = dataset[dataset['label'] == 0]['text']

def wordcloud_draw(data, color='white'):
    words = ' '.join(data)
    cleaned_words = ' '.join([word for word in words.split()
                              if word != 'movie' and word != 'film'])
    wordcloud = WordCloud(stopwords=stopwords,
                          background_color=color,
                          width=2500,
                          height=2000).generate(cleaned_words)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print('Positive words are as follows')
wordcloud_draw(positive)
print('Negative words are as follows')
wordcloud_draw(negative)

## Convert the data into feature matrix
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None, 
                        tokenizer=tokenizer_porter, use_idf=True, norm='l2', smooth_idf=True)
X = tfidf.fit_transform(dataset.text)
y = dataset.label.values

## Split the dataset into the training set and the test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

## Training sentiment analysis with Logistic Regression
from sklearn.linear_model import LogisticRegressionCV
classifier = LogisticRegressionCV(cv=6, scoring='accuracy', random_state=0, n_jobs=-1,
                                  verbose=3, max_iter=500)
classifier.fit(X_train, y_train)

## Predicting the test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), 1))

## Create the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

