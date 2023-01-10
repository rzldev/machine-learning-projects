## Keyword Extraction ##

## Importing the libraries
import numpy as np
import pandas as pd

## Importing the dataset
dataset = pd.read_csv('../data/keyword-extraction/papers.csv')

## Data Preprocessing
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stopwords = set(stopwords.words('english'))

# Create a list of custom stopwords
new_words = ['fig', 'figure', 'image', 'sample', 'using', 'show', 'result', 'large',
             'also', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
             'nine']
stopwords = list(stopwords.union(new_words))

def preprocessing(text):
    # Lower case
    text = text.lower()
    
    # Remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)
    
    # Remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ", text)

    # Conver to list
    text = text.split()    

    # Remove stopwords
    text = [word for word in text if word not in stopwords]

    # Remove words less than 3 chars
    text = [word for word in text if len(word) >= 3]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    
    return ' '.join(text)

docs = dataset['paper_text'].apply(lambda x: preprocessing(x))

## Create the word count with TF-IDF
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df=.95, max_features=10000, ngram_range=(1, 3))
word_count_vector = cv.fit_transform(docs)

## Calculate the reverse frequency of documents
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

## Keyword extraction with TF-IDF vectorization

# Get feature names
feature_names = cv.get_feature_names_out()

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_top_n_from_vector(feature_names, sorted_items, top_n=10):
    '''Get the feature names and tf-idf score of top n items.'''
    
    # Use only top n items from vector
    sorted_items = sorted_items[:top_n]
    
    score_vals = []
    feature_vals = []
    
    # Keep track of feture name and it's coresponding score
    for i, score in sorted_items:        
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[i])
        
    # Create tuples of feature and score
    results = {}
    for i in range(len(feature_vals)):
        results[feature_vals[i]] = score_vals[i]
    
    return results


def get_keywords(i, docs):
    # Generate tf-idf for the given document
    tfidf_vector = tfidf_transformer.transform(cv.transform(docs))
    # Sort the tf-idf vectors by decending orders of scores
    sorted_items = sort_coo(tfidf_vector.tocoo())
    # Extract only the top n items
    keywords = extract_top_n_from_vector(feature_names, sorted_items)
    
    return keywords

def print_results(i, keywords, df):
    print('\n================Title===============')
    print({df['title'][i]})
    print('==============Abstract==============')
    print({df['abstract'][i]})
    print('==============Keywords==============')
    for k in keywords:
        print(k, keywords[k])

i = 101
keywords = get_keywords(i, docs)
print_results(i, keywords, dataset)

