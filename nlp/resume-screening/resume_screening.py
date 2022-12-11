## Resume Screening with NLP ##

## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
dataset = pd.read_csv('UpdatedResumeDataSet.csv', encoding='utf-8')

## Show necessary informations
print('Displaying the distinct categories of resume:')
print(dataset['Category'].unique())

print('\nDisplaying the distinct categories of resume and the number of records belonging to each category:')
print(dataset['Category'].value_counts())

## Visualize the number of categories in the dataset
import seaborn as sns
plt.figure(figsize=(10, 10))
plt.xticks(rotation=90)
sns.countplot(y='Category', data=dataset)
plt.show()

## Visualize the distribution of categories
from matplotlib.gridspec import GridSpec
target_counts = dataset['Category'].value_counts()
target_labels = dataset['Category'].unique()

# Make square figures and axes
plt.figure(1, figsize=(20, 20))
the_grid = GridSpec(2, 2)

cmap = plt.get_cmap('coolwarm')
colors = [cmap(i) for i in np.linspace(0, 1, 3)]
plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')

sourcepie = plt.pie(target_counts, labels=target_labels, autopct='%1.1f%%', shadow=True, colors=colors)
plt.show()

## Helper function to clean the data from noise
import re
def clean_data(data):
    data = re.sub(r'[^a-zA-Z0-9]', r' ', data) # Remove Punctuations
    return data

dataset['Resume'] = dataset.Resume.apply(lambda x: clean_data(x))

## Creating the Wordcloud
import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('punkt')
set_of_stopwords = set(stopwords.words('english')+['``',"''"])
total_words = []
sentences = dataset['Resume'].values
cleaned_sentences = ''
for i in range(0, len(sentences)):
    cleaned_text = clean_data(sentences[1])
    cleaned_sentences += cleaned_text
    required_words = nltk.word_tokenize(cleaned_text)
    for word in required_words:
        if word in set_of_stopwords and word not in string.punctuation:
            total_words.append(word)
            
word_freq_dist = nltk.FreqDist(cleaned_sentences)
most_common_words = word_freq_dist.most_common(50)

wc = WordCloud().generate(cleaned_sentences)
plt.figure(figsize=(15, 15))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
        
## Encoding (data preprocessing)
from sklearn.preprocessing import LabelEncoder
sc = LabelEncoder()
dataset['Category'] = sc.fit_transform(dataset['Category'])

## Create dataset features
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

required_text = dataset['Resume'].values
required_target = dataset['Category'].values
word_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                  stop_words='english',
                                  max_features=1500)
word_vectorizer.fit(required_text)
word_features = word_vectorizer.transform(required_text)
print('Feture Completed...')

## Spltting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(word_features, 
                                                    required_target, 
                                                    test_size=.2, 
                                                    random_state=0)

## Create and train KNN model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

## Create prediction
from sklearn import metrics
prediction = clf.predict(X_test)
print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))

