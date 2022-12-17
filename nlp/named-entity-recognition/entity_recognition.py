## Named Entity Recognition ##

## Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the dataset
dataset = pd.read_csv('../data/named-entity-recognition/ner_dataset.csv', encoding='unicode_escape')
print(dataset.head())

## Data preprocessing
from itertools import chain
def get_dict_map(dataset, column):
    vocab = list(set(dataset[column].to_list()))
    data_to_index = {data:idx for idx, data in enumerate(vocab)}
    index_to_data = {idx:data for idx, data in enumerate(vocab)}
    return data_to_index, index_to_data

token_index, _ = get_dict_map(dataset, 'Word')
tag_index, _ = get_dict_map(dataset, 'Tag')
dataset['Word_idx'] = dataset['Word'].map(token_index)
dataset['Tag_idx'] = dataset['Tag'].map(tag_index)

# Group by and collect columns
dataset_fillna = dataset.fillna(method='ffill', axis=0)
dataset_group = dataset_fillna.groupby(
    ['Sentence #'], 
    as_index=False
    )['Word', 'POS', 'Tag', 'Word_idx', 'Tag_idx'].agg(lambda x: list(x))

## Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences, to_categorical

def get_pad_train_test_val(data_group, dataset):
    # Get max token and tag length
    n_token = len(list(set(dataset['Word'].to_list())))
    n_tag = len(list(set(dataset['Tag'].to_list())))
    
    # Pad tokens (X var)
    tokens = dataset_group['Word_idx'].to_list()
    maxlen = max([len(s) for s in tokens])
    pad_tokens = pad_sequences(tokens, 
                              maxlen=maxlen, 
                              dtype='int32', 
                              padding='post', 
                              value=n_token-1)
    
    # Pad tags (y var) and convert it into one hot encoding
    tags = dataset_group['Tag_idx'].to_list()
    pad_tags = pad_sequences(tags, 
                            maxlen=maxlen,
                            dtype='int32',
                            padding='post',
                            value=tag_index['O'])
    n_tags = len(tag_index)
    pad_tags = [to_categorical(i, num_classes=n_tags) for i in pad_tags]
    
    # Split train, test and validation set
    tokens_, test_tokens, tags_, test_tags = train_test_split(pad_tokens, 
                                                              pad_tags, 
                                                              test_size=.2, 
                                                              random_state=0)
    train_tokens, val_tokens, train_tags, val_tags = train_test_split(tokens_, 
                                                                      tags_,
                                                                      test_size=.2,
                                                                      random_state=0)
    
    print(f'''
          train_tokens length: {len(train_tokens)}
          train_tags length: {len(train_tags)}
          test_tokens length: {len(test_tokens)}
          test_tags length: {len(test_tags)}
          val_tokens length: {len(val_tokens)}
          val_tags length: {len(val_tags)}
          ''')
          
    return train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags
    
train_tokens, val_tokens, test_tokens, train_tags, val_tags, test_tags = get_pad_train_test_val(dataset_group, dataset)

## Training the Named Entity Recognition (NER)
import tensorflow
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.utils import plot_model
from numpy.random import seed

seed(1)
tensorflow.random.set_seed(2)
input_dim = len(list(set(dataset['Word'].to_list()))) + 1
output_dim = 64
input_length = max([len(s) for s in dataset_group['Word_idx'].to_list()])
n_tags = len(tag_index)

# To create the model
def get_bilstm_lstm_model():
    model = Sequential()
    
    # Add Embedding layer
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
    
    # Add Bidirectional LSTM
    model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=.2, recurrent_dropout=.2), merge_mode='concat'))
    
    # Add LSTM
    model.add(LSTM(units=output_dim, return_sequences=True, dropout=.5, recurrent_dropout=.5))
    
    # Add TimeDistributed layer
    model.add(TimeDistributed(Dense(n_tags, activation='relu')))
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    return model

# To train the model
def train_model(X, y, model):
    loss = list()
    for i in range(25):
        # Fit model for one epoch on this sequence (25 epochs)
        hist = model.fit(X, y, batch_size=1000, verbose=1, epochs=1, validation_split=.2)
        loss.append(hist.history['loss'][0])
    return loss

results = pd.DataFrame()
model_bilstm_lstm = get_bilstm_lstm_model()
plot_model(model_bilstm_lstm)
results['with_add_lstm'] = train_model(train_tokens, np.array(train_tags), model_bilstm_lstm)

## Testing the Named Entity Recognition(NER) model
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
text = nlp('My name is Amrizal Fajar. I\'m from Indonesia and i\'m still a beginner in AI Engineering. I hope i can work for Google in the future.')
displacy.render(text, style='ent', jupyter=True)
