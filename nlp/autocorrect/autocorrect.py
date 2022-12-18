## Autocorrect ##

## Importing the data
import pandas as pd
import re

words = []
with open('../data/autocorrect/book.txt', 'r') as f:
    file_data = f.read()
    file_data = file_data.lower()
    words = re.findall('\w+', file_data)

V = set(words)
print(f'The first ten words in the text are: \n{words[:10]}.')
print(f'There are {len(V)} unique words in vocabulary.')

## Finding the frequency of the words
from collections import Counter
word_freq_dict = Counter(words)
print(word_freq_dict.most_common()[:10])

total = sum(word_freq_dict.values())
probs = {}
for k in word_freq_dict.keys():
    probs[k] = word_freq_dict[k]/total
    
## Finding similar words
from  textdistance import Jaccard

def my_autocorrect(input_word):
    word = input_word.lower()
    if word in V:
        return input_word
    else:
        similarities = [1 - Jaccard(qval=2).distance(k, input_word) for k in word_freq_dict.keys()]
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df = df.rename(columns={'index': 'Word', 0: 'Prob'})
        df['Similarity'] = similarities
        output = df.sort_values(['Similarity', 'Prob'], ascending=False).head()
        return output

print(my_autocorrect('Machine'))    
print(my_autocorrect('Learnin'))
print(my_autocorrect('Nevertheles'))
