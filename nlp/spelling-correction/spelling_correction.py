## Spelling Correction ##

from textblob import TextBlob

sentence = 'I want to be a Machene Learnin Enginer'
words = sentence.split(' ')

corrected_words = []
for word in words:
    corrected_words.append(TextBlob(word))

for i in range(len(words)):
    print(words[i] + ' -> ' + str(corrected_words[i].correct()))
