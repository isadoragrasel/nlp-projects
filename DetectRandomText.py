"""
Part 1 - calcNGrams
This part aims to identify if a body if text is actually human written or just a random combination of words.
The train function uses a CountVectorizer to calculate the frequency of trigrams in the training data.
The test function uses the trigram frequency to calculate the score of each sentence in the test data.
The score is calculated by summing the frequency of each trigram in the sentence.
The function returns the index of the sentence with the lowest score, which is the least human-like sentence.

"""

import nltk
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt_tab') # download the punkt tokenizer
nltk.download('wordnet') # download the WordNet lemmatizer 


def calcNGrams_train(trainFile):
	with open(trainFile, 'r', encoding='utf-8') as f: # opens the train file and reads the data
		trainData = f.readlines()

	global trigramFreq # makes the trigramFreq variable global so it can be used in the test function
	trigramFreq = CountVectorizer(ngram_range=(3, 3)) # creates a CountVectorizer object to calculate the frequency of trigrams
	trigramFreq.fit(trainData) # fits the CountVectorizer object to the training data

	pass # this function does not return anything


def calcNGrams_test(sentences):
	sentScores = [] # list to store the scores of each sentence
	sentFeatures = trigramFreq.transform(sentences) # calculates the frequency of trigrams in the test data

	for i in range(sentFeatures.shape[0]): 
		score = sentFeatures[i, :].sum() # calculates the score of each sentence by summing the frequency of each trigram
		sentScores.append(score) # appends the score to the list

	randomIndex = sentScores.index(min(sentScores)) # returns the index of the sentence with the lowest score, aka the least human-like sentence
	return randomIndex


