"""
Part 2 - calcSentiment
This part aims to predict the sentiment of a movie review. 
The train function uses a CountVectorizer to create a table of word counts for each review.
The MultinomialNB model is then fitted to the reviews and labels.
The test function lemmatizes the words in the review, transforms the review into a vector, and predicts the sentiment of the review.
The function returns the sentiment of the review: true for positive, false for negative.

Lemmatizer was chosen over Stemmer because it produces more readable results, and avoids stemming words to the point where they no longer hold meaning.

"""

import nltk
import json
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt_tab') # download the punkt tokenizer
nltk.download('wordnet') # download the WordNet lemmatizer 


model = None # global variable to hold the trained model
vectorizer = None # global variable to hold the vectorizer

def calcSentiment_train(trainFile):
	
	global model, vectorizer
	lemmatizer = WordNetLemmatizer() # creates a WordNetLemmatizer object
	vectorizer = CountVectorizer() # creates a CountVectorizer object

	reviews = [] # list to store the reviews
	labels = [] # list to store the labels (sentiment) of the reviews

	with open(trainFile, 'r', encoding='utf-8') as f: # opens the train file and reads the data
		for line in f:
			data = json.loads(line) # loads the data from the json file
			review = data['review']
			label = data['sentiment']

			processed_review = ' '.join([lemmatizer.lemmatize(word) for word in review.split()]) # lemmatizes the words in the review
			reviews.append(processed_review)
			labels.append(label)

	X = vectorizer.fit_transform(reviews) # creates a table of word counts for each review
	y = labels # assigns the labels to y

	model = MultinomialNB() # creates a MultinomialNB object
	model.fit(X, y) # fits the MultinomialNB object to the reviews and labels

	pass 

def calcSentiment_test(review):
	global model, vectorizer

	lemmatizer = WordNetLemmatizer() # creates a WordNetLemmatizer object
	
	processed_review = ' '.join([lemmatizer.lemmatize(word) for word in review.split()]) # lemmatizes the words in the review
	vectorized_review = vectorizer.transform([processed_review]) # transforms the review into a vector
	prediction = model.predict(vectorized_review) # predicts the sentiment of the review

	return bool(prediction[0]) # returns the sentiment of the review: true for positive, false for negative
