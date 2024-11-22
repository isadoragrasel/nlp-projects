"""
Part 2 - classifySubreddit
This part aims to classify the subreddit of a given comment. 
The function uses a logistic regression classifier to predict the subreddit based on the comment text.
The classifier is trained on a dataset of comments and their corresponding subreddits.
The function returns the predicted subreddit for the given comment.
"""

# import all necessary libraries
import re
import gensim.models.keyedvectors as word2vec
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cosine as cosDist
import numpy as np
import json

# load pre-trained word2vec model
model = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
classifier = None # initalizing classifier

# convert comment to vector
def com_to_vec(comment, model):
	words = re.findall(r'\w+', comment.lower())
	vectors = [model[word] for word in words if word in model]
	if len(vectors) == 0:
		return np.zeros(300)
	
	return np.mean(vectors, axis=0)

# function to train classifier
def classifySubreddit_train(trainFile):
	global classifier
	comments = [] # list to store comment vectors
	labels = [] # list to store labels

	with open(trainFile, 'r', encoding='utf-8') as f: # open file
		for line in f: # read each line
			data = json.loads(line) # load json data
			comment = data['body'] # get comment body
			subreddit = data['subreddit'] # get subreddit
			c_vector = com_to_vec(comment, model) # convert comment to vector
			comments.append(c_vector) # add comment vector to list
			labels.append(subreddit) # add subreddit to list

	X = np.array(comments) # convert comments to numpy array
	y = np.array(labels) # convert labels to numpy array
	classifier = LogisticRegression(max_iter=1000) # initialize logistic regression model
	classifier.fit(X, y) # fit model to data

	pass


def classifySubreddit_test(comment):
	c_vector = com_to_vec(comment, model)
	return classifier.predict([c_vector])[0]