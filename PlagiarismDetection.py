"""
Part 1 - findPlagiarism
This part aims to identify if the target string was written by someone who plagiarized one of the sentences in the list.
The function uses the cosine similarity to calculate the similarity between the target and each sentence in the list.
The function returns the index of the sentence with the highest similarity, which is the sentence that was plagiarized.
Target might not have the same number of words as any of the sentences in the list, so normaization is used to ensure that the similarity is calculated correctly.
"""

# import all necessary libraries
import re
import gensim.models.keyedvectors as word2vec
from scipy.spatial.distance import cosine as cosDist
import numpy as np

# load pre-trained word2vec model
model = word2vec.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
classifier = None # initalizing classifier

# gloabl function to set model
def setModel(Model):
	global model
	model = Model

# function to convert sentence to vector
def sen_to_vec(sentence, model):
	words = re.findall(r'\w+', sentence.lower()) # extract words from sentence and convert to lower case
	vectors = [model[word] for word in words if word in model] # get word vectors for words in the model
	if len(vectors) == 0: # if no vectors found, return zero vector
		return np.zeros(300)

	sentence_vector = np.mean(vectors, axis=0) # mean vector

	norm = np.linalg.norm(sentence_vector) # vector norm to normalize
	return sentence_vector / norm if norm else sentence_vector # return normalized vector

# function to find plagiarism
def findPlagiarism(sentences, target):
	t_vector = sen_to_vec(target, model) # calculate target vector
	similarities = [] # list to store similarities
	for sentence in sentences: # calculate similarity for each sentence
		s_vector = sen_to_vec(sentence, model)
		similarities.append(1 - cosDist(t_vector, s_vector)) # cosine similarity

	return np.argmax(similarities) # return index of highest similarity

