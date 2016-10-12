import code
import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from dataset import Dataset
from nltk.tokenize import word_tokenize


def pipeline():
	# Figure out how to shuffle the data like in the fetch_20newsgroups
	# random_state involved too? --> default is 42, like in the tutorial.
	return Pipeline([
		('vect', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, shuffle=False)),
	])


def trained_model():
	train_set = Dataset(csv='jarvis/train.csv')
	trained_pipeline = pipeline().fit(train_set.data, train_set.targets)
	return [train_set, trained_pipeline]


def clean_input(input):
	input = re.sub('[^ \w]+', '', input)
	stem_and_lower = lambda x: Dataset.stemmer.stem(x.lower())
	stemmed_words = map(stem_and_lower, word_tokenize(input))
	return ' '.join(stemmed_words)


def predict_action_from_input(train_set, model, input):
	input = clean_input(input)
	target_index = model.predict([input])[0]
	return train_set.target_names[target_index]

	
# Get the train_set and trained model.
train_set, model = trained_model()

# Predict an action for Jarvis to perform based on user text input:
action = predict_action_from_input(train_set, model, "What's the weather look like?")

# What's our predicted action?
print "Action: {}".format(action)


# code.interact(local=dict(globals(), **locals()))
