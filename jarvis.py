import code
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from dataset import Dataset
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pickle

def classifier():
	# Figure out how to shuffle the data like in the fetch_20newsgroups
	# random_state involved too? --> default is 42, like in the tutorial.
	return Pipeline([
		('vect', CountVectorizer()),
		('tfidf', TfidfTransformer()),
		('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, shuffle=False)),
	])


def trained_model():
	train_set = Dataset(csv='jarvis/train.csv')
	
	params = {
		'vect__ngram_range': [(1, 1), (1, 2)],
		'tfidf__use_idf': (True, False),
		'clf__alpha': (1e-2, 1e-3)
	}
	
	grid_search_clf = GridSearchCV(classifier(), params, n_jobs=-1)
	model_trained = grid_search_clf.fit(train_set.data, train_set.targets)
	return [train_set, model_trained]


def clean_input(input):
	input = re.sub('[^ \w]+', '', input)
	stemmed_words = map(lambda x: Dataset.stemmer.stem(x.lower()), word_tokenize(input))
	return ' '.join(stemmed_words)


def predict_action_from_input(train_set, model, input):
	input = clean_input(input)
	target_index = model.predict([input])[0]
	return train_set.target_names[target_index]

	
# Get the train_set and trained model.
train_set, model = trained_model()

# Predict an action for Jarvis to perform based on user text input:
action = predict_action_from_input(train_set, model, "What about the weather though?")

# What's our predicted action?
print "Action: {}".format(action)

joblib.dump(model.best_estimator_, 'jarvis.pkl')

# .. later ...
saved_model = joblib.load('jarvis.pkl', 'r')
second_action = predict_action_from_input(train_set, saved_model, "Who am I again?")
print "Action: {}".format(second_action)
