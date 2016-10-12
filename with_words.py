import code
import numpy as np
import pandas
import re
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

stemmer = SnowballStemmer('english')
stops = set(stopwords.words('english'))


def tokenize(str):
	return word_tokenize(str)


def remove_stopwords(words):
	return [w for w in words if not w in stops]


def stem(words):
	return map(lambda x: stemmer.stem(x), words)
	

def clean_seq(seq):
	format_ops = [
		tokenize,
		remove_stopwords,
		stem
	]
	
	for op in format_ops:
		seq = op(seq)
		
	return ' '.join(seq)


def create_word_embeddings(seqs):
	cleaned_seqs = []
	
	for seq in seqs:
		cleaned_seqs.append(clean_seq(seq[0]))
	
	vectorizer = CountVectorizer(analyzer='word', max_features=5000)
	
	return [vectorizer.fit_transform(cleaned_seqs).toarray(), vectorizer]


def create_model(x, y):
	model = Sequential()
	
	layers = (
		Embedding(10, 32, input_length=35),
		LSTM(100),
		Dense(1, init='normal', activation='sigmoid')
	)
	
	for layer in layers:
		model.add(layer)
	
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(x, y, nb_epoch=100, batch_size=5, verbose=1)
	
	return model


seed = 7
np.random.seed(seed)

# Import dataset
dataframe = pandas.read_csv('jarvis.csv', sep='|')
dataset = dataframe.values

# Input:
string_inputs = dataset[:, 0:1]
X, vectorizer = create_word_embeddings(string_inputs)

# Output:
Y = dataset[:, 1]
# Let's one-hot encode our categories (Y) into dummy vars
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)  # Strings are now ints (0,1,2..etc)
dummy_y = np_utils.to_categorical(encoded_Y)  # convert integers to dummy variables (i.e. one hot encoded)

# Create model:
model = create_model(X, dummy_y)

# Evaluate model for first time:
training_score = model.evaluate(X, dummy_y, verbose=1)
print("\n\n%s: %.2f%%\n" % (model.metrics_names[1], training_score[1] * 100))

# Save model to json file
with open('jarvis.json', 'w') as f:
	f.write(model.to_json())

# Save weights to h5 file
model.save_weights('jarvis.h5')

# Read the model json file from disk to make another prediction with it:
json_model_file = open('jarvis.json', 'r')
loaded_model_json = json_model_file.read()
json_model_file.close()

# Create model from json
loaded_model = model_from_json(loaded_model_json)

# Load the saved weights into our model
loaded_model.load_weights('jarvis.h5')

# Evaluate the loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
test_score = loaded_model.evaluate(X, dummy_y, verbose=1)

# Our results from this prediction should have the same accuracy as
# before our model/weights was originally saved to disk.
print("\n\n%s: %.2f%%\n" % (loaded_model.metrics_names[1], test_score[1] * 100))
