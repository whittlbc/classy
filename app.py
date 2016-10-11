import numpy as np
import pandas
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

seed = 7
np.random.seed(seed)

# We currently have our data in a csv - iris.csv
# We need to import that and convert it into a usable array.
# Let's first convert the csv data into pandas' core DataFrame object.
dataframe = pandas.read_csv('iris.csv', header=None)

# Now lets convert our DataFrame object into an array we can use as our dataset.
dataset = dataframe.values  # => array(150x5)

# Let's separate out the input and output column data into different variables (X and Y, respectively).
# First let's find out how many input columns we have though, based on our known number of ouput columns.
output_cols = 1
input_cols = len(dataframe.columns) - output_cols

X = dataset[:, 0:input_cols].astype(float)  # our input values are floats
Y = dataset[:, input_cols]  # our output values are strings

# When modeling multi-class classification problems using neural networks, it's good practice
# to reshape the output attribute from a vector that contains values for each class value to be a
# matrix with a boolean for each class value and whether or not a given instance has that class value or not.
#
# This is called one hot encoding or creating dummy variables from a categorical variable.
#
# Ex:
#
# Before:
# [
# 	Iris-setosa,
# 	Iris-versicolor,
# 	Iris-virginica
# ]
#
# After:
# [
# 	[     1,		          0,			          0     ],
# 	[     0,		          1, 			          0     ],
# 	[     0, 		          0, 			          1     ]
# ]
#
# We can do this by first encoding the strings consistently to integers using the scikit-learn class,
# LabelEncoder. Then convert the vector of integers to a one hot encoding using the Keras function to_categorical().
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)  # Strings are now ints (0,1,2..etc)
dummy_y = np_utils.to_categorical(encoded_Y)  # convert integers to dummy variables (i.e. one hot encoded)

# Time to define our Neural Network model:

# The network topology of this simple one-layer neural network can be summarized as:
# 4 inputs -> [4 hidden nodes] -> 3 outputs

# Store these for further use
input_dim = X.shape[1]
output_dim = dummy_y.shape[1]
hidden_dim = input_dim  # doesn't have to be

# Let's build up our sequential model and fit it.
model = Sequential()
model.add(Dense(hidden_dim, input_dim=hidden_dim, init='normal', activation='relu'))
model.add(Dense(output_dim, init='normal', activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, dummy_y, nb_epoch=100, batch_size=5, verbose=1)

# Evaluate the model to train it, then print out the results.
score = model.evaluate(X, dummy_y, verbose=1)
print("\n\n%s: %.2f%%\n" % (model.metrics_names[1], score[1] * 100))

# Save our model to disk for future predictions:
# 1. Convert model to JSON
# 2. Write JSON to file
# 3. Serialize model weights to HDF5
model_json = model.to_json()

with open("model.json", "w") as json_file:
	json_file.write(model_json)

model.save_weights("model.h5")

# ...later...

# Read the model from disk so that we can use it to make a prediction:
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load the saved weights into our new model:
loaded_model.load_weights("model.h5")

# Evaluate the loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X, dummy_y, verbose=1)

# Our results from this prediction should have the same accuracy as
# before our model/weights was originally saved to disk.
print("\n\n%s: %.2f%%\n" % (loaded_model.metrics_names[1], score[1] * 100))
