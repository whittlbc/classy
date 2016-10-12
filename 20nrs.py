import code
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

categories = [
	'alt.atheism',
	'soc.religion.christian',
	'comp.graphics',
	'sci.med'
]

twenty_train = fetch_20newsgroups(
	subset='train',
	categories=categories,
	shuffle=True,
	random_state=42
)

# text_clf = Pipeline([
# 	('vect', CountVectorizer()),
# 	('tfidf', TfidfTransformer()),
# 	('clf', MultinomialNB())
# ])

text_clf = Pipeline([
	('vect', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

twenty_test = fetch_20newsgroups(
	subset='test',
	categories=categories,
	shuffle=True,
	random_state=42
)

predicted = text_clf.predict(twenty_test.data)

print np.mean(predicted == twenty_test.target)

code.interact(local=dict(globals(), **locals()))
