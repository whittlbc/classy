import pandas
import re
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

class Dataset:
	stemmer = SnowballStemmer('english')
	
	def __init__(self, csv='', sep='|'):
		self.csv = csv
		self.sep = sep
		self.data, self.targets, self.target_names = self.process_data()
	
	
	def process_data(self):
		dataset = self.import_data_from_csv()
		inputs = dataset[:, 0:1]
		target_list = dataset[:, 1]
		cleaned_data = self.clean_data(inputs)
		targets, target_names = self.format_targets(target_list)
		return [cleaned_data, targets, target_names]
	
	
	def import_data_from_csv(self):
		return pandas.read_csv(self.csv, sep=self.sep).values
	
	
	def clean_data(self, inputs):
		cleaned_data = []
		
		for input in inputs:
			input = re.sub('[^ \w]+', '', input[0])
			stemmed_words = map(lambda x: self.stem(x), word_tokenize(input))
			cleaned_data.append(' '.join(stemmed_words))
		
		return cleaned_data
	
	
	def stem(self, string):
		return self.stemmer.stem(string.lower())
	
	
	def format_targets(self, target_list):
		target_map = {}
		targets = []
		target_names = []
		
		for target in target_list:
			if target not in target_map:
				target_map[target] = len(target_map.keys())
				target_names.append(target)
			
			targets.append(target_map[target])
		
		return targets, target_names
