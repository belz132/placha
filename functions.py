import re, pickle
import pandas as pd
import numpy as np

from datetime import datetime as dt

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from flask import Flask, jsonify, request
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from

from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


factory = StemmerFactory()
stemmer = factory.create_stemmer()

alay_dict = pd.read_csv('new_kamusalay.csv', encoding='latin-1', header=None)
alay_dict = alay_dict.rename(columns={0: 'original', 1: 'replacement'})
alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))

id_stopword_dict = pd.read_csv('stopwordbahasa.csv', header=None)
id_stopword_dict = id_stopword_dict.rename(columns={0: 'stopword'})

# max_features = 100000
# tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

sentiment = ['negative', 'neutral', 'positive']

my_feature = open("nn_aug_tfidf.feature", 'rb')
nn_feature_file = pickle.load(my_feature)
my_feature.close()

my_model = open("nn_aug_tfidf.model", 'rb')
nn_model_file = pickle.load(my_model)
my_model.close()

my_tokenizer = open("tokenizer.pickle", 'rb')
tokenizer_file_from_lstm = pickle.load(my_tokenizer)
my_tokenizer.close()

file = open("x_pad_sequences.pickle", 'rb')
feature_file_from_lstm = pickle.load(file)
file.close()

model_file_from_lstm = load_model("model.h5")

def simpler_cleansing(text):

	# Lower Case Operation
	text = text.lower()

	# Removing Non-Alphanumeric Characters
	text = re.sub('[^a-zA-Z0-9]+', ' ', text)

	return text

def cleansing(text):

	# Lower Case Operation
	text = text.lower()

	# Removing Unnecessary Characters
	text = re.sub('\n', ' ', text)
	text = re.sub('rt', ' ', text)
	text = re.sub('user', ' ', text)
	text = re.sub(r'((www\.[^\s]+)|(http?://[^\s]+)|(https?://[^\s]+))', ' ', text)
	text = re.sub(' +', ' ', text)

	# Removing Non-Alphanumeric Characters
	text = re.sub('[^a-zA-Z0-9]+', ' ', text)

	# Normalizing Alay Words
	text = ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])

	# Removing Stopword
	text = ' '.join(['' if word in id_stopword_dict.stopword.values else word for word in text.split(' ')])
	text = text.strip()

	# Stemming
	text = stemmer.stem(text)

	return text

def nn_text_process(original_text):

	feature = nn_feature_file.transform([cleansing(original_text)])

	get_sentiment = nn_model_file.predict(feature)[0]

	json_response = {
		'status_code': 200,
		'description': "Hasil Prediksi Sentiment Analysis Dengan Neural Network (MLPClassifier)",
		'data': {
			'text': original_text,
			'sentiment': get_sentiment
		},
	}

	return json_response

def nn_file_process(file):

	file_df = pd.read_csv(file, encoding='ISO-8859-1', on_bad_lines='skip')

	cleaned_text = []
	sentiment_analysis = []
	response_data =  []

	for ind in file_df.index:

		text = file_df['Tweet'][ind]
		original_text = text

		text = [cleansing(original_text)]

		feature = nn_feature_file.transform(text)

		get_sentiment = nn_model_file.predict(feature)[0]

		json_response = {
			'status_code': 200,
			'description': "Hasil Prediksi Sentiment Analysis Dengan Neural Network (MLPClassifier)",
			'data': {
				'text': original_text,
				'sentiment': get_sentiment
			},
		}

		sentiment_analysis.append(get_sentiment)
		cleaned_text.append(text)
		response_data.append(json_response)		

	file_df['Cleaned_Text'] = cleaned_text
	file_df['Sentiment'] = sentiment_analysis

	curr_time = dt.now()
	d = curr_time.day
	mo = curr_time.month
	y = curr_time.year
	h = curr_time.hour
	mi = curr_time.minute
	s = curr_time.second

	new_df = file_df[['Tweet', 'Cleaned_Text', 'Sentiment']]
	new_df.to_csv(f"{d}_{mo}_{y}_{h}_{mi}_{s}.tsv", sep='\t', index=False)

	return response_data

def lstm_text_process(original_text):

	text = [cleansing(original_text)]

	sequence = tokenizer_file_from_lstm.texts_to_sequences(text)
	feature = pad_sequences(sequence, maxlen=feature_file_from_lstm.shape[1])

	prediction = model_file_from_lstm.predict(feature)
	get_sentiment = sentiment[np.argmax(prediction[0])]

	json_response = {
		'status_code': 200,
		'description': "Hasil Prediksi Sentiment Analysis Dengan LSTM",
		'data': {
			'text': original_text,
			'sentiment': get_sentiment
		},
	}

	return json_response

def lstm_file_process(file):

	file_df = pd.read_csv(file, encoding='ISO-8859-1', on_bad_lines='skip')

	cleaned_text = []
	sentiment_analysis = []
	response_data =  []

	for ind in file_df.index:

		text = file_df['Tweet'][ind]
		original_text = text

		text = [cleansing(original_text)]

		sequence = tokenizer_file_from_lstm.texts_to_sequences(text)
		feature = pad_sequences(sequence, maxlen=feature_file_from_lstm.shape[1])

		prediction = model_file_from_lstm.predict(feature)
		get_sentiment = sentiment[np.argmax(prediction[0])]

		json_response = {
			'status_code': 200,
			'description': "Hasil Prediksi Sentiment Analysis Dengan LSTM",
			'data': {
				'text': original_text,
				'sentiment': get_sentiment
			},
		}

		sentiment_analysis.append(get_sentiment)
		cleaned_text.append(text)
		response_data.append(json_response)		

	file_df['Cleaned_Text'] = cleaned_text
	file_df['Sentiment'] = sentiment_analysis

	curr_time = dt.now()
	d = curr_time.day
	mo = curr_time.month
	y = curr_time.year
	h = curr_time.hour
	mi = curr_time.minute
	s = curr_time.second

	new_df = file_df[['Tweet', 'Cleaned_Text', 'Sentiment']]
	new_df.to_csv(f"{d}_{mo}_{y}_{h}_{mi}_{s}.tsv", sep='\t', index=False)

	return response_data
