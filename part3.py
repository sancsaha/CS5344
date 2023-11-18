import re
from collections import Counter
#import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
#from matplotlib import ticker
#import seaborn as sns
#import plotly.express as px
from pyspark.sql import SparkSession
#from pyspark.sql.types import StructType, StructField, StringType
import nltk
from geopy import geocoders
#from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#from nltk.sentiment.util import *
from textblob import TextBlob
from nrclex import NRCLex

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

#### PART 1 DATA PREPROCESSING ####

# Initialize a Spark session
spark = SparkSession.builder.appName("PandasToRDD").getOrCreate()

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('covid19_tweets.csv', parse_dates=['date'])

# Convert the Pandas DataFrame to a PySpark RDD
rdd = spark.sparkContext.parallelize(df.values.tolist())

# To view the first few rows, you can take and print them
sample_rows = rdd.take(10)
for row in sample_rows:
    print(row)
    
# Apply text preprocessing functions to the 'text' column of rows_rdd
text_rdd = rdd.map(lambda row: row[9])  # Select the 'text' column

# Apply the remove_url function to each element in text_rdd
text_rdd_cleaned = text_rdd.map(lambda text: re.sub(r'https\S+', '', text))

# Define a lambda function to convert text to lowercase
to_lower = lambda text: text.lower()

# Apply the to_lower function to each element in text_rdd_cleaned
text_rdd_lowercase = text_rdd_cleaned.map(to_lower)

# Define a function to remove emojis from text
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Apply the remove_emoji function to each element in text_rdd_lowercase
text_rdd_no_emoji = text_rdd_lowercase.map(remove_emoji)

# Now, text_rdd_no_emoji contains the text with emojis removed
# You can take and print some text without emojis to verify
sample_text_without_emoji = text_rdd_no_emoji.take(20)
for text in sample_text_without_emoji:
    print(text)
    
# Define a function to remove text in square brackets
def remove_square_brackets(text):
    return re.sub(r'\[.*?\]', '', text)

# Define a function to remove punctuation
def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# Define a function to remove words containing numbers
def remove_words_with_numbers(text):
    return ' '.join(word for word in text.split() if not any(c.isdigit() for c in word))

# Apply the remove_square_brackets function to each element in text_rdd_no_emoji
text_rdd_no_brackets = text_rdd_no_emoji.map(remove_square_brackets)

# Apply the remove_punctuation function to each element in text_rdd_no_brackets
text_rdd_no_punctuation = text_rdd_no_brackets.map(remove_punctuation)

# Apply the remove_words_with_numbers function to each element in text_rdd_no_punctuation
text_rdd_no_numbers = text_rdd_no_punctuation.map(remove_words_with_numbers)

# Define a set of stopwords
stop_words = set(stopwords.words('english'))

# Define additional custom words to be added to the stopwords list
more_words = ['#coronavirus', '#coronavirusoutbreak', '#coronaviruspandemic', '#covid19', '#covid_19', '#epitwitter', '#ihavecorona', 'amp', 'coronavirus', 'covid19', 'covidpositive', 'coronavirusupdates']

# Update the stopwords set with custom words
stop_words.update(more_words)

# Define a function to remove stopwords and custom words
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Apply the remove_stopwords function to each element in text_rdd_no_numbers
text_rdd_no_stopwords = text_rdd_no_numbers.map(remove_stopwords)

# Initialize WordNet lemmatizer and Porter stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Define a function to perform text normalization (lemmatization and stemming)
def normalize_text(text):
    words = text.split()
    normalized_words = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words]
    return ' '.join(normalized_words)

# Apply the normalize_text function to each element in text_rdd_no_stopwords
text_rdd_normalized = text_rdd_no_stopwords.map(normalize_text)

# Now, text_rdd_normalized contains the text after normalization
# You can take and print some normalized text to verify
sample_normalized_text = text_rdd_normalized.take(10)
for text in sample_normalized_text:
    print(text)

# Assign the preprocessed text back to the 'text' column in the original DataFrame
df['text'] = text_rdd_normalized.collect()

# Convert the updated DataFrame back to a PySpark RDD
#rdd = spark.sparkContext.parallelize(df.values.tolist())

# To view the first few rows of the updated RDD, you can take and print them
#sample_rows_updated = rdd.take(10)
#for row in sample_rows_updated:
#    print(row)

#### PART 2 USER GEOLOCATION ####

def nominatim(loc):
	g = geocoders.Nominatim(user_agent='geopy')
	ret = g.geocode(loc, language='en', addressdetails=True)
	if ret:
		ret = ret.raw['address'].get('country', None)
	return ret

def photon(loc,):
	g = geocoders.Photon()
	ret = g.geocode(loc, language='en')
	if ret:
		ret = ret.raw['properties'].get('country', None)
	return ret

def arcgis(loc):
	g = geocoders.ArcGIS()
	ret = g.geocode(loc, out_fields="CntryName", langCode="EN")
	if ret:
		ret = ret.raw['attributes'].get('CntryName', None)
	return ret

def get_country(location):
	if location != location:
		return location
	location = re.sub(r"\s\s+" ," ", location.strip())
	clean = ' '.join(re.findall(r'[^ -`{-~]+', location.lower()))
	clean = re.sub(r'\b([a-z]) (?=[a-z]\b)', r'\1', clean)
	if ('usa' in clean) or ('united states' in clean):
		ret = 'United States'
	elif 'canada' in clean:
		ret = 'Canada'
	elif 'india' in clean:
		ret = 'India'
	else:
		cnt = Counter()
		for geo in (nominatim, photon, arcgis):
			try:
				ret = geo(location)
				if ret:
					cnt[ret] += 1
			except:
				pass

		cnt = cnt.most_common(2)
		if (not cnt) or (len(cnt) > 1 and cnt[0][1] == cnt[1][1]):
			return None
		ret = cnt[0][0]

	return ret

user_country = rdd.map(lambda row: get_country(row[1]))
df['user_country'] = user_country.collect()

#### PART 3 SENTIMENT ANALYSIS ####

# Define a function to get sentiment scores
def get_sentiment_scores(text, sia=SentimentIntensityAnalyzer()):
    sentiment = TextBlob(text).sentiment
    return {'polarity':sentiment.polarity, 'subjectivity':sentiment.subjectivity, **sia.polarity_scores(text)}

# Apply the get_sentiment_scores function to each element in text_rdd_normalized
sentiment_scores = text_rdd_normalized.map(get_sentiment_scores)

# Now, sentiment_scores contains sentiment scores for each text
# You can take and print some sentiment scores to verify
sample_sentiment_scores = sentiment_scores.take(10)
for score in sample_sentiment_scores:
    print(score)

# You can also add the sentiment scores to your DataFrame
sentiment_scores = sentiment_scores.collect()
sentiment_scores = pd.json_normalize(sentiment_scores).fillna(0).add_prefix('sentiment_')

# Define a function to get emotions
def get_emotions(text):
	return NRCLex(text).raw_emotion_scores

# Apply the get_emotions function to each element in text_rdd_normalized
emotion_scores = text_rdd_normalized.map(get_emotions)

# Now, emotion_analysis contains emotion scores for each text
# You can take and print some emotion scores to verify
sample_emotion_scores = emotion_scores.take(10)
for emotion in sample_emotion_scores:
    print(emotion)

# You can also add the emotion scores to your DataFrame
emotion_scores = emotion_scores.collect()
emotion_scores = pd.json_normalize(emotion_scores).fillna(0).add_prefix('emotion_')

df = pd.concat([df, sentiment_scores, emotion_scores], axis=1)
df.to_pickle('df.pkl')

# Stop the SparkContext when done
spark.stop()
