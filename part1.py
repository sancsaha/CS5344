import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import ticker
import seaborn as sns
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

# Initialize a Spark session
spark = SparkSession.builder.appName("PandasToRDD").getOrCreate()

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('covid19_tweets.csv')



import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Concatenate all the sources into a single string
source_text = " ".join(df['source'].astype(str))

# Generate the word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(source_text)

# Display the word cloud using Matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Sources')
plt.show()

# Concatenate all the tweets into a single text
text = " ".join(tweet for tweet in df['text'])

# Define stopwords and update with custom stopwords
stopwords = set(STOPWORDS)
stopwords.update(['https', 't', 'co', 'amp', 's'])  # Add custom stopwords if needed

# Create a WordCloud object
wordcloud = WordCloud(stopwords=stopwords, background_color='white').generate(text)

# Display the word cloud
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Prevalent Words for All Tweets')
plt.show()


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Assuming you have already preprocessed the text and have it in a DataFrame called 'df'

# Create a TF-IDF vectorizer
vec = TfidfVectorizer(stop_words='english')
vec.fit(df['text'].values)
features = vec.transform(df['text'].values)

# Perform K-means clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(features)
res = kmeans.predict(features)

# Add cluster labels to the DataFrame
df['Cluster'] = res

# Create word clouds for each cluster
for cluster_id in range(2):
    text_cluster = " ".join(tweet for tweet in df[df['Cluster'] == cluster_id]['text'])
    
    stopwords = set(STOPWORDS)
    stopwords.update(['https', 't', 'co', 'amp', 's'])  # Add custom stopwords
    
    wordcloud = WordCloud(max_words=100, stopwords=stopwords, background_color='white').generate(text_cluster)
    
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(f'Word Cloud for Cluster {cluster_id}')
    plt.show()






# Calculate missing values as a percentage
missing_values = df.isnull().sum() / len(df) * 100

# Create a bar chart to visualize missing values
plt.figure(figsize=(12, 12))
sns.set_context(context='notebook', font_scale=1.5)
plt.bar(missing_values.index, missing_values.values, color='skyblue')
plt.title('Missing values in each column (%)')
plt.xlabel('Columns')
plt.ylabel('Percentage Missing')
plt.xticks(rotation=45, ha='right')
plt.show()

# Visualize missing values in each column using a heatmap
plt.figure(figsize=(10, 6))
sns.set_context(context='notebook', font_scale=1.5)
plt.title('Missing values in each column')
sns.set_context(context='notebook', font_scale=1.5)
sns.heatmap(df.isnull(), cmap='Set3', cbar=False, yticklabels=False)
plt.show()

# Visualize the number of verified and unverified users
plt.figure(figsize=(14, 6))
sns.set_context(context='notebook', font_scale=1.5)
sns.countplot(x='user_verified', data=df, palette='gist_rainbow')
plt.title('Verified Users vs. Unverified Users')
plt.xlabel('User Verified')
plt.ylabel('Count')
plt.xticks([0, 1], ['Unverified', 'Verified'])
plt.show()


# Calculate reach for each unique influencer and store it in a dictionary
influencer_reach_dict = {}
for _, row in df.iterrows():
    username = row['user_name']
    followers = row['user_followers']
    if username in influencer_reach_dict:
        influencer_reach_dict[username] += followers
    else:
        influencer_reach_dict[username] = followers

# Sort the influencers by their reach in descending order
sorted_influencers = sorted(influencer_reach_dict.items(), key=lambda x: x[1], reverse=True)

# Extract the top 10 influencers and their reach
top_10_influencers = sorted_influencers[:10]
top_influencer_names, top_influencer_reach = zip(*top_10_influencers)

# Create a bar chart
plt.figure(figsize=(12, 6))
plt.barh(top_influencer_names, top_influencer_reach, color='skyblue')
plt.xlabel('Reach')
plt.title('Top 10 Unique Influencers and Their Reach')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.gca().invert_yaxis()  # To have the highest reach at the top
plt.show()

# Group the DataFrame by user and count the number of friends for each user
user_friends_count = df.groupby('user_name')['user_friends'].sum()

# Sort the users by the number of friends in descending order and select the top 10
top_10_friendly_users = user_friends_count.sort_values(ascending=False).head(10)

# Plot the top 10 most friendly users
plt.figure(figsize=(12, 6))
top_10_friendly_users.plot(kind='bar', color='skyblue')
plt.title('Top 10 Most Friendly Users')
plt.xlabel('User')
plt.ylabel('Number of Friends')
plt.xticks(rotation=45, ha='right')
plt.show()

# Convert the 'user_created' column to datetime
df['user_created'] = pd.to_datetime(df['user_created'])

# Extract the year from the 'user_created' column
df['user_created_year'] = df['user_created'].dt.year

# Group the DataFrame by user creation year and count the number of users for each year
users_created_by_year = df['user_created_year'].value_counts().sort_index()

# Plot the number of users created year by year
plt.figure(figsize=(12, 6))
users_created_by_year.plot(kind='bar', color='skyblue')
plt.title('Users Created Year by Year')
plt.xlabel('Year')
plt.ylabel('Number of Users')
plt.xticks(rotation=45)
plt.show()

# Function to count hashtags in a tweet
def count_hashtags(tweet):
    return tweet.count('#')

# Create a new column 'hashtags_count' to store the count of hashtags in each tweet
df['hashtags_count'] = df['text'].apply(count_hashtags)

# Group the DataFrame by user and calculate the mean number of hashtags in their tweets
user_mean_hashtags = df.groupby('user_name')['hashtags_count'].mean()

# Sort the users by the mean number of hashtags in descending order and select the top 10
top_10_users_mean_hashtags = user_mean_hashtags.sort_values(ascending=False).head(10)

# Plot the top 10 users with the highest mean number of hashtags
plt.figure(figsize=(12, 6))
top_10_users_mean_hashtags.plot(kind='bar', color='skyblue')
plt.title('Top 10 Users with the Highest Mean Number of Hashtags')
plt.xlabel('User')
plt.ylabel('Mean Number of Hashtags')
plt.xticks(rotation=45, ha='right')
plt.show()





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

from nltk.corpus import stopwords

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

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

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

import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import words
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from collections import Counter
from matplotlib import pyplot as plt
from matplotlib import ticker
import seaborn as sns
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=1000)

# Fit and transform the preprocessed text data
tfidf_matrix = tfidf_vectorizer.fit_transform(text_rdd_normalized.collect())

# Initialize Latent Dirichlet Allocation (LDA) with 100 topics
num_topics = 100  # Modify the number of topics
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)

# Fit LDA to the TF-IDF matrix
lda.fit(tfidf_matrix)

# Function to print the top n keywords for each topic
def print_top_keywords_per_topic(model, vectorizer, n_words=10, n_topics=10):
    for i in range(n_topics):
        topic = model.components_[i]
        top_keywords_idx = topic.argsort()[-n_words:][::-1]
        top_keywords = [vectorizer.get_feature_names_out()[j] for j in top_keywords_idx]
        print(f"Topic {i + 1}: {', '.join(top_keywords)}")

# Print the top 10 topics and their keywords
print("Top 10 topics and their keywords:")
print_top_keywords_per_topic(lda, tfidf_vectorizer, n_words=10, n_topics=10)

# Initialize an empty list to store the relevance labels for each tweet
relevance_labels = []

# Define a threshold for topic contribution (e.g., 1%)
threshold = 0.01

# For each tweet, calculate the contribution of each topic and assign the relevance label
for i, tweet in enumerate(text_rdd_normalized.collect()):
    # Transform the tweet using the TF-IDF vectorizer
    tweet_vector = tfidf_vectorizer.transform([tweet])
    
    # Get the topic distribution for the tweet
    tweet_topic_distribution = lda.transform(tweet_vector)
    
    # Check if any topic has a contribution greater than the threshold
    relevant = any(contribution > threshold for contribution in tweet_topic_distribution[0])
    
    # Append the relevance label to the list (1 if relevant, 0 otherwise)
    relevance_labels.append(1 if relevant else 0)

# Create a table to show example tweets with their associated topic labels
example_tweets_table = pd.DataFrame(columns=["Tweet Text", "Topic Labels"])

for i, (tweet, relevance_label) in enumerate(zip(text_rdd_normalized.collect(), relevance_labels)):
    # Get the topic distribution for the relevant tweet
    tweet_vector = tfidf_vectorizer.transform([tweet])
    tweet_topic_distribution = lda.transform(tweet_vector)
    # Find the topics with contributions above the threshold
    relevant_topics = [str(i + 1) for i, contribution in enumerate(tweet_topic_distribution[0]) if contribution > threshold]
    
    example_tweets_table = example_tweets_table.append({"Tweet Text": tweet, "Topic Labels": " ".join(relevant_topics)}, ignore_index=True)

# Print the example tweets table
print("Example Tweets Table:")
print(example_tweets_table)







from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Define a function to get sentiment scores
def get_sentiment_scores(text):
    sentiment = sia.polarity_scores(text)
    return sentiment

# Apply the get_sentiment_scores function to each element in text_rdd_normalized
sentiment_scores = text_rdd_normalized.map(get_sentiment_scores)

# Now, sentiment_scores contains sentiment scores for each text
# You can take and print some sentiment scores to verify
sample_sentiment_scores = sentiment_scores.take(10)
for score in sample_sentiment_scores:
    print(score)

# You can also add the sentiment scores to your DataFrame
df['sentiment_scores'] = sentiment_scores.collect()

from textblob import TextBlob

# Define a function to get emotions
def get_emotions(text):
    analysis = TextBlob(text)
    emotion = analysis.sentiment
    return emotion

# Apply the get_emotions function to each element in text_rdd_normalized
emotion_analysis = text_rdd_normalized.map(get_emotions)

# Now, emotion_analysis contains emotion scores for each text
# You can take and print some emotion scores to verify
sample_emotion_scores = emotion_analysis.take(10)
for emotion in sample_emotion_scores:
    print(emotion)

# You can also add the emotion scores to your DataFrame
df['emotion_scores'] = emotion_analysis.collect()


# Assign the preprocessed text back to the 'text' column in the original DataFrame
df['text'] = text_rdd_normalized.collect()


# Add sentiment labels to the DataFrame
def get_sentiment_label(compound_score):
    if compound_score >= 0.1:
        return "positive"
    elif compound_score <= -0.1:
        return "negative"
    else:
        return "neutral"

df['sentiment_label'] = sentiment_scores.map(lambda x: get_sentiment_label(x['compound'])).collect()




from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment_label'], test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report







# Create a text classification pipeline
text_clf_mnb = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB())
])

# Train the Random Forest model
text_clf_mnb.fit(X_train, y_train)

# Make predictions on the test data
y_pred_mnb = text_clf_mnb.predict(X_test)

# Evaluate the Random Forest model
accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
report_mnb = classification_report(y_test, y_pred_mnb)

print("MNB Model Accuracy:", accuracy_mnb)
print("MNB Model Classification Report:\n", report_mnb)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred_mnb, labels=["negative", "neutral", "positive"])

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["negative", "neutral", "positive"], yticklabels=["negative", "neutral", "positive"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Create a bar plot to visualize the distribution of predicted sentiment labels
plt.figure(figsize=(6, 4))
sns.countplot(x=y_pred_mnb, palette="Set3")
plt.xlabel('Predicted Sentiment')
plt.ylabel('Count')
plt.title('Distribution of Predicted Sentiment Labels')
plt.show()



import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

# Assuming you have already performed the classification and have the classification report
report = classification_report(y_test, y_pred_mnb, output_dict=True)

# Extract the metrics from the classification report
precision = [report[label]['precision'] for label in report if label != 'accuracy']
recall = [report[label]['recall'] for label in report if label != 'accuracy']
f1_score = [report[label]['f1-score'] for label in report if label != 'accuracy']
labels = [label for label in report if label != 'accuracy']

# Create a DataFrame to store the metrics
metrics_df = pd.DataFrame({'Precision': precision, 'Recall': recall, 'F1-Score': f1_score}, index=labels)

# Plot the line chart
plt.figure(figsize=(10, 6))
for column in metrics_df.columns:
    plt.plot(metrics_df.index, metrics_df[column], marker='o', label=column)

plt.title("Classification Metrics")
plt.xlabel("Sentiment Labels")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

plt.show()









from sklearn.ensemble import RandomForestClassifier

# Create a text classification pipeline with a Random Forest classifier
text_clf_rf = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', RandomForestClassifier(n_estimators=100))  # You can adjust the number of trees (n_estimators)
])

# Train the Random Forest model
text_clf_rf.fit(X_train, y_train)

# Make predictions on the test data
y_pred_rf = text_clf_rf.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

print("Random Forest Model Accuracy:", accuracy_rf)
print("Random Forest Model Classification Report:\n", report_rf)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred_rf, labels=["negative", "neutral", "positive"])

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["negative", "neutral", "positive"], yticklabels=["negative", "neutral", "positive"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Create a bar plot to visualize the distribution of predicted sentiment labels
plt.figure(figsize=(6, 4))
sns.countplot(x=y_pred_rf, palette="Set3")
plt.xlabel('Predicted Sentiment')
plt.ylabel('Count')
plt.title('Distribution of Predicted Sentiment Labels')
plt.show()



import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

# Assuming you have already performed the classification and have the classification report
report = classification_report(y_test, y_pred_rf, output_dict=True)

# Extract the metrics from the classification report
precision = [report[label]['precision'] for label in report if label != 'accuracy']
recall = [report[label]['recall'] for label in report if label != 'accuracy']
f1_score = [report[label]['f1-score'] for label in report if label != 'accuracy']
labels = [label for label in report if label != 'accuracy']

# Create a DataFrame to store the metrics
metrics_df = pd.DataFrame({'Precision': precision, 'Recall': recall, 'F1-Score': f1_score}, index=labels)

# Plot the line chart
plt.figure(figsize=(10, 6))
for column in metrics_df.columns:
    plt.plot(metrics_df.index, metrics_df[column], marker='o', label=column)

plt.title("Classification Metrics")
plt.xlabel("Sentiment Labels")
plt.ylabel("Score")
plt.legend()
plt.grid(True)

plt.show()



spark.stop()


