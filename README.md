# Analyzing the Impact of COVID-19 Using Twitter Data
This project was collaboratively undertaken by the following team members (Group 7):

- Sanchari Saha (A0281052N)
- Tan Yi Kai (A0121571R)
- Le Goff Loick (A0281044M)
- Anocha Sutaveephamochanon (A0268230J)
  
Their combined efforts and expertise contributed to the successful execution and completion of the project.

## Project Overview
The COVID-19 pandemic has triggered a surge in global discourse on Twitter, making it a dynamic platform for discussions and insights into various aspects of the crisis. In this project, we delve into the virtual realm of Twitter to dissect the discourse surrounding COVID-19. Our exploration encompasses a wide range of tasks aimed at unraveling topics, sentiments, and user trends during the specified timeframe.

### Tasks Performed

- Data Quality Assessment
- Exploratory Data Analysis
- Data Preprocessing
- Topic Modeling
- Sentiment and Emotion Analysis
- Word Cloud Analysis
- Predictive ML models in Sentiment Analysis
- Update Classification of words

## Dataset
The dataset utilized in this project is centered around English tweets related to the COVID-19 pandemic, specifically those incorporating the widely used #covid19 hashtag. Acquired through the Twitter API and a Python script, the dataset spans the period from July 25, 2020, to August 30, 2020. Daily queries were executed to capture tweets featuring high-frequency hashtags, resulting in a dataset that serves as the foundational source for comprehensive analysis.

The dataset, licensed under CC0: Public Domain, is accessible on [Kaggle](https://www.kaggle.com/datasets/gpreda/covid19-tweets/data) and comprises a total of 170,000 tweets, with each tweet corresponding to a distinct row in the dataset. Key columns include Timestamp, representing the date and time of each tweet, Tweet Text, Hashtags, and User Information, encompassing details such as user name, follower count, and more.

## Dependencies

The project utilizes the following Python libraries and tools:

- **Pandas:** A data manipulation and analysis library.
- **NumPy:** A library for numerical operations in Python.
- **NLTK (Natural Language Toolkit):** A library for natural language processing tasks.
- **Matplotlib:** A plotting library for creating static, interactive, and animated visualizations.
- **Seaborn:** A statistical data visualization library.
- **Plotly Express:** A library for creating interactive plots and dashboards.
- **Spark (PySpark):** Apache Spark's Python API for distributed data processing.
- **Scikit-learn:** A machine learning library for various tasks, including text classification.
- **WordCloud:** A library for creating word clouds from text data.
- **TextBlob:** A library for processing textual data, including sentiment analysis.
- **TfidfVectorizer:** A scikit-learn library for converting text data to a TF-IDF matrix.

Please make sure to install these dependencies before running the script.



## To run the code

We have 4 different scripts, which perform different tasks and subtasks.

### part1.py: 
This Python script performs sentiment analysis and sentiment classification on a dataset of tweets related to COVID-19. The script utilizes various natural language processing (NLP) techniques, machine learning models, and data visualization tools. Here's an overview of the main functionalities:

#### 1. Data Preprocessing

- The script begins by loading the COVID-19 tweets dataset from a CSV file using the Pandas library.
- Text preprocessing techniques are applied to clean and normalize the tweet text. This includes removing URLs, emojis, square brackets, punctuation, and stopwords. Additionally, text normalization through lemmatization and stemming is performed.

#### 2. Exploratory Data Analysis (EDA)

- Exploratory data analysis is conducted to visualize missing values in the dataset using bar charts and heatmaps.
- Several visualizations are created to analyze user-related information, such as user verification status, top influencers, user creation trends, and more.

#### 3. Topic Modeling with Latent Dirichlet Allocation (LDA)

- The script uses a TF-IDF vectorizer and Latent Dirichlet Allocation (LDA) to identify key topics within the tweet data. The top keywords for each topic are printed, providing insights into prevalent themes in the dataset.

#### 4. Sentiment Analysis

- Sentiment analysis is performed using the NLTK library's VADER sentiment intensity analyzer and TextBlob. Sentiment scores are calculated for each tweet, including compound scores.
- The script adds sentiment labels (positive, negative, neutral) to the dataset based on the compound sentiment scores.

#### 5. Sentiment Classification Models

- Two text classification models, Multinomial Naive Bayes, and Random Forest, are implemented using the scikit-learn library.
- The script splits the dataset into training and testing sets, trains the models, and evaluates their performance using accuracy metrics and confusion matrices.
- Classification reports and visualizations, including confusion matrices and bar plots, provide a comprehensive view of model performance.

### part2.py:
#### 1. Exploratory Data Analysis (EDA) on date-time and location fields

- The script uses pyspark mapReduce to extract date-time related fields.
- The script uses pyspark mapReduce, geopy and googlemap API to extract location related fields.

#### 2. Sentiment polarity Analysis

- The script uses Text-blob to extract text polarity.
- The script uses mapReduce and pivot with country and continent to get the statistic.
  
### part2.ipynb:
Plot the date-time and location related analysis results in part2.ipynb

### part3.py:
- Part 3  consists of sentiment analysis and emotion analysis.

#### 1. Sentiment Analysis
- The script uses NLTK SentimentIntensityAnalyzer and Text Blob to obtain sentiment polarity and subjectivity

#### 2. Emotion Analysis
- The script uses NRC Lex to obtain various emotion scores associated with each tweet.

#### 3. Visualization
- Various charts are added in the Python notebook to aid in visualization.
- These include trend charts for sentiment and emotion scores, broken down by country.
- Word clouds show the keywords that are associated with each sentiment polarity and emotions

### part4:

1. Data Preprocessing

    In the preprocess folder there is a Python script to extract the texts of the tweets from the CSV data file. The output is a text file, where each line is the text of the corresponding tweet of the CSV file.
    We only keep the alphanumeric characters and spaces. The processed data can be found in the processed_data folder. One is the entire month of data, and the other is by weeks

2. Positive and negative word lists

    There are two text files: positive_words and negative-words, they are lists of words with a positive or a negative meaning. These files can be modified if we want to introduce more negative and positive words. 

3. Positive and Negative scores

   With a map-reduce we do a word count, then we only keep the words with a count above the threshold of 1000. For each frequent word, we generate a pair (word, positive) every time a positive word is in the same tweet. We do the same with negative words.
   We count the pairs with a map-reduce. The sum  (count(word, positive) + count(word, negative)) is calculated with a map-reduce.
   The positive score will be count(word, positive) / (count(word, positive) + count(word, negative)).
   
### Usage Instructions

To run the script successfully, ensure that the required Python libraries and dependencies are installed. The script can be executed in a Jupyter notebook or any Python environment that supports the specified libraries. Additionally, users may need to adjust file paths or dataset sources based on their local setup.

Feel free to explore and modify the script to suit your specific analysis goals or integrate it into a larger data processing pipeline.
