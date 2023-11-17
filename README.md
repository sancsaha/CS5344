# Analyzing the Impact of COVID-19 Using Twitter Data

## Project Overview


## Tasks Performed
These tasks collectively contribute to a thorough exploration and understanding of the COVID-19 discourse on Twitter, providing valuable insights into topics, sentiment, and user trends during the specified timeframe.
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
### part3.py:
### part4.py:

### Usage Instructions

To run the script successfully, ensure that the required Python libraries and dependencies are installed. The script can be executed in a Jupyter notebook or any Python environment that supports the specified libraries. Additionally, users may need to adjust file paths or dataset sources based on their local setup.

Feel free to explore and modify the script to suit your specific analysis goals or integrate it into a larger data processing pipeline.
