import re
import sys
from pyspark import SparkConf, SparkContext

# Start the Spark context
conf = SparkConf()
sc = SparkContext(conf=conf)
sc.setLogLevel("WARN")

# Define your lexical fields
positive_words = sc.textFile('positive-words.txt').flatMap(lambda line: line.split(",")).collect()
negative_words = sc.textFile('negative-words.txt').flatMap(lambda line: line.split(",")).collect()

# Regroup all the words of the lexical filed that intrest us, here only positive and negative
lexical_field_words = ["positive", "negative"]


# Function to replace words with their lexical field
def replace_lexical_field(word):
    if word in positive_words:
        return "positive"
    elif word in negative_words:
        return "negative"
    else:
        return word

# Function to split and replace words in a tweet
def extract_words(tweet):
    words = re.split(r'[^\w]+', tweet)
    words = [replace_lexical_field(word) for word in words]
    return words

def extract_word_pairs(tweet, frequent_word):
    words = re.split(r'[^\w]+', tweet)
    words = [replace_lexical_field(word) for word in words]
    pairs = []
    if frequent_word in words:
        pairs = [(frequent_word, word) for word in words if word in lexical_field_words]
    return pairs


# we get the text file that regroup the pr process data
lines = sc.textFile(sys.argv[1])
words = lines.flatMap(extract_words)


word_counts = words.map(lambda word: (word, 1)).reduceByKey(lambda n1, n2: n1 + n2)

# Filter for words that occur more than 1000 times
frequent_words = word_counts.filter(lambda x: x[1] > 1000).map(lambda x: x[0])   
print("Fequent_words: ")
print(frequent_words.collect())

# Save the frequent words to the output file
# frequent_words.saveAsTextFile(sys.argv[2])

# Now we calculate the positive and negative scores of each frequent word
for frequent_word in frequent_words.collect():
    if frequent_word not in lexical_field_words:
        print("Frequent Word:", frequent_word)

        # Create word pairs using the current frequent word, with associate each word with the 
        word_pairs = lines.flatMap(lambda tweet: extract_word_pairs(tweet, frequent_word))

        # Count the occurrences of each word pair
        pair_counts = word_pairs.map(lambda pair: (pair, 1)).reduceByKey(lambda n1, n2: n1 + n2)   
        
        # Step 2: Calculate the total sum of counts
        total_sum = pair_counts.map(lambda pair_count: pair_count[1]).reduce(lambda x, y: x + y)

        # Step 3: Normalize the counts to get the pair scores
        pair_scores = pair_counts.map(lambda pair_count: (pair_count[0], pair_count[1] / total_sum))

        # Save the frequent pairs for this word to an output file
        # pair_scores.saveAsTextFile(sys.argv[2] + '/' + frequent_word)

        print("pair count: ", pair_scores.collect())


# Stop the Spark context
sc.stop()

