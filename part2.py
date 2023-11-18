################################################
# part2 
# since runing spark on vm cannot plot the graphic, so we have to separate it to part2.ipynb
#
# Extract Country, Continent, DateTime from https://www.kaggle.com/datasets/gpreda/covid19-tweets
# Last Edit 12/11/2023
# Anocha Sutaveephamochanon (A0268230J)
# >>> spark-submit part2.py
# 

import re
import sys
import math

import pyspark
from pyspark import SparkConf, SparkContext
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

import pandas as pd
import pickle
############# Anocha's utils ###########################
import unicodedata

def remove_special_encoded_characters(tweet):
    """Removes special encoded characters from a tweet."""
    tweet = unicodedata.normalize('NFKD', tweet)
    tweet = ''.join([c for c in tweet if not unicodedata.combining(c)])
    return tweet

stop_words = set(stopwords.words("english"))

def cleanChunks(s):
    # Tokenize input string
    tokens = word_tokenize(s)   
    # Remove stopwords    
    tokens = [word for word in tokens if word.lower() not in stop_words]
   
    return " ".join(tokens)

def countFreq(words):
    pairs = words.map(lambda w: (w, 1))
    counts = pairs.reduceByKey(lambda n1, n2: n1 + n2) 
    counts = counts.sortBy(lambda p: (p[1]))
    counts = counts.filter(lambda p: str(p[0]).lower() not in stop_words)
    counts = counts.filter(lambda p: len(str(p[0])) > 1)
    
    return counts

###############################################################
conf = SparkConf()
sc = SparkContext(conf=conf)

################# Load tweets #####################
lines = sc.textFile("./covid19_tweets_clearNL.csv")

rddRawTweets = lines.map(lambda l: l.split(",")) #    # clean unknown unicode when using python2 #.encode('ascii','ignore')
#### The fields are:
#   ['user_name', 'user_location', 'user_description', 'user_created',
#    'user_followers', 'user_friends', 'user_favourites', 'user_verified',
#    'date', 'text', 'hashtags', 'source', 'is_retweet']

# print("Read total {} values to RDD\n".format(rddRawTweets.count))
# print("The first 5 are {}\n".format(rddRawTweets.map(lambda l: (l[0],l[1])).take(5))) ## Test: get first 5 (user_name,user_location)

rddUserNameNLoc = rddRawTweets.map(lambda l: (l[0],l[8],l[10], l[1].strip().replace('"','')))

################ Get location n count #####################
from pyspark.sql import SparkSession
# Create another Spark session for dataframe
spark = SparkSession.builder.appName("WorkWithDataFrame").getOrCreate()

rddLocation = sc.parallelize(sorted((rddRawTweets.map(lambda l: l[1].strip().replace('"',''))) \
                .countByValue().items() \
                , key=lambda x:x[1], reverse=True \
              ))  

print('Number of unique values in \'{}\': {}'.format('rddLocation',len(rddLocation.collect())))
print('Top 20 places in \'{}\': {}'.format('rddLocation',rddLocation.collect()[:20]))

pd.to_pickle(spark.createDataFrame(rddLocation,["user_location","count"]).toPandas(), "./dfrddLocation.pkl") # for plot
print('--------------------------------------')

# Read the CSV file into a DataFrame
dfOrg = spark.read.csv("./covid19_tweets.csv", header=True, inferSchema=True)
dfOrg.show()

# Read the CSV file into a DataFrame
dfRaw = spark.read.csv("./covid19_tweets_clearNL.csv", header=True, inferSchema=True)
dfRaw.show()


################################################################
################ Get country n continent #######################
# from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, expr
from geopy.geocoders import Nominatim
import pycountry_convert as pc

#
# #### define User-Define-Function ####
#

def map_loc_to_country(loc): #using geopy
    geolocator = Nominatim(user_agent="loc_to_country_mapper")
    location = geolocator.geocode(loc)
    if location:
        return location.address.split(",")[-1].strip()
    else:
        return None


def map_country_to_continent(country): #using geopy
    try:
        country_alpha2 = pc.country_name_to_country_alpha2(country)
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    except:
        country_continent_name = "NA"
    return country_continent_name



import requests

client = googlemaps.Client(enterprise_credentials="-Personal-")

def gmap_loc_to_country(loc): #using googlemap
    
    api_key = "-Personal-"
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"

    out = ['','','',''] #countrylongname,countryshortname, lat,lng

    # Set up the parameters for the API request
    params = {
        "address": loc,
        "key": api_key,
    }

    # Send the API request
    response = requests.get(base_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])

        if results:
            # The country is usually found in the "country" component of the address
            address_components = results[0].get("address_components", [])
            for component in address_components:
                if "country" in component.get("types", []):
                    # print("Country for {}: {}".format(loc,component["long_name"]))
                    country_name = component["long_name"]
                    out[0] = component["long_name"]
                    out[1] = component["short_name"]
                    # return country_name
            geometry = results[0].get("geometry",[])                    
            out[2] = geometry["location"]["lat"]
            out[3] = geometry["location"]["lng"]
            return out
                
        else:
            # print("Location {} not found.".format(loc))
            return "NA"
    else:
        # print("API request failed for {} with status code: {}".format(loc,response.status_code))
        return "NA"


#### user_location to country to continent


dateCountryList = rddUserNameNLoc.map(lambda l:(l[1],l[2],l[3])).collect()
df = spark.createDataFrame(dateCountryList, ["date","hashtags","user_location"])
# countryList = rddLocation.map(lambda l:(l[0],)).collect()
# df = spark.createDataFrame(countryList, ["user_location"])

spark.udf.register("gmap_loc_to_country",gmap_loc_to_country)
df = df.withColumn("country", expr("gmap_loc_to_country(user_location)"))# countryLongName, countryShortName, Lat, Lng

df.show()


with open('./dfLocationCountry.pkl', 'wb') as f:
    pickle.dump(df.rdd.collect(), f)




################# user_location => country query related
# 
# #### New session for http request
spark2 = SparkSession.builder.appName("LocToContinentMapping").getOrCreate()

# # Register map_country_to_continent as a UDF
spark2.udf.register("map_country_to_continent", map_country_to_continent)

# Use the UDF to map "country" to "continent" and create a new column
df = df.withColumn("continent", lit(None))
df = df.withColumn("continent", col("continent").cast("string"))

# Map the "country" column to the "continent" column using the UDF
df = df.withColumn("continent", expr("map_country_to_continent(country)"))

# # Show the DataFrame
df.show()

with open('./dfLocationCountryContinent.pkl', 'wb') as f:
    pickle.dump(df.rdd.collect(), f)



# Register map_loc_to_country as a  UDF
spark2.udf.register("map_loc_to_country", map_loc_to_country)

# Use the UDF to map "user_location" to "country" and create a new column
df = df.withColumn("country", lit(None))
df = df.withColumn("country", col("country").cast("string"))

# Map the "town" column to the "country" column using the UDF
df = df.withColumn("country", expr("map_loc_to_country(user_location)"))



# import pickle
# # Save the DataFrame to a pickle file
# # Error ---> ("buffering=True") coz requests ver in geopy cannot not mapy with request in vm -> use google map API instead to control requests version
try:
    with open('./loc_country_continent.pkl', 'wb') as f:
        pickle.dump(df.rdd.collect(), f)
except:
    print("cannot save the pickle")

# df.rdd.saveAsPickleFile("./loc_country_continent.pkl")
spark2.stop()


# #### Group by country the occurrences of each country
country_counts = df.groupBy("country").count()
country_counts.show()
rdd = sc.parallelize(dfrdd)

rddCountry = sc.parallelize(sorted((rdd.map(lambda l: (l[1].strip().replace('"','')))) \
                .countByValue().items() \
                , key=lambda x:x[1], reverse=True \
              ))  

print('Number of unique countries in \'{}\': {}'.format('rddCountry',len(rddCountry.collect())))
print('Top 20 places in \'{}\': {}'.format('rddCountry',rddCountry.collect()[:20]))

print('--------------------------------------')

rddContinent = sc.parallelize(sorted((rdd.map(lambda l: (l[2].strip().replace('"','')))) \
                .countByValue().items() \
                , key=lambda x:x[1], reverse=True \
              ))  

print('Number of unique continents in \'{}\': {}'.format('rddContinent',len(rddCountry.collect())))
print('Counts in \'{}\': {}'.format('rddContinent',rddContinent.collect()))

print('--------------------------------------')

# Group by continent and count the occurrences of each continent
continent_counts = df.groupBy("continent").count()
continent_counts.show()

# #### Save the DataFrame to a pickle file
continent_counts.rdd.saveAsPickleFile("./continent_count.pkl")


################## Get polarity from text #######################
#

from pyspark.sql import SparkSession

# spark = SparkSession.builder.appName("ReadDataFrameFromPickle").getOrCreate()
with open("./dfLocationCountryContinent.pkl", 'rb') as file:
    load_df = pickle.load(file)

dfCntry = spark.createDataFrame(load_df)
dfCntry.show()
print("dfCntry",dfCntry.count())
# left join on two dataframes 
dfAll = dfRaw.join(dfCntry, dfRaw.user_location == dfCntry.user_location, "left")
print("dfAll",dfAll.count())
dfAll.show() 
dfW = dfAll.drop(*[ 'user_description', 'user_created', 'user_followers', 'user_friends', 'user_favourites', 'user_verified','source', 'is_retweet'])
print("dfW",dfW.count())
dfW.show()

with open('./dfTimeCountryContinent.pkl', 'rb') as f:
    load_df = pickle.load(f)

dfTC = spark.createDataFrame(load_df)
dfTC.show()


from textblob import TextBlob
def getPolarity(text):
    try:
        p = TextBlob(text).sentiment[0]
    except:
        p = 0.0
    return p

# # Register map_loc_to_country as a  UDF
spark.udf.register("getPolarity", getPolarity)

dfTC = dfTC.withColumn("polarity", expr("getPolarity(text)"))
dfTC.show()

with open('./dfTimeCountryContinentPolar.pkl', 'wb') as f:
    pickle.dump(dfTC.rdd.collect(), f)

with open('./dfUserDateContinent_.pkl', 'rb') as f:
    load_df = pickle.load(f)

dfTC = spark.createDataFrame(load_df)
df_cols = dfTC.columns

# Get index of the duplicate columns
duplicate_col_index = [idx for idx, val in enumerate(df_cols) if val in df_cols[:idx]]
for i in duplicate_col_index:
    df_cols[i] = df_cols[i] + '_joined'
dfTC = dfTC.toDF(*df_cols)
dfTC.show()
with open('./dfUserDateContinent.pkl', 'wb') as f:
    pickle.dump(dfTC.rdd.collect(), f)


pd.to_pickle(dfTC.toPandas(), "./dfAllwPolar.pkl")  # For plot





#######################################################
################# date-time related
#

dfW = dfTC

from pyspark.sql.functions import col, row_number
from pyspark.sql.types import TimestampType
from pyspark.sql.functions import year, month, dayofmonth
from pyspark.sql.functions import avg, count, date_format, desc, asc, concat, rank, first

# Use the withColumn method to cast the column to a timestamp data type
dfW = dfW.withColumn("date", dfW["date"].cast(TimestampType()))

dfW = dfW.withColumn("year", year("date"))
dfW = dfW.withColumn("month", month("date"))
dfW = dfW.withColumn("dayofmonth", dayofmonth("date"))
dfw = dfW.withColumn("continent", dfW.continent)
dfW = dfW.withColumn("country", dfW.country)
dfW = dfW.withColumn("user", dfW.user_name)
dfW = dfW.withColumn('text',dfW.text)

###########################
# # Datetime vs tweet plot
# # 
df = dfW.filter("year = 2020")
df = df.filter("month != 4")
df = df[['dateNoTime','user','text','dayofmonth','month','year']]
df = df.groupBy("dateNoTime","user").agg( count('text'), first('dayofmonth'), first('month'), first('year')) \
                .withColumnRenamed("count(text)", "countText") \
                .withColumnRenamed("first(dayofmonth, false)", "dayofmonth") \
                .withColumnRenamed("first(month, false)", "month") \
                .withColumnRenamed("first(year, false)", "year")                              
                    
df = df.orderBy("dateNoTime",col("count(text)").desc())
df = df.withColumn('user-date', concat(col("user"), lit('_'),col("dateNoTime")))
df.show()
df = df.groupBy("dateNoTime").agg(first('user-date'), max('countText'), first('dayofmonth'), first('month'), first('year')) \
                .withColumnRenamed("max(countText)", "maxCountText") \
                .withColumnRenamed("first(user-date, false)", 'user-date') \
                .withColumnRenamed("first(dayofmonth, false)", "dayofmonth") \
                .withColumnRenamed("first(month, false)", "month") \
                .withColumnRenamed("first(year, false)", "year")                              
    
df = df.orderBy(col('dayofmonth').asc(),col('month').asc(),col('year').asc())
df = df.drop(*["year", "month", "dayofmonth","dateNoTime"])
df.show()
pd.to_pickle(df.toPandas(), "./dfDateUser.pkl")



###########################
# Datetime vs location
df = dfW.filter("year = 2020")
df = df.filter("month != 4")
df = df.groupBy("continent").agg({'text':'count'})
df.show()
pd.to_pickle(df.toPandas(), "./dfContinentCount.pkl") # for plot


df = dfW.filter("year = 2020")
df = df.filter("month != 4")
df = df.groupBy("year","month","dayofmonth","text").agg({'text':'count'}).withColumnRenamed("count(text)", "count")
df = df.orderBy(["year", "month", "dayofmonth","count"], ascending=[True, True, True, False])
df = df.withColumn("date", concat(col('year'), lit('_'), col('month'), lit('_'), col('dayofmonth')))
# df = df.drop(*['year','month','dayofmonth'])
df = df.groupBy("year","month","dayofmonth","date").agg({"date":'sum'})
df = df.orderBy(["year", "month", "dayofmonth"], ascending=[True, True, True])
df.show()
pd.to_pickle(df.toPandas(), "./dfDateCount.pkl") # for plot


df = dfW.filter("year = 2020")
df = df.filter("month != 4")
df = df.groupBy("year","month","dayofmonth","continent").agg({'polarity':'avg', 'continent':'count'})
df = df.orderBy(["year", "month", "dayofmonth","continent"], ascending=[True, True, True, False])
df = df.withColumn("date", concat(col('year'), lit('_'), col('month'), lit('_'), col('dayofmonth')))
df.show()
pd.to_pickle(df.toPandas(), "./dfDateContinetPolar.pkl") # for plot

# with open('./dfTimeContinentPolarAvg.pkl', 'wb') as f:
#     pickle.dump(df.rdd.collect(), f)

df = dfW.filter("year = 2020")
df = df.filter("month != 4")
df = df.groupBy("year","month","dayofmonth","country").agg({'polarity':'avg', 'country':'count'})
df = df.orderBy(["year", "month", "dayofmonth","country"], ascending=[True, True, True, False])
df = df.withColumn("date", concat(col('year'), lit('_'), col('month'), lit('_'), col('dayofmonth')))
df.show()
# pd.to_pickle(df.toPandas(), "./dfDateCountryPolar.pkl")


pivoted = (df
    .groupBy("date")
    .pivot(
        "Continent",
        ["Asia", "Oceania", "South America", "North America", "Antartica", "Africa", "Europe"])  # Optional list of levels
    .sum("count"))  
pivoted = pivoted.orderBy(["date"], ascending=[True])
pivoted.show()
pivoted.write.csv('./dfDateContinentPivot.csv')
pivoted.repartition(1) \
   .write.format("com.databricks.spark.csv") \
   .option("header", "true") \
   .save('./dfDateContinentPivot.csv')
pd.to_pickle(pivoted.toPandas(), "./dfDateContinentPivot.pkl")
with open('./dfDateContinentPivot_.pkl', 'wb') as f:  # for plot
    pickle.dump(pivoted.rdd.collect(), f)

df = dfW.filter("year = 2020")
df = df.filter("month != 4")
df = df.withColumn("date", concat(col('year'), lit('_'), col('month'), lit('_'), col('dayofmonth')))
df = df.groupBy("date","country","continent").agg({'date':'count', 'country':'count'})
df = df.orderBy(["continent","country","date"], ascending=[True, True, True])
df.show()
# pd.to_pickle(df.toPandas(), "./dfDateCountryContinent.pkl")

# with open('./dfTimeCountryPolarAvg.pkl', 'wb') as f:
#     pickle.dump(df.rdd.collect(), f)


with open('./dfTimeCountryContinentPolarAvg.pkl', 'rb') as f:
    load_df = pickle.load(f)    

dfTC = spark.createDataFrame(load_df)
df = df.filter("continent != null")
dfTC.show()
# dfTC.write.csv('./dateContiCountPolarCsv')

df = dfTC.withColumn("date_continent", concat(col('year'), lit('_'), col('month'), lit('_'), col('dayofmonth'), lit('_'), col('continent')))
df.show()

df = df.drop(*["year", "month", "dayofmonth","continent"])
df = df.select(col("date_continent"),col('count(continent)').alias('count_continent'), col('avg(polarity)').alias('avg_polarity'))
df = df.filter("count_continent != 0")
df.show(200)


df = dfW.groupBy("year", "month").count()
df.filter("year = 2020").show()
# df.sort(desc("year")).show(110) #all years found in the corpus
# print("Total outliers: ",sum(df.filter("year != 2020")['count'].collect()))
# print(df.filter("year != 2020").columns())
print(df.filter("year != 2020").columns)

# print(df.count())

df = dfW.groupBy("year", "continent").count()
df.filter("year = 2020").show()


df = dfW.groupBy("year","month","continent").count()
df.filter("year = 2020").show()
df.filter("year = 2020").orderBy(["year", "month", "continent"], ascending=[False, False, False]).show()
# print("Total not 2020: ", df.filter("year != 2020").count.sum())

df = dfW.groupBy("year","month","country").count()
df.filter("year = 2020").show()
df.filter("year = 2020").orderBy(["year", "month", "country"], ascending=[False, False, False]).show()
df.show()

from pyspark.sql.window import Window
df = dfW.filter("year = 2020")
df = df.filter("month != 4")

df = df.groupBy("continent","country").agg(count('text')) \
    .withColumnRenamed("count(text)", "countText")
df = df.orderBy(["continent","country","countText"], ascending=[True, True, False])
df.show()

windowContinent = Window.partitionBy("continent").orderBy(col("countText").desc())
df = df.withColumn("row",row_number().over(windowContinent)) \
  .filter(col("row") <= 5) \
  .drop("row")
df.show()  
df = df.withColumn("Country-Continent",  concat(col('country'), lit('_'), col('continent')))  
df = df.drop(*['country','continent'])
df.show()
pd.to_pickle(df.toPandas(), "dfTop5CountryContinent.pkl")




##############
# stop 2 separate spark sessions
spark.stop()
sc.stop()

# Thank you. 
