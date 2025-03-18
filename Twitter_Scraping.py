import snscrape.modules.twitter as sntwitter
import pandas as pd
import streamlit as st
import pymongo
from pymongo import MongoClient
from PIL import Image
from datetime import datetime, date

# MongoDB Connection
client = pymongo.MongoClient("mongodb+srv://manjupriya:Manju2000@cluster0.cqgge1g.mongodb.net/test")
db = client.Guvi
collection = db.Twitter_Scrapping

# Streamlit UI Config
img = Image.open("media/twitter.png")
st.set_page_config(page_title="Twitter Scraping", page_icon=img, layout="wide")

# Streamlit Inputs
result = st.text_input("Enter Search Term")
from_date = st.date_input("From", value=date(2023, 1, 1))
to_date = st.date_input("To", value=date.today())

# Function to Scrape Tweets
def scrape_tweets(search_term, from_date, to_date, limit=5):
    tweets = []
    for i, tweet in enumerate(
        sntwitter.TwitterSearchScraper(f'{search_term} since:{from_date} until:{to_date}').get_items()
    ):
        if len(tweets) >= limit:
            break
        tweets.append({
            "Datetime": tweet.date,
            "Tweet Id": tweet.id,
            "User Name": tweet.user.username,
            "URL": tweet.url,
            "Content": tweet.rawContent,
            "LikeCount": tweet.likeCount,
            "Language": tweet.lang,
            "Source": tweet.source
        })
    return tweets

# Scrape & Display Tweets
if result:
    tweets_data = scrape_tweets(result, from_date, to_date)
    if tweets_data:
        tweets_df = pd.DataFrame(tweets_data)
        st.write(tweets_df)

        # Store in MongoDB
        collection.insert_many(tweets_data)
        st.success("Data saved to MongoDB successfully!")
    else:
        st.warning("No tweets found for the given query.")
