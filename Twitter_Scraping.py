import snscrape.modules.twitter as sntwitter
import pandas as pd
import datetime
from datetime import date
import streamlit as st
import pymongo
from pymongo import MongoClient

today=dte.today()
now=today.strftime('%Y-%m-%d')
To=now
From='2023-01-01'

client=pymongo.MongoClient("mongodb+srv://manjupriya:Manju2000@cluster0.cqgge1g.mongodb.net/test")
db = client.Guvi
collection= db.Twitter_Scrapping
img = Image.open("media/twitter.png")
st.set_page_config(page_title="Twitter scraping",page_icon = img,layout = "wide")

def main(Name):
  tweets = []
  for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'{Name} since:{From} until:{To}').get_items()):
    if len(tweets)>=5:
        break
    tweets.append([tweet.date,tweet.id,tweet.user.username,tweet.url,tweet.rawContent,tweet.likeCount,tweet.lang,tweet.source])
    
  tweets_df = pd.DataFrame(tweets, columns=['Datetime', 'Tweet Id','User Name','URL','Content','LikeCount','Language','Source'])
  return tweets_df.head()

main(input())

tweets_df.reset_index(inplace=True)
data_dict = tweets_df.("records")
collection.insert_many(data_dict)

result=st.text_area("Name")
st.date_input("From")
st.date_input("To")
if result:
  st.json(data_dict)
  st.write(data_dict)
  
