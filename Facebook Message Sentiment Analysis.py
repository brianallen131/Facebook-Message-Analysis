
# Import the messages dataset
import pandas as pd
fb = pd.read_csv('fb_data_features.csv')


# Import NLP sentiment analyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def scoreSentiment(text):
    try:
        return sia.polarity_scores(text)['pos'] 
    except:
        return .5


# Calculate a positivity score for each message and export to csv
fb["positivity_score"]=fb.apply(lambda x: scoreSentiment(x['text']), axis=1)
fb.to_csv("fb_data_features_sentiment.csv")

scoreSentiment("ew, gross!")




