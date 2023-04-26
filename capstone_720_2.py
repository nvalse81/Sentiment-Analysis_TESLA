# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as tkr
import seaborn as sns
import nltk
# nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
import re

# %%
prices=pd.read_csv('stock_yfinance_data.csv') # read in stock data yfinance

# %%
type(prices['Date'][0])

# %%
prices['Date'] = pd.to_datetime(prices['Date']).dt.date.astype('datetime64[ns]')

# %%
type(prices['Date'][0])

# %%
prices.head()

# %%
prices=prices.sort_values(by=['Date']).reset_index()
prices.head(15)
prices=prices.rename(columns={"Stock Name":"ticker_symbol","Date": "date", "Close":"close_value", "Volume":"volume", "Open":"open_value", "High":"high_value", "Low":"low_value"})

# %%
prices.head()

# %%
def create_indicators(data):
    
    prices = data.sort_values(by=['date']).reset_index()

    # create simple moving average
    n=[10,20,50,100]
    for i in n:
        prices.loc[:,(str("MA"+str(i)))]=prices['close_value'].rolling(i).mean()    

    # Calculate MACD  
    day26=prices['close_value'].ewm(span=26, adjust=False).mean()
    day12=prices['close_value'].ewm(span=12, adjust=False).mean()
    prices.loc[:,('macd')]=day12-day26 
    prices.loc[:,('signal')]=prices['macd'].ewm(span=9, adjust=False).mean()

    # Calculate RSI 
    up = np.log(prices.close_value).diff(1)
    down = np.log(prices.close_value).diff(1)

    up[up<0]=0
    down[down>0]=0

    # Calculate the EWMA
    roll_up = up.ewm(span=14).mean()
    roll_down = down.abs().ewm(span=14).mean()

    # Calculate the RSI based on EWMA
    RS1 = roll_up / roll_down
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    prices.loc[:,('rsi')]=RSI1

    return prices

# create dict, by ticker
d = dict(tuple(prices.groupby('ticker_symbol')))



# %%
print(d)

# %%
d = {k:create_indicators(v) for k, v in d.items()}

# %%
d['TSLA'].head(150)

# %%
# get subset of prices from start to end date
def subset_prices(d,ticker,start,end):
    x=d[ticker]
    x=x[((x.date>=start)&(x.date<=end))]
    return x

# %%
# from azure.ai.textanalytics import TextAnalyticsClient
# from azure.core.credentials import AzureKeyCredential



# %%
# credential = AzureKeyCredential("908ac542166e4abaa90153eb9648e2ee")
# client = TextAnalyticsClient(endpoint="https://text-sent-720.cognitiveservices.azure.com/", credential=credential)

# %%
# sentence = 'Room was clean, but staff was rude.'

# res = client.analyze_sentiment(documents=[sentence])

# %%
# print('Scores : {}'.format(res[0]))

# %%
# client.close()

# %%
tweets = pd.read_csv('stock_tweets.csv')

# %%
tweets['date'] = pd.to_datetime(tweets['Date']).dt.date.astype('datetime64[ns]')

# %%
tweets['time'] = pd.to_datetime(tweets['Date'])

# %%
tweets['time'] = tweets['time'].dt.time

# %%
tweets['Date'] = pd.to_datetime(tweets['Date'])

# %%
type(tweets['Date'][0])

# %%
tweets.tail()

# %%
merged_df = pd.merge(tweets[tweets['Stock Name']=='TSLA'], d['TSLA'], on=['date'], how='outer')

# %%
merged_df.head()

# %%
grouped_df = merged_df.groupby('date').agg({'Tweet': 'count', 'volume': 'mean'})

# %%
grouped_df.head()

# %%

grouped_df.dropna().plot(y='Tweet', label='Tweet Volume',figsize=(12, 4))
plt.title('Tweets Volume')
plt.xlabel('Date')
plt.ylabel('Volume')
grouped_df.dropna().plot(y='volume', label='Trading Volume',figsize=(12, 4))

plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Trading Volume')
plt.legend()
plt.show()


# %%
grouped_df.dropna().corr()

# %%
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

# %%
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# %%
example = "Mainstream media has done an amazing job at brainwashing people. Today at work, we were asked what companies we believe in  and  I said @Tesla because they make the safest cars  and  EVERYONE disagreed with me because they heardâ€œthey catch on fire  and  the batteries cost 20k to replaceâ€"


# %%
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
print(output)
scores = output[0][0].detach().numpy()
print(scores)
import torch.nn.functional as F
probs = F.softmax(output.logits, dim=-1)
probs[0][0]=0.0
probs[0][1]=0.792
probs[0][2]=0.208
print(probs)
weights = torch.tensor([1, 0, -1])  # positive, neutral, negative
score = (probs * weights).sum().item()
print(score,"&&&&&&")

print(scores)
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)


# %%
def Preprocess_Tweets(data):
		
	data['Text_Cleaned'] = data['Tweet']

	## FIX HYPERLINKS
	data['Text_Cleaned'] = data['Text_Cleaned'].replace(r'https?:\/\/.*[\r\n]*', ' ',regex=True)
	data['Text_Cleaned'] = data['Text_Cleaned'].replace(r'www.*[\r\n]*', ' ',regex=True)
	data['Text_Cleaned'] = data['Text_Cleaned'].str.replace('https', '', regex=False)


	## FIX EMOJIS
	data['Text_Cleaned'] = data['Text_Cleaned'].str.replace(':)', '', regex=False)
	data['Text_Cleaned'] = data['Text_Cleaned'].str.replace(':-)', '', regex=False)
	data['Text_Cleaned'] = data['Text_Cleaned'].str.replace(':(', '', regex=False)
	data['Text_Cleaned'] = data['Text_Cleaned'].str.replace(':-(', '', regex=False)
	data['Text_Cleaned'] = data['Text_Cleaned'].str.replace('0_o', '', regex=False)
	data['Text_Cleaned'] = data['Text_Cleaned'].str.replace(';)', '', regex=False)
	data['Text_Cleaned'] = data['Text_Cleaned'].str.replace('=^.^=', '', regex=False)
	data['Text_Cleaned'] = data['Text_Cleaned'].str.replace(':-D', '', regex=False)
	data['Text_Cleaned'] = data['Text_Cleaned'].str.replace(':D', '', regex=False)
 
	## Other
	data['Text_Cleaned'] = data['Text_Cleaned'].str.replace('&amp;', ' and ', regex=False)
	
	   
		


	return data

# %%
Preprocess_Tweets(merged_df)
merged_df.head()

# %%
merged_df.to_csv('cleaned_tweets.csv', index=False)

# %%
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

# %%
sentiment_scores = merged_df['Text_Cleaned'].apply(polarity_scores_roberta)

# %%
# Create new columns for the sentiment scores
merged_df['positive'] = sentiment_scores.apply(lambda x: x['roberta_pos'])
merged_df['negative'] = sentiment_scores.apply(lambda x: x['roberta_neg'])
merged_df['neutral'] = sentiment_scores.apply(lambda x: x['roberta_neu'])

# %%
merged_df.to_csv('cleaned_tweets_with_scores.csv', index=False)

# %%
merged_df.head()

# %%
# Group the DataFrame by date and calculate mean sentiment scores
grouped_data = merged_df[merged_df['date'] > pd.to_datetime('2022-08-01')].groupby('date')[['positive', 'neutral', 'negative']].mean()

grouped_data = grouped_data.div(grouped_data.sum(axis=1), axis=0)
# Reshape the DataFrame using the melt() method
melted_data = grouped_data.melt(var_name='sentiment', value_name='score', ignore_index=False)
colors = {'positive': 'green', 'neutral': 'yellow', 'negative': 'red'}
# Create the stacked barplot using Seaborn
plt.figure(figsize=(15, 8))
sns.barplot(x=melted_data.index, y='score', hue='sentiment', data=melted_data, palette=colors, orient='v')
plt.xticks()

# %%
tsl = d['TSLA']
tsl[tsl['date'] > pd.to_datetime('2022-08-01')].plot(x = 'date',y='close_value', label='Closing',figsize=(15, 6))
plt.ylabel('Price')

# %%
groups = merged_df[merged_df['date'] > pd.to_datetime('2022-08-01')].groupby('date').agg({'Text_Cleaned': 'count', 'volume': 'mean', 'positive': 'mean', 'neutral': 'mean', 'negative': 'mean', 'MA10': 'mean', 'MA20': 'mean', 'MA50': 'mean', 'MA100': 'mean', 'macd': 'mean','rsi': 'mean'})
cols_to_plot = ['tweets_vol','positive', 'neutral', 'negative', 'MA10', 'MA20', 'MA50', 'MA100', 'macd', 'volume','rsi']
groups.rename(columns={'Text_Cleaned': 'tweets_vol'}, inplace=True)
corr_matrix = groups[cols_to_plot].corr()

# Plot the correlation matrix using seaborn heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Plot')
plt.show()

# %%
df_argmax = merged_df

# %%
grouped_df.head()

# %%
# Importing Wordcloud for visualization
from wordcloud import WordCloud
# Importing stopwords to remove those english word which will not add any value to our model
from nltk.corpus import stopwords

# %%
text_neg = " ".join(merged_df[merged_df['negative'] > 0.7]['Text_Cleaned'].tolist())

# %%
wordcloud = WordCloud(width = 2000, height = 1000, random_state=42, background_color='black', collocations=True, stopwords = stopwords.words('english')).generate(text_neg)
plt.figure(figsize=(20, 30))
# Display image
plt.imshow(wordcloud) 
plt.axis("off")
plt.show()

# %%
wordcloud = WordCloud(width = 2000, height = 1000, random_state=42, background_color='white', collocations=False, stopwords = stopwords.words('english')).generate(text_neg)
plt.figure(figsize=(20, 30))
# Display image
plt.imshow(wordcloud) 
plt.axis("off")
plt.show()

# %%
text_pos = " ".join(merged_df[merged_df['positive'] > 0.7]['Text_Cleaned'].tolist())

# %%
wordcloud = WordCloud(width = 2000, height = 1000, random_state=42, background_color='black', collocations=True, stopwords = stopwords.words('english')).generate(text_pos)
plt.figure(figsize=(20, 30))
# Display image
plt.imshow(wordcloud) 
plt.axis("off")
plt.show()

# %%
wordcloud = WordCloud(width = 2000, height = 1000, random_state=42, background_color='white', collocations=False, stopwords = stopwords.words('english')).generate(text_pos)
plt.figure(figsize=(20, 30))
# Display image
plt.imshow(wordcloud) 
plt.axis("off")
plt.show()

# %%
text_pos_1 = " ".join(merged_df[merged_df['positive'] > 0.5]['Text_Cleaned'].tolist())

# %%
wordcloud = WordCloud(width = 2000, height = 1000, random_state=42, background_color='white', collocations=False, stopwords = stopwords.words('english')).generate(text_pos_1)
plt.figure(figsize=(20, 30))
# Display image
plt.imshow(wordcloud) 
plt.axis("off")
plt.show()

# %%
text_neg_1 = " ".join(merged_df[merged_df['negative'] > 0.5]['Text_Cleaned'].tolist())

# %%
wordcloud = WordCloud(width = 2000, height = 1000, random_state=42, background_color='white', collocations=False, stopwords = stopwords.words('english')).generate(text_neg_1)
plt.figure(figsize=(20, 30))
# Display image
plt.imshow(wordcloud) 
plt.axis("off")
plt.show()


