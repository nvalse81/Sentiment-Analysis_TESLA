# %%
import pandas as pd
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from bs4 import BeautifulSoup
from wordcloud import WordCloud, STOPWORDS
import os

# %%
import nltk
nltk.download('vader_lexicon')

# %%
finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'GOOGL', 'MSFT']

# %%
#actually getting the data from the website
news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)

    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table



# %%
news_tables['MSFT']

# %%
# Filtering based time and date
parsed_data = []

for ticker, news_table in news_tables.items():

    for row in news_table.findAll('tr'):

        if(row.a == None):
            continue
            
        titles = row.a.text
        date_data = row.td.text.split(' ')

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([ticker, date, time, titles])

# %%
news = pd.DataFrame(parsed_data, columns=['Ticker', 'Date', 'Time', 'Titles']) #creating a dataframe
news['Date'] = pd.to_datetime(news['Date']).dt.date
news['Time'] = pd.to_datetime(news['Time']).dt.time
news.head()

# %%
type(news['Time'][0])

# %%
type(news['Date'][0])

# %%
vader = SentimentIntensityAnalyzer()

# %%
scores = news['Titles'].apply(vader.polarity_scores)

# %%
scores.head()

# %%
scores_df = pd.DataFrame.from_records(scores)
scores_df.head()

# %%
scored_news = news.join(scores_df)
scored_news.head()

# %%
scored_news.to_csv('my_data.csv')

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

# Group by date and ticker columns from scored_news and calculate the mean
mean_c = scored_news.groupby(['Date', 'Ticker']).mean()

# Unstack the column ticker
mean_c = mean_c.unstack('Ticker')

# Get the cross-section of compound in the 'columns' axis
mean_c = mean_c.xs('compound', axis='columns')
# Plot a bar chart with pandas

mean_c.plot(kind='bar', figsize=(10,5), width=1)

# %%
from transformers import pipeline

model = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_task = pipeline("sentiment-analysis", model=model)
sentiment_task("Covid cases are increasing fast!")

# %%
example = "today is a bad day"
sentiment_task(example)

# %%
scored_news['roberta_sent'] = scored_news['Titles'].apply(lambda x: sentiment_task(x)[0]['label'])

# %%
scored_news['roberta_score'] = scored_news['Titles'].apply(lambda x: sentiment_task(x)[0]['score'])

# %%
scored_news.head()

# %%
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

credential = AzureKeyCredential("you_need_to_request_for_the_key")
client = TextAnalyticsClient(endpoint="https://text-sent-720.cognitiveservices.azure.com/", credential=credential)

# %%
sentence = 'Room was clean, but staff was rude.'

res = client.analyze_sentiment(documents=[sentence])

# %%
print('Scores : {}'.format(res[0]))

# %%
client.close()

# %%
from torch.nn import functional as F
import torch

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# %%
encoded_text  = tokenizer('Today is a best day', return_tensors = "pt")

# %%
output = model(**encoded_text)

# %%
softmax(output[0][0].detach().numpy()) # the first element is the negative score second is neutral and third is positive

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
sentiment_scores = scored_news['Titles'].apply(polarity_scores_roberta)

# %%
# Create new columns for the sentiment scores
scored_news['rob_positive'] = sentiment_scores.apply(lambda x: x['roberta_pos'])
scored_news['rob_negative'] = sentiment_scores.apply(lambda x: x['roberta_neg'])
scored_news['rob_neutral'] = sentiment_scores.apply(lambda x: x['roberta_neu'])

# %%
scored_news.head()

# %%
scored_news['date_time'] = pd.to_datetime(scored_news['Date'].astype(str) + ' ' + scored_news['Time'].astype(str))

# %%
scored_news.head()

# %%
new_df = scored_news[scored_news['Date'] >= pd.Timestamp('2023-04-20')]

# %%

colors = {'rob_positive':'green', 'rob_negative':'red', 'rob_neutral':'yellow'}
new_df[new_df['Ticker'] == 'AMZN'].plot.area(x='date_time', y=['rob_positive', 'rob_neutral','rob_negative' ], figsize=(15,5), title='Amazon', color = colors)
plt.legend(loc='upper left')

# %%
import yfinance as yf

# get data for a specific ticker between two dates
ticker = "AMZN"
start_date = "2023-04-20"
end_date = "2023-04-25"
data = yf.download(ticker, start=start_date, end=end_date, interval = '5m')



# %%
data['Close'].plot(figsize=(15,5), title='Amazon')

# %%
data.tail(10)

# %%
new_df[new_df['Ticker'] == 'GOOGL'].plot.area(x='date_time', y=['rob_positive', 'rob_neutral','rob_negative' ], figsize=(15,5), title='Google', color = colors)
plt.legend(loc='upper left')

# %%
ticker = "GOOGL"
start_date = "2023-04-20"
end_date = "2023-04-25"
data1 = yf.download(ticker, start=start_date, end=end_date, interval = '5m')


# %%
data1['Close'].plot(figsize=(15,5), title='Google')

# %%
new_df[new_df['Ticker'] == 'MSFT'].plot.area(x='date_time', y=['rob_positive', 'rob_neutral','rob_negative' ], figsize=(15,5), title='Microsoft', color = colors)
plt.legend(loc='upper left')

# %%
ticker = "MSFT"
start_date = "2023-04-20"
end_date = "2023-04-25"
data2 = yf.download(ticker, start=start_date, end=end_date, interval = '5m')


# %%
data2['Close'].plot(figsize=(15,5), title='Microsoft')

# %%
new_df[new_df['Ticker'] == 'AMZN'].plot(x='date_time', y='compound', figsize=(15,5), title='Amazon')

# %%
#example for vader vs roberta

# %%
ex = "This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go."

# %%
vader.polarity_scores(ex)

# %%
# Run for Roberta Model
# Run for Roberta Model
# Run for Roberta Model
encoded_text = tokenizer(ex, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
print(scores)
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)

# %%
new_df.to_csv('news_sentiment.csv', index=False)


