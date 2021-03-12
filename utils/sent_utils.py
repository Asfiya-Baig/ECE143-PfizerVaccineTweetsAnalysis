import re
import nltk
import random
import string
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.express as ex
import plotly.graph_objs as go
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from wordcloud import WordCloud,STOPWORDS
from textblob import TextBlob

def cal_sub_pol(df):
    """calculates subjectivity and polarity score using textblob
    
    Args:
        df ([dataframe]): input dataframe
    """
    
    assert isinstance(df, pd.DataFrame), "input must be dataframe"
    def getTextSubjectivity(txt):
        return TextBlob(txt).sentiment.subjectivity
    def getTextPolarity(txt):
        return TextBlob(txt).sentiment.polarity
    df['Subjectivity'] = df['text'].apply(getTextSubjectivity)
    df['Polarity'] = df['text'].apply(getTextPolarity)

    def getTextAnalysis(a):
        """assign sentiment based on polarity scores
        
        Args:
            a (float): polarity score
        """
        assert isinstance(a,float), "expect score to be float"
        
        if a < 0:
            return "Negative"
        elif a == 0:
            return "Neutral"
        else:
            return "Positive"

    def getSubjectivity(b):
        """assign objectivity based on objectivity scores
        
        Args:
            b (float): objectivity score
        """
        assert isinstance(b,float), "expect score to be float"
        
        if b < 0.5: 
            return "Objective"
        else:
            return "Subjective"

    df['Score'] = df['Polarity'].apply(getTextAnalysis)
    df['SubScore'] = df['Subjectivity'].apply(getSubjectivity)

def plot_all_sent(df):
    """plot sentiment percentage of all tweets
    
    Args:
        df ([dataframe]): input dataframe
    """
    assert isinstance(df, pd.DataFrame), "input must be dataframe"
    
    labels = df.groupby('Score').count().index.values
    values = df.groupby('Score').size().values
    plt.bar(labels, values, color = ['red', 'blue', 'lime'])
    plt.title(label = "Textblob Sentiment Analysis", 
                      fontsize = '15')

    positive = df[df['Score'] == 'Positive']
    print(str(positive.shape[0]/(df.shape[0])*100) + " % of positive tweets")
    positive = df[df['Score'] == 'Neutral']
    print(str(positive.shape[0]/(df.shape[0])*100) + " % of neutral tweets")
    positive = df[df['Score'] == 'Negative']
    print(str(positive.shape[0]/(df.shape[0])*100) + " % of negative tweets")
    plt.show()
    
def plot_sub_sent(df):
    """plot sentiment percentage of subjective tweets
    
    Args:
        df ([dataframe]): input dataframe
    """
    assert isinstance(df, pd.DataFrame), "input must be dataframe"
    
    f_sub = df.loc[df['SubScore'] == 'Subjective']

    labels = f_sub.groupby('Score').count().index.values
    values = f_sub.groupby('Score').size().values
    plt.bar(labels, values, color = ['red', 'blue', 'lime'])
    plt.title(label = "Subjective Sentiments", 
                      fontsize = '15')

    positive = f_sub[f_sub['Score'] == 'Positive']
    print(str(positive.shape[0]/(f_sub.shape[0])*100) + " % of positive tweets")
    positive = f_sub[f_sub['Score'] == 'Neutral']
    print(str(positive.shape[0]/(f_sub.shape[0])*100) + " % of neutral tweets")
    positive = f_sub[f_sub['Score'] == 'Negative']
    print(str(positive.shape[0]/(f_sub.shape[0])*100) + " % of negative tweets")
    plt.show()
    
def plot_obj_sent(df):
    """plot sentiment percentage of objective tweets
    
    Args:
        df ([dataframe]): input dataframe
    """
    assert isinstance(df, pd.DataFrame), "input must be dataframe"
    
    f_ob = df.loc[df['SubScore'] == 'Objective']

    labels = f_ob.groupby('Score').count().index.values
    values = f_ob.groupby('Score').size().values
    plt.bar(labels, values, color = ['red', 'blue', 'lime'])
    plt.title(label = "Objective Sentiments", 
                      fontsize = '15')

    positive = f_ob[f_ob['Score'] == 'Positive']
    print(str(positive.shape[0]/(f_ob.shape[0])*100) + " % of positive tweets")
    positive = f_ob[f_ob['Score'] == 'Neutral']
    print(str(positive.shape[0]/(f_ob.shape[0])*100) + " % of neutral tweets")
    positive = f_ob[f_ob['Score'] == 'Negative']
    print(str(positive.shape[0]/(f_ob.shape[0])*100) + " % of negative tweets")
    plt.show()
    
def plot_com(df):
    """plot common words in most positive and negative tweets
    
    Args:
        df ([dataframe]): input dataframe
    """
    assert isinstance(df, pd.DataFrame), "input must be dataframe"
    
    most_pos = df[df['Polarity'].between(0.7,1)]
    most_neg = df[df['Polarity'].between(-1,-0.7)]

    pos_text = ' '.join(most_pos.text)
    neg_text = ' '.join(most_neg.text)

    pwc = WordCloud(width=2400,height=1600,colormap='summer',background_color='white').generate(pos_text)
    nwc = WordCloud(width=2400,height=1600,colormap='autumn',background_color='white').generate(neg_text)

    f, axs = plt.subplots(2,2,figsize=(15,15))
    plt.subplot(2,1,1)
    plt.title('Common Words in Most Positive Tweets',fontsize=16,color = 'green')
    plt.imshow(pwc)
    plt.axis('off')
    plt.subplot(2,1,2)
    plt.title('Common Words in Most Negative Tweets',fontsize=16,color = 'red')
    plt.imshow(nwc)
    plt.axis('off')
    plt.show()

def plot_most(df):
    """plot most positive and negative words in all tweets
    
    Args:
        df ([dataframe]): input dataframe
    """
    assert isinstance(df, pd.DataFrame), "input must be dataframe"
    
    most_pos = df[df['Polarity'].between(0.7,1)]
    most_neg = df[df['Polarity'].between(-1,-0.7)]
    pos_text = ' '.join(most_pos.text)
    neg_text = ' '.join(most_neg.text)

    pos_dict = dict()
    for word in pos_text.split():
        w = word.strip()
        pos_dict[w] = pos_dict.get(w,0)+1
    pos_dict = {k: v for k, v in sorted(pos_dict.items(), key=lambda item: item[1],reverse=True)}

    neg_dict = dict()
    for word in neg_text.split():
        w = word.strip()
        neg_dict[w] = neg_dict.get(w,0)+1
    neg_dict = {k: v for k, v in sorted(neg_dict.items(), key=lambda item: item[1],reverse=True)}

    top_10_pos = list(pos_dict.keys())[:8000]
    top_10_neg = list(neg_dict.keys())[:10000]

    f, axs = plt.subplots(2,2,figsize=(15,15))
    plt.subplot(2,1,1)
    w_c = WordCloud(width=2400,height=1600,colormap='summer',background_color='white').generate(' '.join(top_10_pos))
    plt.title('Most Positive Words',fontsize=25,color='Green')
    plt.imshow(w_c)
    plt.axis('off')
    plt.subplot(2,1,2)
    w_c = WordCloud(width=2400,height=1600,colormap='autumn',background_color='white').generate(' '.join(top_10_neg))
    plt.title('Most Negative Words',fontsize=19,color='red')
    plt.imshow(w_c)
    plt.axis('off')
    plt.show()