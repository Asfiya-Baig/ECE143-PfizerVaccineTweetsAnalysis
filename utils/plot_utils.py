import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def plot_count(feature, title, x_title, y_title, dataframe, num_show=10, horizontal=False):
    """plots bar graph based on COUNT of a feature. Can plot it horizontally or vertically

    Args:
        feature ([str]): feature whose COUNTS will be added up
        title ([str]): title of plot
        x_title ([str]): title of x axis
        y_title ([str]): title of y axis
        dataframe ([datframe]): input dataframe
        num_show (int, optional): number of entries to show in decreasing order of counts. Defaults to 10.
        horizontal (bool, optional): boolean indicating if plot should be horizontal. Defaults to False.
    """   
    # validate inputs
    assert isinstance(feature, str), "feature must be string"
    assert isinstance(title, str), "title must be string"
    assert isinstance(x_title, str), "x title must be string"
    assert isinstance(y_title, str), "y title must be string"
    assert isinstance(dataframe, pd.DataFrame), "title must be dataframe"
    assert isinstance(num_show, int), "num_show must be integer"
    assert isinstance(horizontal, bool), "horizontal must be boolean"

    # create a figure
    plt.figure(figsize=(10,12))

    # plot
    if horizontal:
        sns.barplot(dataframe[feature].value_counts().values[0:num_show], dataframe[feature].value_counts().index[0:num_show], color='orange')
        plt.xlabel(y_title,fontsize=14)
        plt.ylabel(x_title, fontsize=14)
    else:
        sns.barplot(dataframe[feature].value_counts().index[0:num_show], dataframe[feature].value_counts().values[0:num_show])
        plt.xlabel(x_title,fontsize=14)
        plt.xticks(rotation=45)
        plt.ylabel(y_title, fontsize=14)

    # show plot
    plt.title(title, fontsize=14)
    plt.show()
    
def plot_by_feature(feature, title, x_title, y_title, dataframe, num_show=10, horizontal=False):
    """plots bar graph based on VALUE of a feature. Can plot it horizontally or vertically

    Args:
        feature ([str]): feature whose VALUE will used
        title ([str]): title of plot
        x_title ([str]): title of x axis
        y_title ([str]): title of y axis
        dataframe ([datframe]): input dataframe
        num_show (int, optional): number of entries to show in decreasing order of counts. Defaults to 10.
        horizontal (bool, optional): boolean indicating if plot should be horizontal. Defaults to False.
    """      
    # validate inputs
    assert isinstance(feature, str), "feature must be string"
    assert isinstance(title, str), "title must be string"
    assert isinstance(x_title, str), "x title must be string"
    assert isinstance(y_title, str), "y title must be string"
    assert isinstance(dataframe, pd.DataFrame), "title must be dataframe"
    assert isinstance(num_show, int), "num_show must be integer"
    assert isinstance(horizontal, bool), "horizontal must be boolean"

    # create a figure
    plt.figure(figsize=(10,12))

    # plot
    dataframe_copy = dataframe.drop_duplicates(subset=["user_name"]).sort_values([feature], ascending=False)
    if horizontal:
        sns.barplot(dataframe_copy[feature][:num_show], dataframe_copy['user_name'][:num_show], palette= ('red', 'blue', 'blue', 'blue', 'blue', 'yellow', 'blue', 'blue', 'orange', 'green' ))
        plt.xlabel(y_title,fontsize=14)
        plt.ylabel(x_title, fontsize=14)
    else:
        sns.barplot(dataframe_copy['user_name'][:num_show], dataframe_copy[feature][:num_show])
        plt.xlabel(x_title,fontsize=14)
        plt.xticks(rotation=45)
        plt.ylabel(y_title, fontsize=14)

    # show plot
    plt.title(title, fontsize=14)
    plt.show()
    
def plot_over_time(dataframe, feature):
    """plot of tweets over time

    Args:
        dataframe ([datframe]): input dataframe
        feature ([str]): feature
    """   
    # validate inputs
    assert isinstance(feature, str), "feature must be string"
    assert isinstance(dataframe, pd.DataFrame), "title must be dataframe"
     
    # extract the feature (date)
    dataframe[feature] =  pd.to_datetime(dataframe[feature])

    # count number of rows with the same date (tweets per day)
    counts_per_date = dataframe[feature].dt.date.value_counts()
    counts_per_date = counts_per_date.sort_index()

    # bar plot of tweets over time
    plt.figure(figsize=(20,8))
    sns.barplot(counts_per_date.index, counts_per_date.values, alpha=0.8, color='orange')
    plt.xticks(rotation='vertical')
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Number of tweets', fontsize=12)
    plt.title("Tweets over time")
    plt.show()