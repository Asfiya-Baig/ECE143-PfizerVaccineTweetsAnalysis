import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def plot_count(feature, title, x_title, y_title, df, num_show=10, horizontal=False):
    plt.figure(figsize=(10,12))
    if horizontal:
        sns.barplot(df[feature].value_counts().values[0:num_show], df[feature].value_counts().index[0:num_show], color='orange')
        plt.xlabel(y_title,fontsize=14)
        plt.ylabel(x_title, fontsize=14)
    else:
        sns.barplot(df[feature].value_counts().index[0:num_show], df[feature].value_counts().values[0:num_show])
        plt.xlabel(x_title,fontsize=14)
        plt.xticks(rotation=45)
        plt.ylabel(y_title, fontsize=14)

    plt.title(title, fontsize=14)
    plt.show()
    
def plot_by_feature(feature, title, x_title, y_title, df, num_show=10, horizontal=False):
    plt.figure(figsize=(10,12))

    df_copy = df.drop_duplicates(subset=["user_name"]).sort_values([feature], ascending=False)
    if horizontal:
        sns.barplot(df_copy[feature][:num_show], df_copy['user_name'][:num_show], palette= ('red', 'blue', 'blue', 'blue', 'blue', 'yellow', 'blue', 'blue', 'orange', 'green' ))
        plt.xlabel(y_title,fontsize=14)
        plt.ylabel(x_title, fontsize=14)
    else:
        sns.barplot(df_copy['user_name'][:num_show], df_copy[feature][:num_show])
        plt.xlabel(x_title,fontsize=14)
        plt.xticks(rotation=45)
        plt.ylabel(y_title, fontsize=14)

    plt.title(title, fontsize=14)
    plt.show()
    
def plot_over_time(dataset, feature):
    dataset[feature] =  pd.to_datetime(dataset[feature])
    counts_per_date = dataset[feature].dt.date.value_counts()
    counts_per_date = counts_per_date.sort_index()
    plt.figure(figsize=(20,8))
    sns.barplot(counts_per_date.index, counts_per_date.values, alpha=0.8, color='orange')
    plt.xticks(rotation='vertical')
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Number of tweets', fontsize=12)
    plt.title("Tweets over time")
    plt.show()