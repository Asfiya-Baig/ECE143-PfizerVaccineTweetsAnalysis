import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
import gensim
import nltk

def remove_stopwords(input_text):
    '''
    Function to remove English stopwords from a Pandas Series.
    
    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series 
    '''
    # nltk.download('stopwords')
    stopwords_list = stopwords.words('english')
    # Some words which might indicate a certain sentiment are kept via a whitelist
    whitelist = ["n't", "not", "no"]
    words = input_text.split() 
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
    return " ".join(clean_words) 
    
def remove_mentions(input_text):
    '''
    Function to remove mentions, preceded by @, in a Pandas Series
    
    Parameters:
        input_text : text to clean
    Output:
        cleaned Pandas Series 
    '''
    return re.sub(r'@\w+', '', input_text)

def filter_word(sentence):
  '''
  description: this function will filter a sentence
  '''
  return [word.lower() for word in nltk.word_tokenize(sentence) if word.isalnum]

def tokenize(text):
  '''
  description: this function will tockenize the text
  '''
  return [filter_word(sentence) for sentence in nltk.sent_tokenize(text)]

def plot_WE(x, vocabulary, threshold, word2vec_model, my_pca):
  '''
  description: this function will find the list of words whose samilarity with 
  the word x is higher than the threshold
  input:
  x: the target word
  vocabulary: the list of words in the vocabulary
  threshold: the similarity threshold float between 0 and 1
  word2vec_model: the Word2Vec model
  my_pca: the numpy 2d array stroing the PCA result
  '''
  assert isinstance(x, str)
  assert isinstance(vocabulary, list)
  assert all(isinstance(element, str) for element in vocabulary)
  assert isinstance(threshold, float)
  assert threshold >=0 and threshold <= 1
  assert isinstance(word2vec_model, gensim.models.Word2Vec)
  assert isinstance(my_pca, np.ndarray)
  
  ind = vocabulary.index(x)
  neighbor = []
  for i in range(len(vocabulary)):
    if word2vec_model.wv.similarity(x, vocabulary[i]) > threshold:
      neighbor.append(i)

  # plot the data
  plt.scatter(my_pca[neighbor, 0], my_pca[neighbor, 1], alpha=1, color='lightblue')
  # annotate plot
  for i, word in enumerate(vocabulary):
    # if len(word) < 2:
    #   break
    if i in neighbor:
      if i == ind:
        plt.annotate(x, xy=(my_pca[ind, 0], my_pca[ind, 1]), color='red', ha='center')
      else:
        plt.annotate(word, xy=(my_pca[i, 0], my_pca[i, 1]), color='black', size=8, ha='center')
    else:
      continue

  plt.show();
  return neighbor