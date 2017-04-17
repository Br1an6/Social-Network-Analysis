# coding: utf-8

# # Project 3:  Recommendation systems
#
# Here I implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    temp_list = []
    for gen in movies['genres']:
        temp_list.append(tokenize_string(gen))

    movies['tokens'] = pd.Series(temp_list) 
    return movies
    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    temp_list = []
    list_of_matrix = []
    vocab = {}
    for gen in movies['tokens']:
        for t in gen:
            if t not in vocab:
                vocab[t] = 1
            else:
                vocab[t] += 1
    t_list = []
    t_list.append([ key for key, val in sorted(vocab.items())])
    t_list = t_list[0]
    for gen in movies['tokens']:
        temp_w_list = [0 for i in range(len(t_list)) ]
        for t in gen:
            idf = math.log10(len(movies.index)/vocab[t])
            max_k = Counter(gen).most_common(1)
            tf =  gen.count(t) / max_k[0][1]
            w = tf * idf
            temp_w_list[t_list.index(t)] = w
        X = csr_matrix([temp_w_list])
        list_of_matrix.append(X)
    movies['features'] = pd.Series(list_of_matrix) 
    num = 0
    for key, val in sorted(vocab.items()):
        vocab[key] = num
        num += 1
    return movies, vocab


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    norm_a = np.sqrt((a.toarray()**2).sum(axis=1))[0]
    norm_b = np.sqrt((b.toarray()**2).sum(axis=1))[0]
    return (a.dot(b.transpose()) / (norm_a * norm_b))[0, 0]


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    ret_list = []
    for i in range(len(ratings_test.index)):
        uid = ratings_test.userId.iloc[i]
        mid = ratings_test.movieId.iloc[i]
        temp_df = ratings_train[ratings_train.userId==uid]
        x1 = movies.features[movies.movieId==mid].values[0]
        list_cs = []
        weight_a = 0.
        weight_for_all_zero = []
        for n in range(len(temp_df.index)):
            mid2 = temp_df.movieId.iloc[n]
            x2 = movies.features[movies.movieId==mid2].values[0]
            cs = cosine_sim(x1, x2)
            if cs >= 0:
                list_cs.append(cs)
                weight_a += temp_df.rating.iloc[n] * cs
                weight_for_all_zero.append(temp_df.rating.iloc[n]) 
        if sum(list_cs) != 0:
            weight_a /= sum(list_cs)	
        else:
            weight_a = sum(weight_for_all_zero) / len(weight_for_all_zero)
        ret_list.append(weight_a)

    return np.array(ret_list)

def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
