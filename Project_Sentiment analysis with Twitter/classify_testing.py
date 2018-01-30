"""
classify_testing.py
"""
import sys
import pickle
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
import pandas as pd

def main():
    """
    f = open('./data/tweets.txt', 'rb')
    tweets = pickle.load(f)
    f2 = open('./data/users.txt', 'rb')
    users = pickle.load(f2)
    user_list = sorted([u['screen_name'] for u in users])
    """
    test_data_pd = pd.read_csv('./data/trainData.csv', encoding = "utf8")
    test_labels = test_data_pd['polarity'].tolist()
    test_data = test_data_pd['text'].tolist()

    train_data_pd = pd.read_csv('./data/trainData.csv', encoding = "utf8")
    train_labels = train_data_pd['polarity'].tolist()
    train_data = train_data_pd['text'].tolist()
    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)


    # Perform classification with SVM, kernel=rbf
    classifier_rbf = svm.SVC()
    t0 = time.time()
    classifier_rbf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1


    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1


    # Perform classification with SVM, kernel=poly, degree=3
    classifier_poly = svm.SVC(kernel='poly')
    t0 = time.time()
    classifier_poly.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_poly = classifier_poly.predict(test_vectors)
    t2 = time.time()
    time_poly_train = t1-t0
    time_poly_predict = t2-t1

    print("Results for SVC(kernel=rbf)")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))

    print("Results for SVC(kernel=linear)")
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    print(classification_report(test_labels, prediction_linear))

    print("Results for SVC(kernel=poly)")
    print("Training time: %fs; Prediction time: %fs" % (time_poly_train, time_poly_predict))
    print(classification_report(test_labels, prediction_poly))
    
if __name__ == '__main__':
    main()