#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 14:04:48 2020

@author: qianyingliao
"""
import os
import pandas as pd
import re
import numpy as np
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV

train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/lib/publicdata/imdb_te.csv" # test data for grade evaluation

def getStopWords():
    stopwords = list()
    stopwords_string = ""
    with open("stopwords.en.txt", 'r') as f:
        stopwords_string = f.read()
    stopwords = stopwords_string.split("\n")
    stopwords.extend(["dont", "aint", "arent", "couldnt", "didnt", "doesnt", "hadnt"])
    stopwords.extend(["hasnt", "havent", "isnt", "mightnt", "mustnt", "neednt", "shouldnt"])
    stopwords.extend(["mightnt", "mustnt", "neednt", "shouldnt", "wasnt", "neednt", "werent"])
    stopwords.extend(["wont", "wouldnt","hence", "thus", "whilst","thereafter", "hereafter"])
    stopwords.extend(["regardless", "nevertheless","hereby", "thus", "nonetheless", "still"])
    return  stopwords

def preprocess(data, stopwords):
    # Remove all punctuations and keep capital letters and space
    data = re.sub(r'[^a-z ]', '', data)
    # Remove words whose length is less than 3
    data = re.sub(r'\b\w{1,2}\b', '', data)
    #remove the br tag and stop words
    for stopword in stopwords:
        data = re.sub(' '+stopword+' ', ' ', data)
    # Remove long words
    #data = re.sub(r'\b\w{10,}\b', '', data)
    # remove extra space
    data = ' '.join(data.split())
    return data

def readTrainData(inpath, polarity, train_data, output_data):
    entries = os.listdir(inpath)
    stopwords = getStopWords()
    alldata = ""
    for entry in entries:
        with open(os.path.join(inpath,entry), 'r') as f:
            data = f.read().lower()
            output_data.append([data, polarity])
            data = preprocess(data, stopwords)
            alldata+=(data+" ")
            train_data.append([data, polarity])
    return alldata

def readTestData( ):
    test_data = pd.read_csv(test_path,encoding = "ISO-8859-1")
    test_data = pd.DataFrame(test_data['text'])
    stopwords = getStopWords()
    texts = list()
    for text in test_data['text']:
        text = text.lower()
        text = preprocess(text, stopwords)
        texts.append(text)
    return texts

def compileAndStoreData(outpath="./", name="imdb_tr.csv"):
    '''Implement this module to extract
    and combine text files under train_path directory into 
    imdb_tr.csv. Each text file in train_path should be stored 
    as a row in imdb_tr.csv. And imdb_tr.csv should have two 
    columns, "text" and label'''
    train_data = list()
    train_data.append(["text", "polarity"])
    data = list()
    data.append(["text", "polarity"])
    alltext = readTrainData(os.path.join(train_path,"pos"), 1, train_data, data)
    alltext += readTrainData(os.path.join(train_path,"neg"), 0, train_data, data)
    data = pd.DataFrame(data[1:],columns=data[0])
    train_data = pd.DataFrame(train_data[1:],columns=train_data[0])
    data.to_csv(os.path.join(outpath,"imdb_tr.csv"), encoding='utf-8', index=True)
    
    texts = readTestData()
    test_data = pd.DataFrame()
    test_data['text'] = pd.Series(texts)
    return train_data, alltext, test_data

def PerformNGram(data, ngram1, ngram2):
    count_vect = CountVectorizer(ngram_range=(ngram1, ngram2), max_df = 0.50, min_df = 0.001)
    X_counts = count_vect.fit_transform(data['text'])
    volcabularies = count_vect.get_feature_names()
    return X_counts, volcabularies

def PerformNGramWithVolcabularies(data, ngram1, ngram2, volcabularies):
    count_vect = CountVectorizer(ngram_range=(ngram1, ngram2), vocabulary=volcabularies)
    X_counts = count_vect.fit_transform(data['text'])
    return X_counts

def performNGramTFIDF(data, ngram1, ngram2):
    count_vect = TfidfVectorizer(ngram_range=(ngram1, ngram2), sublinear_tf=True, max_df = 0.50, min_df = 0.001)
    X_counts = count_vect.fit_transform(data['text'])
    volcabularies = count_vect.get_feature_names()
    return X_counts, volcabularies

def performNGramTFIDFWithVolcabularies(data, ngram1, ngram2, volcabularies):
    count_vect = TfidfVectorizer(ngram_range=(ngram1, ngram2), vocabulary=volcabularies, sublinear_tf=True)
    X_counts = count_vect.fit_transform(data['text'])
    return X_counts

def SGD(X, Y):
    parameters = {'loss':["hinge"], "penalty":["l1"]}
    model = linear_model.SGDClassifier()
    clf = GridSearchCV(model, parameters, cv = 3, scoring="accuracy")  
    clf.fit(X, Y)
    #print("SGD Socres:", clf.cv_results_)
    return clf

def PredictWithSGD(X, CLF, FILENAME):
    Y = CLF.predict(X)
    Y = pd.DataFrame(Y)
    Y.to_csv(FILENAME, encoding='utf-8', index=False, header=None)

def main():
    train_data, alltext, test_data = compileAndStoreData()
    Y = train_data['polarity']
    Y = np.asarray(Y)

    X_Unigram, volcabularies1 = PerformNGram(train_data, 1, 1)
    X_Bigram, pair_volcabularies1 = PerformNGram(train_data, 1, 2)
    X_Unigram_TFIDF, volcabularies2 = performNGramTFIDF(train_data, 1, 1)
    X_Bigram_TFIDF, pair_volcabularies2 = performNGramTFIDF(train_data, 1, 2)

    CLF1 = SGD(X_Unigram, Y)
    CLF2 = SGD(X_Bigram, Y)
    CLF3 = SGD(X_Unigram_TFIDF, Y)
    CLF4 = SGD(X_Bigram_TFIDF, Y)

    X_Unigram_Test = PerformNGramWithVolcabularies(test_data, 1, 1, volcabularies1)
    X_Bigram_Test = PerformNGramWithVolcabularies(test_data, 1, 2, pair_volcabularies1)
    X_Unigram_TFIDF_Test = performNGramTFIDFWithVolcabularies(test_data, 1, 1, volcabularies2)
    X_Bigram_TFIDF_Test = performNGramTFIDFWithVolcabularies(test_data, 1, 2, pair_volcabularies2)

    PredictWithSGD(X_Unigram_Test, CLF1, "unigram.output.txt")
    PredictWithSGD(X_Bigram_Test, CLF2, "bigram.output.txt")
    PredictWithSGD(X_Unigram_TFIDF_Test, CLF3, "unigramtfidf.output.txt")
    PredictWithSGD(X_Bigram_TFIDF_Test, CLF4, "bigramtfidf.output.txt")

if __name__ == '__main__':

    main()
#main()