import pickle
from nltk.classify import ClassifierI
from statistics import mode
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def labels(self):
        pass

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = float(choice_votes) / float(len(votes))
        return conf


documents_f = open("G:/PythonProjects/pickled_algos/documents1.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("G:/PythonProjects/pickled_algos/word_features5k1.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

open_file = open("G:/PythonProjects/pickled_algos/LogisticRegression_classifier5k1.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("G:/PythonProjects/pickled_algos/LinearSVC_classifier5k1.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("G:/PythonProjects/pickled_algos/DT_classifier5k1.pickle", "rb")
DT_classifier = pickle.load(open_file)
open_file.close()

open_file = open("G:/PythonProjects/pickled_algos/RF_classifier5k1.pickle", "rb")
RF_classifier = pickle.load(open_file)
open_file.close()

open_file = open("G:/PythonProjects/pickled_algos/SGDC_classifier5k1.pickle", "rb")
SGDC_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(
        LinearSVC_classifier,
        LogisticRegression_classifier,
        DT_classifier,
        RF_classifier,
        SGDC_classifier)


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

df = pd.read_csv('train.tsv', sep='\t')
print("Reading complete")
df.set_index('PhraseId', inplace = True)
df = df[['Phrase','Sentiment']]
df = df.loc[:10000, :]

tt = 0
tf = 0
ft = 0
ff = 0
conft = 0.0
conff = 0.0
for p,s in zip(df['Phrase'],df['Sentiment']):
    try:
        sent,conf = sentiment(p)
        if s>=2:
            if sent == 'happy =) ':
                tt += 1
                conft += conf
            else:
                tf += 1
                conff += conf
        else:
            if sent == 'sad =( ':
                ff += 1
                conft += conf
            else:
                ft += 1
                conff += conf
    except BaseException, e:
        print e

print len(df[['Phrase']])
print tt, tf
print ft, ff
print tt+ff
print conft/tt+ff
print conff/tf+ft
