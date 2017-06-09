import nltk
import random
# from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import app
import threading


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
        conf = choice_votes / len(votes)
        return conf

def train():
    df = pd.read_csv('train.tsv', sep='\t')
    #print("Reading complete")
    df.set_index('PhraseId', inplace = True)
    df = df[['Phrase','Sentiment']]
    # move this up here
    all_words = []
    documents = []

    #  j is adject, r is adverb, and v is verb
    # allowed_word_types = ["J","R","V"]
    allowed_word_types = ["J", "R"]
    #a = "."
    #print 'Using all words'
    prev_len = 0
    i = 0
    all_words = []
    for p,s in zip(df['Phrase'],df['Sentiment']):
        #print p, s
        try:
            words = word_tokenize(p)
            #print words
            pos = nltk.pos_tag(words)
            #print pos
            for w in pos:
                if w[1][0] in allowed_word_types:
                    all_words.append(w[0].lower())
            curr_len = len(all_words)
            if prev_len < curr_len:
                if s>=2:
                    documents.append((p, "happy =) "))
                else:
                    documents.append((p, "sad =( "))
            prev_len = curr_len
    #        if i%100 == 0:
    #            print 'next 100'
    #            print i
    #            print len(all_words)
    #            print len(documents)
    #
    #        i += 1
        except BaseException, e:
            print e


    save_documents = open("pickled_algos/documents1.pickle", "wb")
    pickle.dump(documents, save_documents)
    save_documents.close()

    #print("Docs pickled")

    all_words = nltk.FreqDist(all_words)

    word_features = list(all_words.keys())[:5000]

    save_word_features = open("pickled_algos/word_features5k1.pickle", "wb")
    pickle.dump(word_features, save_word_features)
    save_word_features.close()

    #print("word features pickled")

    #documents_f = open("G:/PythonProjects/pickled_algos/documents1.pickle", "rb")
    #documents = pickle.load(documents_f)
    #documents_f.close()

    #print "Pickled docs opened"

    #word_features5k_f = open("G:/PythonProjects/pickled_algos/word_features5k1.pickle", "rb")
    #word_features = pickle.load(word_features5k_f)
    #word_features5k_f.close()

    #print "Pickled words opened"


    def find_features(document):
        words = word_tokenize(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)

        return features


    featuresets = []
    #print "Creating featuresets"

    #print len(documents)
    y = 0
    for (rev, category) in documents:
        y += 1
        if y == 2561 or y == 3761 or y == 10367:
            continue
        else:
            featuresets.append((find_features(rev), category))

    #print "Featuresets created"

    random.shuffle(featuresets)
    featuresets = featuresets[:60000]
    #print(len(featuresets))

    testing_set = featuresets[int(len(featuresets) * 0.7):]
    training_set = featuresets[:int(len(featuresets) * 0.7)]

    #print("Train and test ssplit done.")

    #classifier = nltk.NaiveBayesClassifier.train(training_set)
    #print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
    #classifier.show_most_informative_features(15)
    #save_classifier = open("G:/PythonProjects/pickled_algos/originalnaivebayes5k1.pickle", "wb")
    #pickle.dump(classifier, save_classifier)
    #save_classifier.close()

    #print("Done")

    #print("Hope mnb works")

    #MNB_classifier = SklearnClassifier(MultinomialNB())
    #MNB_classifier.train(training_set)
    #print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)

    #save_classifier = open("G:/PythonProjects/pickled_algos/MNB_classifier5k1.pickle", "wb")
    #pickle.dump(MNB_classifier, save_classifier)
    #save_classifier.close()

    #print("Done")

    #BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    #BernoulliNB_classifier.train(training_set)
    #print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

    #save_classifier = open("G:/PythonProjects/pickled_algos/BernoulliNB_classifier5k1.pickle", "wb")
    #pickle.dump(BernoulliNB_classifier, save_classifier)
    #save_classifier.close()
    #print("Done")

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    #print("LogisticRegression_classifier accuracy percent:",(nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

    save_classifier = open("pickled_algos/LogisticRegression_classifier5k1.pickle", "wb")
    pickle.dump(LogisticRegression_classifier, save_classifier)
    save_classifier.close()

    #print("Done")

    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    #print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

    save_classifier = open("pickled_algos/LinearSVC_classifier5k1.pickle", "wb")
    pickle.dump(LinearSVC_classifier, save_classifier)
    save_classifier.close()

    #print("Done")

    SGDC_classifier = SklearnClassifier(SGDClassifier())
    SGDC_classifier.train(training_set)
    #print("SGDClassifier accuracy percent:", nltk.classify.accuracy(SGDC_classifier, testing_set) * 100)

    save_classifier = open("pickled_algos/SGDC_classifier5k1.pickle", "wb")
    pickle.dump(SGDC_classifier, save_classifier)
    save_classifier.close()

    #print("Done")

    #Ada_classifier = SklearnClassifier(AdaBoostClassifier())
    #Ada_classifier.train(training_set)
    #print("AdaClassifier accuracy percent:", nltk.classify.accuracy(Ada_classifier, testing_set) * 100)

    #save_classifier = open("G:/PythonProjects/pickled_algos/Ada_classifier5k1.pickle", "wb")
    #pickle.dump(Ada_classifier, save_classifier)
    #save_classifier.close()

    #print("Done")

    #DT_classifier = SklearnClassifier(DecisionTreeClassifier())
    #DT_classifier.train(training_set)
    #print("DTClassifier accuracy percent:", nltk.classify.accuracy(DT_classifier, testing_set) * 100)

    #save_classifier = open("pickled_algos/DT_classifier5k1.pickle", "wb")
    #pickle.dump(DT_classifier, save_classifier)
    #save_classifier.close()

    #print("Done")

    #RF_classifier = SklearnClassifier(RandomForestClassifier())
    #RF_classifier.train(training_set)
    #print("RFClassifier accuracy percent:", nltk.classify.accuracy(RF_classifier, testing_set) * 100)

    #save_classifier = open("G:/PythonProjects/pickled_algos/RF_classifier5k1.pickle", "wb")
    #pickle.dump(RF_classifier, save_classifier)
    #save_classifier.close()

    #print("Done")

class ModelThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        train()
        app.reload()
        return

if __name__ == '__main__':
    train()
