from flask import Flask, flash, render_template, request
import pickle
from nltk.classify import ClassifierI
from statistics import mode
import nltk
import new_data_train
nltk.download('punkt')

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


documents_f = open("pickled_algos/documents1.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()

word_features5k_f = open("pickled_algos/word_features5k1.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()

voted_classifier = None

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

def reload():
    open_file = open("pickled_algos/LogisticRegression_classifier5k1.pickle", "rb")
    LogisticRegression_classifier = pickle.load(open_file)
    open_file.close()

    open_file = open("pickled_algos/LinearSVC_classifier5k1.pickle", "rb")
    LinearSVC_classifier = pickle.load(open_file)
    open_file.close()

    #open_file = open("pickled_algos/DT_classifier5k1.pickle", "rb")
    #DT_classifier = pickle.load(open_file)
    #open_file.close()

    #open_file = open("pickled_algos/RF_classifier5k1.pickle", "rb")
    #RF_classifier = pickle.load(open_file)
    #open_file.close()

    open_file = open("pickled_algos/SGDC_classifier5k1.pickle", "rb")
    SGDC_classifier = pickle.load(open_file)
    open_file.close()

    global voted_classifier
    voted_classifier = VoteClassifier(LinearSVC_classifier,LogisticRegression_classifier,SGDC_classifier)
    print 'reload complete'


def sentiment(text):
    feats = find_features(text)
    print voted_classifier
    return voted_classifier.classify(feats), voted_classifier.confidence(feats)

reload()
app = Flask(__name__)


@app.route('/', methods = ['GET','POST'])
def index():
    ret_obj = ''
    sent_obj = ''
    PhraseId = 156061
    SentenceId = 8545
    PhraseId1 = 156061
    SentenceId1 = 8545
    if request.method == 'POST':
        if request.form['submit']=='Check':
            print 'hi'
            sent, conf = sentiment(str(request.form['in_text']))
            print conf
            ret_obj = str(request.form['in_text'])
            if str(sent) == 'happy =) ':
                if conf ==1:
                    sent_obj = 'positive'
                else:
                    sent_obj = 'neutral'
            else:
                if conf == 1:
                    sent_obj = 'negative'
                else:
                    sent_obj = 'neutral'
            return render_template('index.html', ret_obj = ret_obj, sent_obj = sent_obj)
        if request.form['submit']=='Wrong':
            if str(request.form['sent_text']) == 'positive':
                sent_scr = 2
            if str(request.form['sent_text']) == 'negative':
                sent_scr = 3
            f = open('train.tsv','a')
            f.write('\n'+str(PhraseId)+'\t'+str(SentenceId)+'\t'+str(request.form['in_text'])+'\t'+str(sent_scr))
            PhraseId += 1
            SentenceId += 1
            f.close()
            if (PhraseId - PhraseId1)%1000 == 0:
                retrain_thread = new_data_train.ModelThread()
                retrain_thread.start()
            return render_template('index.html', ret_obj = ret_obj, sent_obj = sent_obj)
    if request.method == 'GET':
        return render_template('index.html', ret_obj = ret_obj, sent_obj = sent_obj)

if __name__=='__main__':
    app.run()
