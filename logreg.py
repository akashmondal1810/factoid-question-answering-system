from csv import DictReader, DictWriter
import numpy as np
import sklearn
from numpy import array
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize.api import TokenizerI
from nltk.tokenize.util import regexp_span_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model

kTARGET_FIELD = 'correctAnswer'
kTEXT_FIELD = 'question'
kID_FIELD = 'id'
kA = 'answerA'
kB = 'answerB'
kC = 'answerC'
kD = 'answerD'

english_stemmer = PorterStemmer()

# class TryTokenizer(object):
#     def __init__(self):
#         self.tokenizer = RegexpTokenizer(r'\w+')
#     def __call__(self, doc):
#         return [english_stemmer.stem(token) for token in doc]

class Featurizer:
    def __init__(self):
        # Count vector is giving better accuracy then tfidf
        self.vectorizer = CountVectorizer()
        # self.vectorizer = TfidfVectorizer(lowercase=False)

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

if __name__ == "__main__":

    train = list(DictReader(open("data/sci_train.csv", 'r')))
    # test = list(DictReader(open("data/sci_test.csv", 'r')))
    # testing and training on 10 % of the data
    test = train[-int(len(train)*0.1):]
    train = train[:-int(len(train)*0.1)]
    feat = Featurizer()

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    test_labels = []
    for line in test:
        if not line[kTARGET_FIELD] in test_labels:
            test_labels.append(line[kTARGET_FIELD])


    print("Label set: %s" % str(labels))


   
    x_train = feat.train_feature(x[kTEXT_FIELD] + ' ' + x[kTARGET_FIELD] + ' ' + x[kA] + ' ' + x[kB] + ' ' + x[kC] + ' ' + x[kD] for x in train)
    x_test = feat.test_feature(x[kTEXT_FIELD] + ' ' + x[kTARGET_FIELD] + ' ' + x[kA] + ' ' + x[kB] + ' ' + x[kC] + ' ' + x[kD] for x in test)

    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))
    y_test = array(list(test_labels.index(x[kTARGET_FIELD])
                         for x in test))

    # print(len(train), len(y_train))
    # print(set(y_train))

    lr = LogisticRegression(C=1,
                                penalty='l1',
                                fit_intercept=True)
    lr.fit(x_train, y_train)

    predictions = lr.predict(x_test);

    accuracy = 100.0 * sklearn.metrics.accuracy_score(y_train, lr.predict(x_train));
    print("Log Reg training :")
    print(accuracy)

    accuracy_val = 100.0 * sklearn.metrics.accuracy_score(y_test, lr.predict(x_test));
    print("Log Reg Validation:")
    print(accuracy_val)

    
