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
from sklearn.utils import shuffle

kTARGET_FIELD = 'correctAnswer'
kTEXT_FIELD = 'question'
kID_FIELD = 'id'
kA = 'answerA'
kB = 'answerB'
kC = 'answerC'
kD = 'answerD'

class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer()#lowercase=False)
                                            #analyzer = 'word')
        # self.vectorizer = TfidfVectorizer(lowercase=False)

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

if __name__ == "__main__":

    n = 0.7  # train validation split

    train = list(DictReader(open("data/sci_train.csv", 'r')))
    # test = list(DictReader(open("data/sci_test.csv", 'r')))
    # sample = list(DictReader(open("data/sci_sample.csv", 'r')))
    train = shuffle(train);
    print("Total length: ", len(train))
    test = train[-int(len(train)*(1.0 - n)):]
    train = train[:int(len(train)*n)]

    # print("Length of train: ", len(train), " test: ", len(test))
    # print("% train vs test:", len(train)/(len(train) + len(test)))
    feat = Featurizer()

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    print(len(labels))


    # test_labels = []
    # for line in test:
    #     if not line[kTARGET_FIELD] in test_labels:
    #         test_labels.append(line[kTARGET_FIELD])
   
       
    lablelist =["A","B","C","D"]

    x_train = feat.train_feature(x[kTEXT_FIELD] + ' ' + x["answer" + i]
                                 for i in lablelist for x in train)
    print("new total x_train length",x_train.shape[0])


    
    x_test = feat.test_feature(x[kTEXT_FIELD] + ' ' + x["answer" + i] 
                                for i in lablelist for x in test)
    print("new total x_test length",x_test.shape[0])


    # Assigining 1 to correct answer and 0 to wrong answer
    

    # y_train = array(list(labels.index(x[kTARGET_FIELD])
    #                      for x in train))

    y_train =array(list(1 if x["answer" + x[kTARGET_FIELD]] == x["answer" + i] else 0 
                            for i in lablelist for x in train))    
    print("new total y_train length",y_train.shape[0])

    y_test = array(list(1 if x["answer" + x[kTARGET_FIELD]] == x["answer" + i] else 0 
                            for i in lablelist for x in test))
    
    # y_test = array(list(1 if x[kTARGET_FIELD] else 0  for x in sample))
    # print("new total y_test length",y_test.shape[0])

    print(y_train)
    print("Training started...")

    lr = LogisticRegression(C=1, penalty='l2', fit_intercept=True)
    lr.fit(x_train, y_train)

    # predictions = lr.predict(x_test);

    accuracy = 100.0 * sklearn.metrics.accuracy_score(y_train, lr.predict(x_train));
    print("Log Reg training :")
    print(accuracy)

    accuracy_val = 100.0 * sklearn.metrics.accuracy_score(y_test, lr.predict(x_test));
    print("Log Reg Validation:")
    print(accuracy_val)


    predictions = lr.predict(x_test)
    o = DictWriter(open("testpredictions.csv", 'w'), ["id", "correctAnswer"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'correctAnswer': labels[pp]}
        o.writerow(d)

    
        
