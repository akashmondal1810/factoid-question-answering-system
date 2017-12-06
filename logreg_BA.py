from csv import DictReader, DictWriter
import numpy as np
import sklearn
from numpy import array
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from word2vec_gen import *
from gensim.models import Word2Vec

import os.path

kTARGET_FIELD = 'correctAnswer'
kTEXT_FIELD = 'question'
kID_FIELD = 'id'
kA = 'answerA'
kB = 'answerB'
kC = 'answerC'
kD = 'answerD'


class lr_classifier():

    def __init__(self, w2v_model, c=1, pen='l2'):
        self.lr = LogisticRegression(C=c, penalty=pen)
        self.model = w2v_model


    def form_vector(self, question, answer, sv):
        return np.hstack((sv(question), sv(answer)))
        

    def form_all_vectors(self, data, sv):
        label_list = ["A", "B", "C", "D"]
        x_train = []
        y_train = []
        for x in data:
            for label in label_list:
                alabel = 'answer' + label
                v = self.form_vector(x[kTEXT_FIELD], x[alabel], sv)
                x_train.append(v)

                if label == x['correctAnswer']:
                    y_train.append(1)
                else:
                    y_train.append(0)

        return x_train, y_train


    def train(self, x_train, y_train):
        self.lr.fit(x_train, y_train)

    def predict(self, x_test):
        predictions = []

        true_ind = np.argmax(self.lr.classes_)
        
        for i in range(0,len(x_test),4):
            X = x_test[i:i+4]
            p = self.lr.predict_proba(X)
            p = p[:,[true_ind]]

            predictions.append(np.argmax(p))

        return predictions
        

    def test(self, x_test, y_test):
        success = 0
        failure = 0

        predictions = self.predict(x_test)
        
        for i in range(0,test_len):
            ans = predictions[i]

            if y_test[4*i+ans] == 1:
                success += 1
            else:
                failure += 1

        return [success, failure]



if __name__ == "__main__":

    n = 0.8  # train validation split

    train = list(DictReader(open("data/filtered_train.csv", 'r')))
    extra_train = list(DictReader(open("data/extra_train.csv", 'r')))
    # test = list(DictReader(open("data/sci_test.csv", 'r')))
    # sample = list(DictReader(open("data/sci_sample.csv", 'r')))
    train = shuffle(train)
    extra_train = shuffle(extra_train)

    
    test = train[-int(len(train) * (1.0 - n)):]
    train = train[:int(len(train) * n)]

    train_len = len(train)
    test_len = len(test)
    
    print("Train size: ", train_len)
    print("Test size: ", test_len)

    print("Training word2vec...")

    filepath = "data/word2vec.dat"
    if os.path.exists(filepath):
        model = w2v(load_data=True, old_data=filepath)
        print("Existing data loaded.")
    else:
        model = w2v(data_path='data/wiki_corpus.txt')
        model.model.save(filepath)
        print("New word2vec data created.")


    lr = lr_classifier(model)

    print("Forming vectors...")
    x_train, y_train = lr.form_all_vectors(train, model.sv)
    x_test, y_test = lr.form_all_vectors(test, model.sv)


    print("Training logreg...")

    lr.train(x_train, y_train)
    [success, failure] = lr.test(x_train, y_train)
    accuracy_train = success / (success + failure) * 100

    print("Log Reg training :")
    print(accuracy_train)

    [success, failure] = lr.test(x_test, y_test)
    accuracy_val = success / (success + failure) * 100
    
    print("Log Reg Validation:")
    print(accuracy_val)

    predictions = lr.predict(x_test)

    # print(predictions[:10])

##    o = DictWriter(open("data/testpredictions.csv", 'w'), ["id", "correctAnswer"])
##    o.writeheader()
##    for ii, pp in zip([x['id'] for x in test], predictions):
##        d = {'id': ii, 'correctAnswer': pp}
##        o.writerow(d)
