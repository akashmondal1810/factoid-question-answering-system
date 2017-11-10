from csv import DictReader, DictWriter
import numpy as np
import sklearn
from numpy import array
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from word2vec_gen import *


kTARGET_FIELD = 'correctAnswer'
kTEXT_FIELD = 'question'
kID_FIELD = 'id'
kA = 'answerA'
kB = 'answerB'
kC = 'answerC'
kD = 'answerD'


def formVector(data):
    # lablelist = ["A", "B", "C", "D"]w
    x_train = []
    y_train = []
    for x in data:
        # form the large array
        v = np.hstack((model.sv(x[kTEXT_FIELD]), model.sv(x[kA])))
        v = np.hstack((v, model.sv(x[kB])))
        v = np.hstack((v, model.sv(x[kC])))
        v = np.hstack((v, model.sv(x[kD])))

        y = x[kTARGET_FIELD]

        x_train.append(v)
        y_train.append(y)

    # x_train = [
    #     for i in lablelist for x in data]
    # # print("new total x_train length", x_train.shape[0])
    # y_train = array(list(1 if x["answer" + x[kTARGET_FIELD]] == x["answer" + i] else 0
    #                      for i in lablelist for x in data))
    return x_train, y_train


if __name__ == "__main__":

    n = 0.7  # train validation split

    train = list(DictReader(open("data/sci_train.csv", 'r')))
    # test = list(DictReader(open("data/sci_test.csv", 'r')))
    # sample = list(DictReader(open("data/sci_sample.csv", 'r')))
    train = shuffle(train)

    print("Total length: ", len(train))
    test = train[-int(len(train) * (1.0 - n)):]
    train = train[:int(len(train) * n)]

    # print("Length of train: ", len(train), " test: ", len(test))
    # print("% train vs test:", len(train)/(len(train) + len(test)))

    # labels = []
    # for line in train:
    #     if not line[kTARGET_FIELD] in labels:
    #         labels.append(line[kTARGET_FIELD])

    # labels = sorted(labels)

    # test_labels = []
    # for line in test:
    #     if not line[kTARGET_FIELD] in test_labels:
    #         test_labels.append(line[kTARGET_FIELD])

    #model = w2v('data/wiki_corpus.txt')
    model = w2v()

    # Assigining 1 to correct answer and 0 to wrong answer
    x_train, y_train = formVector(train)
    # print("new total x_train length", x_train.shape[0])

    x_test, y_test = formVector(test)
    # print("new total x_test length", x_test.shape[0])

    # print(y_train)
    print("Training started...")

    lr = LogisticRegression(C=1, penalty='l2', fit_intercept=True)
    lr.fit(x_train, y_train)

    accuracy = 100.0 * \
        sklearn.metrics.accuracy_score(y_train, lr.predict(x_train))
    print("Log Reg training :")
    print(accuracy)

    accuracy_val = 100.0 * \
        sklearn.metrics.accuracy_score(y_test, lr.predict(x_test))
    print("Log Reg Validation:")
    print(accuracy_val)

    predictions = lr.predict(x_test)

    # print(predictions[:10])

    o = DictWriter(open("testpredictions.csv", 'w'), ["id", "correctAnswer"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'correctAnswer': pp}
        o.writerow(d)
