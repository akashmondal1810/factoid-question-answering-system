from csv import DictReader, DictWriter

from numpy import array
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier

kID = 'id'
kQUESTION = 'question'
kANSWER = 'correctAnswer'
kA = 'answerA'
kB = 'answerB'
kC = 'answerC'
kD = 'answerD'


class Featurizer:
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def train_feature(self, examples):
        return self.vectorizer.fit_transform(examples)

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)


if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("data/sci_train.csv", 'r')))
    test = list(DictReader(open("data/sci_test.csv", 'r')))

    feat = Featurizer()

    labels = ['A', 'B', 'C', 'D']
    print("Label set: %s" % str(labels));

    N = len(train)
    N1 = len(test)

    trainQuestions = [x[kQUESTION] for x in train[0: N]];
    trainAnswerOptions = [x[kANSWER] for x in train[0: N]];
    trainAnswers = [{'A': x[kA], 'B': x[kB], 'C': x[kC], 'D': x[kD]}
                    for x in train[0: N]];

    testQuestions = [x[kQUESTION] for x in test[0: N1]];
    testAnswers = [{'A': x[kA], 'B': x[kB], 'C': x[kC], 'D': x[kD]}
                   for x in test[0: N1]];
    testID = [x[kID] for x in test[0: N1]];

    print('Question: ', str(trainQuestions[0]), '\n', 'Answer: ', str(
        trainAnswers[0][trainAnswerOptions[0]]), '\n')

    x_train = feat.train_feature(trainQuestions)
    x_test = feat.test_feature(testQuestions)

    y_train = trainAnswerOptions

    # array(list(labels.index(x[kTARGET_FIELD]) for x in train))
    # print(len(train), len(y_train))
    # print(set(y_train))

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    # feat.show_top10(lr, labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("output/predictions.csv", 'w'),
                   ["id", "correctAnswer"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'correctAnswer': pp}
        o.writerow(d)
